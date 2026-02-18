# scripts/convert_casehold.py
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset


@dataclass
class ConvertConfig:
    dataset_id: str = "casehold/casehold"
    subset: str = "all"  # "all" or "fold_1"..."fold_10"
    splits: List[str] = None  # default set below
    out_dir: str = "."
    out_prefix: str = "casehold"

    # sampling / sizing
    seed: int = 1337
    max_pos: Optional[int] = 10000
    max_neg: Optional[int] = 10000
    neg_per_example: int = 1  # 1 gives roughly balanced when max_pos/max_neg match

    # evidence formatting
    split_evidence: bool = True
    max_evidence_segments: int = 8
    min_segment_chars: int = 40


def _safe_int(x) -> int:
    # CaseHOLD label is commonly a string class "0".."4"
    try:
        return int(x)
    except Exception:
        return int(str(x).strip())


def split_prompt_to_evidence(
    prompt: str,
    *,
    max_segments: int,
    min_chars: int,
) -> List[str]:
    """
    Turn one long citing_prompt into multiple evidence chunks.

    Strategy:
      1) split on blank lines/newlines
      2) if still too long/too few, fall back to sentence-ish split
      3) keep only reasonable-length segments
    """
    txt = (prompt or "").replace("\r\n", "\n").strip()
    if not txt:
        return []

    # 1) paragraph split
    parts = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
    parts = [p for p in parts if len(p) >= min_chars]

    # 2) if not enough segments, do a rough sentence split
    if len(parts) < 2:
        rough = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z(])", txt)
        rough = [s.strip() for s in rough if s.strip()]
        parts = [s for s in rough if len(s) >= min_chars]

    # 3) cap
    if not parts:
        return [txt]
    return parts[:max_segments]


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    cfg = ConvertConfig()
    cfg.splits = cfg.splits or ["train", "validation", "test"]

    out_dir = Path(cfg.out_dir)
    out_pos = out_dir / f"{cfg.out_prefix}_pos.jsonl"
    out_neg = out_dir / f"{cfg.out_prefix}_neg.jsonl"
    out_meta = out_dir / f"{cfg.out_prefix}_meta.json"

    rng = random.Random(cfg.seed)

    ds = load_dataset(cfg.dataset_id, cfg.subset)

    pos_rows: List[Dict[str, Any]] = []
    neg_rows: List[Dict[str, Any]] = []

    seen_examples = 0

    for split in cfg.splits:
        if split not in ds:
            continue

        # shuffle deterministically (datasets.shuffle keeps everything lazy-friendly)
        split_ds = ds[split].shuffle(seed=cfg.seed)

        for ex in split_ds:
            if cfg.max_pos is not None and len(pos_rows) >= cfg.max_pos:
                break
            if cfg.max_neg is not None and len(neg_rows) >= cfg.max_neg:
                # still allow filling pos, but typically we stop when both hit targets
                if cfg.max_pos is not None and len(pos_rows) >= cfg.max_pos:
                    break

            seen_examples += 1

            example_id = ex.get("example_id", None)
            prompt = ex.get("citing_prompt", "") or ""
            gold_idx = _safe_int(ex.get("label"))

            holdings = [
                ex.get("holding_0", ""),
                ex.get("holding_1", ""),
                ex.get("holding_2", ""),
                ex.get("holding_3", ""),
                ex.get("holding_4", ""),
            ]

            # Build evidence list
            if cfg.split_evidence:
                evidence_list = split_prompt_to_evidence(
                    prompt,
                    max_segments=cfg.max_evidence_segments,
                    min_chars=cfg.min_segment_chars,
                )
            else:
                evidence_list = [prompt.strip()] if prompt.strip() else []

            if not evidence_list:
                continue

            # POS = gold holding
            gold_holding = holdings[gold_idx] if 0 <= gold_idx < 5 else ""
            if not isinstance(gold_holding, str) or not gold_holding.strip():
                continue

            pos_rows.append({
                "id": f"{example_id}:{gold_idx}:{split}",
                "group_id": example_id,
                "split": split,
                "claim": gold_holding.strip(),
                "evidence": evidence_list,              # what your runner uses
                "evidence_texts": evidence_list,        # compatibility with older loaders
                "label": 1,                             # binary label for your pipeline
                "meta": {
                    "dataset": cfg.dataset_id,
                    "subset": cfg.subset,
                    "gold_index": gold_idx,
                    "choice_index": gold_idx,
                }
            })

            # NEG = sample incorrect holdings
            wrong = [i for i in range(5) if i != gold_idx and isinstance(holdings[i], str) and holdings[i].strip()]
            if not wrong:
                continue

            k = max(0, int(cfg.neg_per_example))
            if k > 0:
                chosen = rng.sample(wrong, k=min(k, len(wrong)))
                for j in chosen:
                    if cfg.max_neg is not None and len(neg_rows) >= cfg.max_neg:
                        break
                    neg_rows.append({
                        "id": f"{example_id}:{j}:{split}",
                        "group_id": example_id,
                        "split": split,
                        "claim": holdings[j].strip(),
                        "evidence": evidence_list,
                        "evidence_texts": evidence_list,
                        "label": 0,
                        "meta": {
                            "dataset": cfg.dataset_id,
                            "subset": cfg.subset,
                            "gold_index": gold_idx,
                            "choice_index": j,
                        }
                    })

        # stop early if both hit targets
        if (cfg.max_pos is not None and len(pos_rows) >= cfg.max_pos) and (cfg.max_neg is not None and len(neg_rows) >= cfg.max_neg):
            break

    write_jsonl(pos_rows, out_pos)
    write_jsonl(neg_rows, out_neg)

    meta = {
        "config": asdict(cfg),
        "seen_examples": seen_examples,
        "pos_rows": len(pos_rows),
        "neg_rows": len(neg_rows),
        "outputs": {
            "pos": str(out_pos),
            "neg": str(out_neg),
        }
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"✅ Wrote POS: {out_pos} ({len(pos_rows)} rows)")
    print(f"✅ Wrote NEG: {out_neg} ({len(neg_rows)} rows)")
    print(f"🧾 Meta:     {out_meta}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("Convert CaseHOLD into Certum JSONL format")
    p.add_argument("--out_dir", default=".", help="Output directory")
    p.add_argument("--subset", default="all", help="Subset: all | fold_1..fold_10")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max_pos", type=int, default=5000)
    p.add_argument("--max_neg", type=int, default=5000)
    p.add_argument("--neg_per_example", type=int, default=1)
    p.add_argument("--split_evidence", action="store_true", help="Split citing_prompt into multiple evidence chunks")
    p.add_argument("--no_split_evidence", action="store_true", help="Do NOT split prompt (single evidence string)")
    p.add_argument("--max_evidence_segments", type=int, default=8)
    p.add_argument("--min_segment_chars", type=int, default=40)

    args = p.parse_args()

    # apply args to config via globals (simple and explicit)
    # (if you want this more structured, I can refactor to pass cfg around)
    ConvertConfig.out_dir = args.out_dir
    ConvertConfig.subset = args.subset
    ConvertConfig.seed = args.seed
    ConvertConfig.max_pos = args.max_pos
    ConvertConfig.max_neg = args.max_neg
    ConvertConfig.neg_per_example = args.neg_per_example
    ConvertConfig.max_evidence_segments = args.max_evidence_segments
    ConvertConfig.min_segment_chars = args.min_segment_chars
    if args.no_split_evidence:
        ConvertConfig.split_evidence = False
    elif args.split_evidence:
        ConvertConfig.split_evidence = True

    main()
