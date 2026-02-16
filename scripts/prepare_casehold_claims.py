#!/usr/bin/env python3

"""
Convert CaseHOLD into a unified Certum-compatible JSONL file.

Output format (same as HaluEval extractor):

{
    "id": str,
    "group_id": str,
    "split": str,
    "claim": str,
    "evidence": [str],
    "label": 0 or 1   # 1 = supported (gold), 0 = unsupported
}

This produces a SINGLE file containing both positive and negative rows.
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset


# =========================================================
# Evidence Splitting
# =========================================================

def split_prompt_to_evidence(
    prompt: str,
    *,
    max_segments: int,
    min_chars: int,
) -> List[str]:
    """
    Split long legal prompt into multiple evidence segments.

    Strategy:
      1) Paragraph split
      2) Fallback sentence-like split
      3) Cap segments
    """

    txt = (prompt or "").replace("\r\n", "\n").strip()
    if not txt:
        return []

    # Paragraph split
    parts = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
    parts = [p for p in parts if len(p) >= min_chars]

    # Fallback sentence split
    if len(parts) < 2:
        rough = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z(])", txt)
        rough = [s.strip() for s in rough if s.strip()]
        parts = [s for s in rough if len(s) >= min_chars]

    if not parts:
        return [txt]

    return parts[:max_segments]


# =========================================================
# Main Conversion Logic
# =========================================================

def main():
    ap = argparse.ArgumentParser("Prepare CaseHOLD unified dataset")

    ap.add_argument("--subset", default="all",
                    help="Subset: all | fold_1..fold_10")

    ap.add_argument("--splits", nargs="+",
                    default=["train", "validation", "test"])

    ap.add_argument("--out", default="casehold_claims.jsonl",
                    help="Output JSONL file")

    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--max_rows", type=int, default=None,
                    help="Optional cap on total rows")

    ap.add_argument("--neg_per_example", type=int, default=1,
                    help="How many incorrect holdings per example")

    ap.add_argument("--split_evidence", action="store_true",
                    help="Split citing_prompt into segments")

    ap.add_argument("--max_evidence_segments", type=int, default=8)

    ap.add_argument("--min_segment_chars", type=int, default=40)

    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("Loading CaseHOLD...")
    ds = load_dataset("casehold/casehold", args.subset)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0

    with out_path.open("w", encoding="utf-8") as f:

        for split in args.splits:

            if split not in ds:
                continue

            split_ds = ds[split].shuffle(seed=args.seed)

            for ex in split_ds:

                if args.max_rows and total_written >= args.max_rows:
                    break

                example_id = ex.get("example_id")
                prompt = ex.get("citing_prompt", "") or ""

                try:
                    gold_idx = int(ex.get("label"))
                except Exception:
                    gold_idx = int(str(ex.get("label")).strip())

                holdings = [
                    ex.get("holding_0", ""),
                    ex.get("holding_1", ""),
                    ex.get("holding_2", ""),
                    ex.get("holding_3", ""),
                    ex.get("holding_4", ""),
                ]

                # Build evidence list
                if args.split_evidence:
                    evidence_list = split_prompt_to_evidence(
                        prompt,
                        max_segments=args.max_evidence_segments,
                        min_chars=args.min_segment_chars,
                    )
                else:
                    evidence_list = [prompt.strip()] if prompt.strip() else []

                if not evidence_list:
                    continue

                # -----------------------------------------------------
                # 1️⃣ Supported (Gold)
                # -----------------------------------------------------

                gold_holding = holdings[gold_idx]
                if isinstance(gold_holding, str) and gold_holding.strip():

                    row = {
                        "id": f"{example_id}:{gold_idx}:{split}",
                        "group_id": example_id,
                        "split": split,
                        "claim": gold_holding.strip(),
                        "evidence": evidence_list,
                        "label": 1,
                    }

                    f.write(json.dumps(row) + "\n")
                    total_written += 1

                # -----------------------------------------------------
                # 2️⃣ Unsupported (Wrong Holdings)
                # -----------------------------------------------------

                wrong_indices = [
                    i for i in range(5)
                    if i != gold_idx
                    and isinstance(holdings[i], str)
                    and holdings[i].strip()
                ]

                if not wrong_indices:
                    continue

                k = min(args.neg_per_example, len(wrong_indices))
                chosen = rng.sample(wrong_indices, k=k)

                for j in chosen:

                    if args.max_rows and total_written >= args.max_rows:
                        break

                    row = {
                        "id": f"{example_id}:{j}:{split}",
                        "group_id": example_id,
                        "split": split,
                        "claim": holdings[j].strip(),
                        "evidence": evidence_list,
                        "label": 0,
                    }

                    f.write(json.dumps(row) + "\n")
                    total_written += 1

            if args.max_rows and total_written >= args.max_rows:
                break

    print(f"✅ Wrote {total_written} rows to {out_path}")


if __name__ == "__main__":
    main()
