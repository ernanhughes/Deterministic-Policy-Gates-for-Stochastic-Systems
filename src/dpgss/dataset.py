# src/dpgss/dataset.py
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Iterator, Optional
import json
import random

from .cache import FeverousCache  # Circular import guard - place in separate module


def load_examples(
    kind: str,
    path: Path,
    n: int,
    seed: int,
    *,
    cache: Optional[FeverousCache] = None,
    model: str = "",
    include_context: bool = True,
    require_complete: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load (claim, evidence-set) pairs.

    For FEVEROUS, if `cache` is provided, evidence strings are resolved from the cache DB
    and evidence embeddings are pulled from the cache (so energy is computed on real text,
    not on element-id strings).
    """
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}

    if kind == "feverous":
        rows, st = load_feverous_pairs(
            path,
            cache=cache,
            model=model,
            include_context=include_context,
            require_complete=require_complete,
        )
        stats = st

        rng.shuffle(rows)
        for r in rows:
            claim = r.get("claim")
            ev = r.get("evidence")
            if not isinstance(claim, str) or not claim.strip() or not isinstance(ev, list) or not ev:
                continue
            ev = stable_unique([str(x).strip() for x in ev if str(x).strip()])
            if not ev:
                continue
            ex = {
                "claim": claim.strip(),
                "evidence": ev,
                "label": r.get("label"),
                "id": r.get("id"),
                "evidence_set": r.get("evidence_set"),
            }
            if "evidence_vecs" in r:
                ex["evidence_vecs"] = r["evidence_vecs"]
            out.append(ex)
            if len(out) >= n:
                break
        return out, stats

    if kind == "jsonl":
        rows = list(iter_jsonl(path))
        rng.shuffle(rows)

        claim_keys = ["claim", "claim_text", "text"]
        evidence_keys = ["evidence_texts", "rationale", "rationale_texts", "evidence_sentence_texts", "evidence_text"]

        for r in rows:
            claim = _pick_first_str(r, claim_keys)
            ev = _pick_evidence_list(r, evidence_keys)
            if not claim or not ev:
                continue
            ev = stable_unique([str(x).strip() for x in ev if str(x).strip()])
            if not ev:
                continue
            out.append({"claim": claim, "evidence": ev, "label": r.get("label")})
            if len(out) >= n:
                break
        return out, stats

    raise ValueError("kind must be: feverous | jsonl")


def load_feverous_pairs(
    path: Path,
    cache: Optional[FeverousCache],
    model: str,
    include_context: bool,
    require_complete: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build claim↔evidence-set pairs.

    Each FEVEROUS example can have multiple evidence sets; we treat each set as a separate
    (claim, evidence) pair to preserve the correct mapping.
    """
    pairs: List[Dict[str, Any]] = []
    stats = {
        "claims_seen": 0,
        "evidence_sets_seen": 0,
        "evidence_sets_kept": 0,
        "evidence_sets_dropped": 0,
        "missing_ids_total": 0,
    }

    for ex in load_feverous(path):
        stats["claims_seen"] += 1
        claim = ex.get("claim", "")
        label = ex.get("label", "")
        ex_id = ex.get("id", None)

        for j, eset in enumerate(_iter_evidence_sets(ex)):
            stats["evidence_sets_seen"] += 1
            ids = required_ids_for_evidence_set(eset, include_context)

            if cache is None:
                # Fallback: use the raw ids as "evidence" (not recommended).
                pairs.append({
                    "id": ex_id,
                    "set_idx": j,
                    "label": label,
                    "claim": claim,
                    "evidence": ids,
                    "evidence_ids": ids,
                    "evidence_vecs": None,
                })
                stats["evidence_sets_kept"] += 1
                continue

            # Validate completeness and fetch texts+vecs.
            texts, vecs, missing = cache.get_texts_and_vecs(ids, model)
            if missing:
                stats["missing_ids_total"] += len(missing)
                if require_complete:
                    stats["evidence_sets_dropped"] += 1
                    continue

            # If not requiring complete, we keep what we have.
            if not texts or vecs.size == 0:
                stats["evidence_sets_dropped"] += 1
                continue

            pairs.append({
                "id": ex_id,
                "set_idx": j,
                "label": label,
                "claim": claim,
                "evidence": texts,
                "evidence_ids": ids,
                "evidence_vecs": vecs,
            })
            stats["evidence_sets_kept"] += 1

    return pairs, stats

def load_feverous(path: Path) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def _iter_evidence_sets(example: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    ev = example.get("evidence", [])
    if isinstance(ev, dict):
        yield ev
    elif isinstance(ev, list):
        for e in ev:
            if isinstance(e, dict):
                yield e


def required_ids_for_evidence_set(evidence_set: Dict[str, Any], include_context: bool) -> List[str]:
    """Return the *element_ids* needed to consider an evidence set complete.

    FEVEROUS evidence sets contain:
      - content: ["Page_sentence_0", "Page_cell_0_1_1", ...]
      - context: { content_id: ["Page_title", "Page_section_4", ...], ... }

    If include_context=True we require both the content ids and their linked context ids.
    """
    content_ids = list(evidence_set.get("content", []) or [])
    if not include_context:
        return _stable_unique([str(x) for x in content_ids])

    ctx = evidence_set.get("context", {}) or {}
    ctx_ids: List[str] = []
    for cid in content_ids:
        ctx_ids.extend(ctx.get(cid, []) or [])
    all_ids = [str(x) for x in content_ids] + [str(x) for x in ctx_ids]
    return _stable_unique(all_ids)

# -----------------------------------------------------------------------------
# IO + dataset adapters
# -----------------------------------------------------------------------------
def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def _stable_unique(xs: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out



def stable_unique(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _pick_first_str(row: dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _pick_evidence_list(row: dict, keys: List[str]) -> Optional[List[str]]:
    for k in keys:
        v = row.get(k)
        if isinstance(v, list) and v:
            out = [str(x).strip() for x in v if str(x).strip()]
            if out:
                return out
        if isinstance(v, str) and v.strip():
            return [v.strip()]
    return None
