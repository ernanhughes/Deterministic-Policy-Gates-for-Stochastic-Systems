# src/verity_gate/dataset_feverous.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeverousExample:
    idx: int
    claim: str
    evidence_sentences: List[str]


def _extract_evidence_texts(row: dict, *, max_sentences: int = 64) -> List[str]:
    """
    FEVEROUS dev challenges structure varies slightly depending on preprocessing.
    We try a few robust patterns. Goal: produce a list of evidence sentences/strings.
    """
    out: List[str] = []

    # Most common: row["evidence"] list -> each has "context" dict -> values are lists of strings
    ev_list = row.get("evidence") or []
    if isinstance(ev_list, list):
        for ev in ev_list:
            ctx = (ev or {}).get("context") or {}
            if isinstance(ctx, dict):
                for v in ctx.values():
                    if isinstance(v, list):
                        for s in v:
                            if isinstance(s, str) and s.strip():
                                out.append(s.strip())

    # Some formats might have "sentences" or "context" directly
    if not out:
        ctx = row.get("context")
        if isinstance(ctx, list):
            out.extend([x.strip() for x in ctx if isinstance(x, str) and x.strip()])

    # Dedup preserve order
    seen = set()
    deduped: List[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
        if len(deduped) >= max_sentences:
            break

    return deduped


def iter_feverous_jsonl(
    path: Path,
    *,
    max_rows: Optional[int] = None,
    max_evidence_sentences: int = 64,
) -> Iterator[FeverousExample]:
    log.info("Loading FEVEROUS JSONL from: %s", path)
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and count >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            claim = row.get("claim") or ""
            if not isinstance(claim, str) or not claim.strip():
                continue

            ev = _extract_evidence_texts(row, max_sentences=max_evidence_sentences)

            yield FeverousExample(
                idx=count,
                claim=claim.strip(),
                evidence_sentences=ev,
            )
            count += 1
            if count % 500 == 0:
                log.info("…loaded %d examples so far", count)

    log.info("Finished loading. Total examples iterated: %d", count)
