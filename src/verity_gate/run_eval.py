# src/verity_gate/run_eval.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .dataset_feverous import iter_feverous_jsonl
from .embedding.hf_embedder import HFEmbedder
from .energy.hallucination import hallucination_energy_svd
from .policy.gate import apply_policy_gate, load_policies
from .viz import write_plots, write_cut_list

log = logging.getLogger(__name__)


def analyze_blog_like_dataset(
    *,
    dataset_path: Path,
    out_dir: Path,
    cfg: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = load_policies(cfg)
    embed_cfg = cfg["embedding"]
    eval_cfg = cfg["eval"]
    energy_cfg = cfg["energy"]

    embedder = HFEmbedder(model_name=embed_cfg["model_name"], device=embed_cfg.get("device", "cpu"))
    log.info("Embedder ready: %s", embed_cfg["model_name"])

    max_rows = eval_cfg.get("max_rows")  # None => full dataset
    max_evidence_sentences = int(eval_cfg.get("max_evidence_sentences", 64))

    top_k = int(energy_cfg.get("top_k", 12))
    rank_r = int(energy_cfg.get("rank_r", 8))

    results_path = out_dir / "results.jsonl"
    log.info("Writing per-example results to: %s", results_path)

    totals = {p: {"ACCEPT": 0, "REVIEW": 0, "REJECT": 0, "n": 0} for p in policies.keys()}
    energies = {p: [] for p in policies.keys()}

    with results_path.open("w", encoding="utf-8") as w:
        for ex in iter_feverous_jsonl(dataset_path, max_rows=max_rows, max_evidence_sentences=max_evidence_sentences):
            # If evidence is empty, we still run: energy should be 1.0 => reject for stricter policies.
            claim_vec = embedder.embed_one(ex.claim)
            ev_vecs = embedder.embed_many(ex.evidence_sentences) if ex.evidence_sentences else np.zeros((0, embedder.dim), dtype=np.float32)

            er = hallucination_energy_svd(
                claim_vec,
                ev_vecs,
                top_k=top_k,
                rank_r=rank_r,
                return_debug=True,
            )

            row_out = {
                "idx": ex.idx,
                "claim": ex.claim,
                "evidence_count": len(ex.evidence_sentences),
                "energy": er.energy,
                "explained": er.explained,
                "topk_used": er.used_count,
                "rank_r": er.rank_r,
                "sv": er.sv[:10] if er.sv else [],
                "topk_scores": er.topk_scores[:10] if er.topk_scores else [],
                "decisions": {},
            }

            for name, pol in policies.items():
                d = apply_policy_gate(er.energy, er.explained, pol)
                row_out["decisions"][name] = {
                    "decision": d.decision,
                    "reason": d.reason,
                }
                totals[name][d.decision] += 1
                totals[name]["n"] += 1
                energies[name].append(er.energy)

            w.write(json.dumps(row_out, ensure_ascii=False) + "\n")

            if ex.idx % 250 == 0 and ex.idx > 0:
                _print_summary(totals, energies, every="checkpoint")

    _print_summary(totals, energies, every="final")

    # Write plots + cut list
    write_plots(out_dir=out_dir, energies=energies, totals=totals, cfg=cfg)
    write_cut_list(out_dir=out_dir, results_jsonl=results_path, cfg=cfg)


def _print_summary(totals: Dict[str, Dict[str, int]], energies: Dict[str, List[float]], every: str) -> None:
    lines = []
    lines.append("")
    lines.append(f"=== Gate Summary ({every}) ===")
    for pol, t in totals.items():
        n = max(1, t["n"])
        accept = t["ACCEPT"]
        review = t["REVIEW"]
        reject = t["REJECT"]

        e = energies[pol]
        if e:
            e_np = np.asarray(e, dtype=np.float32)
            stats = f"energy: mean={float(e_np.mean()):.4f}  p50={float(np.percentile(e_np, 50)):.4f}  p90={float(np.percentile(e_np, 90)):.4f}"
        else:
            stats = "energy: (no data)"

        lines.append(
            f"[{pol}] n={n}  ACCEPT={accept} ({accept/n:.1%})  REVIEW={review} ({review/n:.1%})  REJECT={reject} ({reject/n:.1%})  | {stats}"
        )
    log.info("\n".join(lines))


def main():
    import argparse
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/experiment.yaml")
    ap.add_argument("--out", type=str, default="artifacts/run_full")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    analyze_blog_like_dataset(
        dataset_path=Path(args.dataset),
        out_dir=Path(args.out),
        cfg=cfg,
    )

if __name__ == "__main__":
    main()
