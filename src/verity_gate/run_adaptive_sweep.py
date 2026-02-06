from pathlib import Path
import json
import random
import numpy as np
from tqdm import tqdm

from verity_gate.dataset import load_feverous, extract_evidence
from verity_gate.embedder import HFEmbedder
from verity_gate.gate import evaluate_claim
from verity_gate.policy_adaptive import apply_adaptive_policy_gap


DATASET = Path("datasets/feverous/feverous_dev_challenges.jsonl")
OUT = Path("artifacts/adaptive_sweep_100.json")

N = 1000
PERCENTILES = [1, 5, 10, 20, 30]  # adaptive frontier


def main():
    embedder = HFEmbedder()

    # ----------------------------
    # Sample dataset
    # ----------------------------
    all_rows = list(load_feverous(DATASET))
    sample = random.sample(all_rows, N)

    print(f"🔬 Running adaptive sweep on {N} samples")

    records = []
    gaps = []

    # ----------------------------
    # Pass 1: compute energies
    # ----------------------------
    for ex in tqdm(sample, desc="Computing energies"):
        evidence = extract_evidence(ex)
        if not evidence:
            continue

        claim = ex["claim"]

        ev_vecs = embedder.embed(evidence)
        claim_vec = embedder.embed([claim])[0]

        base, _, _, _, oracle_energy, energy_gap = evaluate_claim(
            claim_vec,
            ev_vecs,
            regime="standard",  # ignored here
            top_k=12,
            rank_r=8,
            embedder=embedder,
            evidence_texts=evidence,
        )

        if energy_gap is None:
            continue

        row = {
            "claim": claim,
            "energy": base.energy,
            "oracle_energy": oracle_energy,
            "energy_gap": energy_gap,
            "decisions": {},
        }

        records.append(row)
        gaps.append(energy_gap)

    gaps = np.asarray(gaps, dtype=np.float32)

    # ----------------------------
    # Learn adaptive thresholds
    # ----------------------------
    tau_by_percentile = {
        p: float(np.percentile(gaps, p))
        for p in PERCENTILES
    }

    print("📐 Learned adaptive thresholds:")
    for p, tau in tau_by_percentile.items():
        print(f"  P{p:>2}% → τ = {tau:.4f}")

    # ----------------------------
    # Pass 2: apply adaptive policy
    # ----------------------------
    for row in records:
        for p, tau in tau_by_percentile.items():
            decision = apply_adaptive_policy_gap(
                energy_gap=row["energy_gap"],
                tau_accept=tau,
            )
            row["decisions"][f"P{p}"] = decision

    # ----------------------------
    # Summary counts
    # ----------------------------
    summary = {}
    for p in PERCENTILES:
        key = f"P{p}"
        summary[key] = {
            "accept": sum(r["decisions"][key] == "accept" for r in records),
            "review": sum(r["decisions"][key] == "review" for r in records),
            "reject": sum(r["decisions"][key] == "reject" for r in records),
        }

    # ----------------------------
    # Write output
    # ----------------------------
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(
            {
                "percentiles": PERCENTILES,
                "tau_by_percentile": tau_by_percentile,
                "summary": summary,
                "samples": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("✅ Sweep complete")
    print("📊 Acceptance by percentile:")
    for k, v in summary.items():
        print(f"  {k:<4} → accept={v['accept']:>3}")

    print(f"📝 Wrote → {OUT}")


if __name__ == "__main__":
    main()
