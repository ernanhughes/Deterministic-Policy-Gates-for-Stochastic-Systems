# src/verity_gate/run_eval.py
from pathlib import Path
import json
import numpy as np

from verity_gate.dataset import load_feverous, extract_evidence
from verity_gate.embedder import HFEmbedder
from verity_gate.gate import evaluate_claim
from verity_gate.policy import apply_policy
from verity_gate.policy_adaptive import AdaptivePolicy
from tqdm import tqdm


DATASET = Path("datasets/feverous/feverous_dev_challenges.jsonl")
OUT_PATH = Path("artifacts/results_full.jsonl")


def main():
    embedder = HFEmbedder()
    raw_results = []

    # ==================================================
    # PASS 1 — Collect energies + oracle energies
    # ==================================================
    examples = list(load_feverous(DATASET))
    for ex in tqdm(examples, desc="🔍 Pass 1: computing energies"):
        evidence = extract_evidence(ex)
        if not evidence:
            continue

        claim = ex["claim"]
        ev_vecs = embedder.embed(evidence)
        claim_vec = embedder.embed([claim])[0]

        res, decision_fixed, decision_adaptive, probe, oracle_energy, energy_gap = evaluate_claim(
            claim_vec,
            ev_vecs,
            regime="standard",
            top_k=12,
            rank_r=8,
            embedder=embedder,
            evidence_texts=evidence,
        )


        raw_results.append({
            "claim": claim,
            "res": res,
            "oracle_energy": oracle_energy,
            "energy_gap": energy_gap,
            "probe": probe,
        })

    # ==================================================
    # Fit adaptive policy from oracle distribution
    # ==================================================
    oracle_energies = np.array(
        [r["oracle_energy"] for r in raw_results if r["oracle_energy"] is not None]
    )

    oracle_mu = float(np.mean(oracle_energies))
    oracle_sigma = float(np.std(oracle_energies))

    policy_adaptive = AdaptivePolicy(oracle_mu, oracle_sigma)

    print(
        f"\n📐 Adaptive Policy Learned"
        f"\n   oracle μ = {oracle_mu:.4f}"
        f"\n   oracle σ = {oracle_sigma:.4f}"
        f"\n   accept ≤ {policy_adaptive.accept_tau:.4f}"
        f"\n   review ≤ {policy_adaptive.review_tau:.4f}\n"
    )


    # ==================================================
    # PASS 2 — Apply adaptive policy
    # ==================================================
    final_results = []

    for r in tqdm(raw_results, desc="⚖️ Pass 2: applying policies"):
        res = r["res"]
        decision_fixed = apply_policy(res.energy, "standard")
        decision_adaptive = policy_adaptive.decide(res.energy)

        final_results.append({
            "claim": r["claim"],

            # Core metrics
            "energy": res.energy,
            "coverage": res.explained,
            "oracle_energy": r["oracle_energy"],
            "energy_gap": r["energy_gap"],

            # Structure
            "effective_rank": res.effective_rank,
            "topk": res.topk,
            "rank_r": res.rank_r,

            # Robustness
            "energy_probe": r["probe"],
            "energy_probe_var": float(np.var(r["probe"])),

            # Decisions
            "decision_fixed": decision_fixed,
            "decision_adaptive": decision_adaptive,
        })

    # ==================================================
    # Write results
    # ==================================================
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in final_results:
            f.write(json.dumps(row) + "\n")

    print(f"✅ Wrote {len(final_results)} rows → {OUT_PATH}")


if __name__ == "__main__":
    main()
