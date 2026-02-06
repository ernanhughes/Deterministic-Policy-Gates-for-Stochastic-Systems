from verity_gate.energy.hallucination import hallucination_energy_svd
from verity_gate.policy import apply_policy, apply_adaptive_policy
import numpy as np


def evaluate_claim(
    claim_vec,
    evidence_vecs,
    regime: str,
    *,
    embedder=None,
    evidence_texts=None,
    **energy_kwargs,
):
    # -------------------------------
    # 1) Base energy
    # -------------------------------
    base = hallucination_energy_svd(
        claim_vec,
        evidence_vecs,
        **energy_kwargs,
    )

    # -------------------------------
    # 2) Oracle / control energy
    # -------------------------------
    oracle_energy = None
    if embedder is not None and evidence_texts:
        control_claim = evidence_texts[0]
        control_vec = embedder.embed([control_claim])[0]

        oracle = hallucination_energy_svd(
            control_vec,
            evidence_vecs,
            **energy_kwargs,
        )
        oracle_energy = oracle.energy

    energy_gap = None
    if oracle_energy is not None:
        energy_gap = base.energy - oracle_energy

    # -------------------------------
    # 3) Robustness probe
    # -------------------------------
    probe = []
    for k in (8, 12, 20):
        r = hallucination_energy_svd(
            claim_vec,
            evidence_vecs,
            top_k=k,
            rank_r=energy_kwargs.get("rank_r", 8),
            return_debug=False,
        )
        probe.append(r.energy)

    # -------------------------------
    # 4) Decisions
    # -------------------------------
    decision_fixed = apply_policy(base.energy, regime)

    decision_adaptive = apply_adaptive_policy(
        energy=base.energy,
        oracle_energy=oracle_energy,
        regime=regime,
    )

    return (
        base,
        decision_fixed,
        decision_adaptive,
        probe,
        oracle_energy,
        energy_gap,
    )
