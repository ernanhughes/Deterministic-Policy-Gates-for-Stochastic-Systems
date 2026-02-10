from typing import List, Tuple
import numpy as np
from .energy import HallucinationEnergyComputer
from .custom_types import EnergyResult

def compute_energy_core(
    *,
    claim_vec: np.ndarray,
    evidence_vecs: np.ndarray,
    top_k: int,
    rank_r: int,
) -> EnergyResult:
    """
    SEMANTIC LOCK:
    This function must remain equivalent to hallucination_energy_svd.
    No embedding. No text. No policy. No side effects.
    """
    computer = HallucinationEnergyComputer(top_k=top_k, rank_r=rank_r)
    return computer.compute(claim_vec, evidence_vecs)

def evaluate_claim_compat(
    *,
    claim_vec: np.ndarray,
    evidence_vecs: np.ndarray,
    regime: str,
    top_k: int,
    rank_r: int,
) -> Tuple[EnergyResult, str, List[float]]:

    # Base energy
    base = compute_energy_core(
        claim_vec=claim_vec,
        evidence_vecs=evidence_vecs,
        top_k=top_k,
        rank_r=rank_r,
    )

    # Robustness probe (IDENTICAL semantics)
    probe = []
    for k in (8, 12, 20):
        kk = max(1, min(k, evidence_vecs.shape[0]))
        r = compute_energy_core(
            claim_vec=claim_vec,
            evidence_vecs=evidence_vecs,
            top_k=kk,
            rank_r=rank_r,
        )
        probe.append(float(r.energy))

    decision_fixed = apply_policy(base.energy, regime)
    return base, decision_fixed, probe

def compute_energy_for_pair_compat(
    ex: dict,
    *,
    embedder,
    claim_cache: dict,
    regime: str,
    top_k: int,
    rank_r: int,
):
    c = ex["claim"]

    # ---- CLAIM VECTOR (cached, authoritative) ----
    claim_vec = claim_cache.get(c)
    if claim_vec is None:
        claim_vec = embedder.embed([c])[0]
        claim_cache[c] = claim_vec

    # ---- EVIDENCE VECTORS (authoritative) ----
    if "evidence_vecs" in ex and ex["evidence_vecs"] is not None:
        ev_vecs = ex["evidence_vecs"]
    else:
        ev_vecs = embedder.embed(ex["evidence"])

    base, decision_fixed, probe = evaluate_claim_compat(
        claim_vec=claim_vec,
        evidence_vecs=ev_vecs,
        regime=regime,
        top_k=min(top_k, ev_vecs.shape[0]),
        rank_r=rank_r,
    )

    return (
        float(base.energy),
        float(base.explained),
        int(base.effective_rank),
        int(ev_vecs.shape[0]),
        probe,
        decision_fixed,
    )

POLICIES = {
    "editorial": 0.55,
    "standard": 0.45,
    "strict": 0.30,
}


def apply_policy(energy: float, regime: str, delta: float = 0.10) -> str:
    tau = float(POLICIES[regime])
    if energy <= tau:
        return "accept"
    if energy <= tau + float(delta):
        return "review"
    return "reject"

