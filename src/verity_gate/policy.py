# src/verity_gate/policy.py
POLICIES = {
    "editorial": 0.55,
    "standard": 0.45,
    "strict": 0.30,
}

ADAPTIVE_POLICIES = {
    "editorial": {
        "gap_accept": 0.20,
        "gap_review": 0.35,
    },
    "standard": {
        "gap_accept": 0.08,
        "gap_review": 0.18,
    },
    "strict": {
        "gap_accept": 0.06,
        "gap_review": 0.12,
    },
}


def apply_policy(energy: float, regime: str, delta: float = 0.10) -> str:
    tau = POLICIES[regime]
    if energy <= tau:
        return "accept"
    if energy <= tau + delta:
        return "review"
    return "reject"


def apply_adaptive_policy(
    energy: float,
    oracle_energy: float,
    regime: str,
) -> str:
    """
    Adaptive policy:
    Decision is based on *excess semantic energy*
    beyond what evidence itself requires.
    """
    if oracle_energy is None:
        return "reject"

    gap = energy - oracle_energy
    cfg = ADAPTIVE_POLICIES[regime]

    if gap <= cfg["gap_accept"]:
        return "accept"
    if gap <= cfg["gap_review"]:
        return "review"
    return "reject"

