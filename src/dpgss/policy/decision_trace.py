from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class DecisionTrace:
    """
    Deterministic explanation of a policy decision.

    A structured record of the policy state at decision time.
    """

    # Core signals
    energy: float
    difficulty: float
    effectiveness: float

    # Policy thresholds
    tau_accept: Optional[float]
    tau_review: Optional[float]
    margin_band: Optional[float]

    # Policy metadata
    policy_name: str
    hard_negative_gap: float

    # Final action
    verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


def why_rejected(trace: DecisionTrace) -> str:
    """
    Deterministic explanation for REJECT verdicts.
    """

    if trace.verdict != "reject":
        return "Not rejected."

    reasons = []

    if trace.tau_review is not None and trace.energy > trace.tau_review:
        reasons.append("Energy exceeds review threshold.")

    if trace.difficulty > 0.75:
        reasons.append("Difficulty exceeds hard threshold.")

    if trace.effectiveness < 0.05:
        reasons.append("Insufficient effectiveness margin.")

    if not reasons:
        reasons.append("Rejected by policy fallback.")

    return " | ".join(reasons)

def why_reviewed(trace: DecisionTrace) -> str:
    if trace.verdict != "review":
        return "Not reviewed."

    reasons = []

    if trace.margin_band and abs(trace.energy - trace.tau_accept) <= trace.margin_band:
        reasons.append("Within policy margin band.")

    if trace.difficulty > 0.4:
        reasons.append("Moderate difficulty region.")

    if trace.effectiveness < 0.25:
        reasons.append("Low effectiveness margin.")

    return " | ".join(reasons)
