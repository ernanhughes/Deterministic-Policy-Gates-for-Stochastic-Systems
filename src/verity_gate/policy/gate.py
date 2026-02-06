# src/verity_gate/gate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Policy:
    name: str
    tau_accept: float
    tau_review: float  # review if energy <= tau_review, else reject


def load_policies(cfg: Dict) -> Dict[str, Policy]:
    # Expect cfg like:
    # policies:
    #   strict:   {accept: 0.20, review: 0.35}
    #   standard: {accept: 0.35, review: 0.55}
    #   editorial:{accept: 0.55, review: 0.75}
    p = cfg.get("policies", {})
    out: Dict[str, Policy] = {}
    for name, v in p.items():
        out[name] = Policy(
            name=name,
            tau_accept=float(v["accept"]),
            tau_review=float(v["review"]),
        )
    return out


@dataclass(frozen=True)
class GateDecision:
    decision: str  # "ACCEPT" | "REVIEW" | "REJECT"
    energy: float
    explained: float
    policy: str
    reason: str


def apply_policy_gate(energy: float, explained: float, policy: Policy) -> GateDecision:
    if energy <= policy.tau_accept:
        return GateDecision(
            decision="ACCEPT",
            energy=energy,
            explained=explained,
            policy=policy.name,
            reason=f"energy={energy:.4f} <= accept={policy.tau_accept:.4f}",
        )
    if energy <= policy.tau_review:
        return GateDecision(
            decision="REVIEW",
            energy=energy,
            explained=explained,
            policy=policy.name,
            reason=f"energy={energy:.4f} <= review={policy.tau_review:.4f}",
        )
    return GateDecision(
        decision="REJECT",
        energy=energy,
        explained=explained,
        policy=policy.name,
        reason=f"energy={energy:.4f} > review={policy.tau_review:.4f}",
    )
