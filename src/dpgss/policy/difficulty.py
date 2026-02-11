from __future__ import annotations
from typing import Dict

from dpgss.policy.difficulty_metrics import DifficultyMetrics

DEFAULT_WEIGHTS = {
    "evidence_sparsity": 0.25,
    "rank_collapse": 0.25,
    "sensitivity": 0.20,
    "robustness": 0.15,
    "hard_negative_gap": 0.15,
}

DEFAULT_THRESHOLDS = {
    "low": 0.40,
    "high": 0.75,
}

class DifficultyIndex:
    """
    Policy-level difficulty index.

    Combines normalized, interpretable difficulty signals into a scalar ∈ [0,1].
    All weights are policy parameters (not learned model parameters).
    """

    def __init__(self, cfg: Dict):
        # Load config safely with defaults
        diff_cfg = cfg.get("difficulty", {})

        self.weights = {**DEFAULT_WEIGHTS, **diff_cfg.get("weights", {})}
        self.thresholds = {**DEFAULT_THRESHOLDS, **diff_cfg.get("thresholds", {})}

        # Normalize weights defensively (sum → 1.0)
        total = sum(self.weights.values())
        if total <= 0:
            self.weights = DEFAULT_WEIGHTS.copy()
            total = sum(self.weights.values())

        for k in self.weights:
            self.weights[k] /= total

    def compute(self, m: DifficultyMetrics) -> float:
        """
        Compute difficulty index ∈ [0,1].
        """

        score = 0.0
        w = self.weights

        # (1) Evidence sparsity
        score += w["evidence_sparsity"] * (1.0 if m.evidence_count <= 1 else 0.0)

        # (2) Structural collapse (effective rank)
        if m.effective_rank > 0 and m.evidence_count > 0:
            rank_ratio = m.effective_rank / max(2, m.evidence_count)
            score += w["rank_collapse"] * (1.0 - min(1.0, rank_ratio))
        else:
            score += w["rank_collapse"]

        # (3) Brittleness under leave-one-out
        score += w["sensitivity"] * min(1.0, max(0.0, m.sensitivity))

        # (4) Robustness under parameter perturbation
        score += w["robustness"] * min(1.0, max(0.0, m.robustness_variance))

        # (5) Adversarial regime indicator (gap collapse)
        gap = max(0.0, m.hard_negative_gap)
        gap_term = 1.0 - min(1.0, gap)
        score += w["hard_negative_gap"] * gap_term

        return float(min(1.0, max(0.0, score)))
