from dataclasses import dataclass

@dataclass(frozen=True)
class DifficultyMetrics:
    evidence_count: int
    effective_rank: int
    sensitivity: float
    robustness_variance: float
    hard_negative_gap: float
