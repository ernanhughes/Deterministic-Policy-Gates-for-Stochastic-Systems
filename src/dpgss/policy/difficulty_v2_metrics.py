from dataclasses import dataclass

@dataclass(frozen=True)
class DifficultyV2Metrics:
    sim_margin: float
    sensitivity: float
    evidence_count: int  
    effective_rank: int  
