from dataclasses import dataclass
from typing import Optional


@dataclass
class GateResult:
    supported: bool
    similarity: float
    hallucination_energy: float
    policy: str
    reason: Optional[str] = None
