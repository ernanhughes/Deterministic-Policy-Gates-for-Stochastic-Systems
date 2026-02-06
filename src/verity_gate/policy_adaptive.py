from dataclasses import dataclass

@dataclass
class AdaptivePolicy:
    oracle_mu: float
    oracle_sigma: float
    k_accept: float = 2.0
    k_review: float = 4.0

    @property
    def accept_tau(self) -> float:
        return self.oracle_mu + self.k_accept * self.oracle_sigma

    @property
    def review_tau(self) -> float:
        return self.oracle_mu + self.k_review * self.oracle_sigma

    def decide(self, energy: float) -> str:
        if energy <= self.accept_tau:
            return "accept"
        if energy <= self.review_tau:
            return "review"
        return "reject"

def apply_adaptive_policy_gap(
    *,
    energy_gap: float,
    tau_accept: float,
    tau_review: float | None = None,
) -> str:
    """
    Tunable adaptive policy for analysis / frontier mapping.
    """
    if energy_gap <= tau_accept: 
        return "accept"

    if tau_review is not None and energy_gap <= tau_review:
        return "review"

    return "reject"
