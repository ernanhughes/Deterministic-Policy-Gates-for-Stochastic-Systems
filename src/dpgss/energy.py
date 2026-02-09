from typing import Tuple, List
import numpy as np
from scipy.linalg import svd
from .types import EnergyResult

def evaluate_claim(
    claim_vec: np.ndarray,
    evidence_vecs: np.ndarray,
    *,
    regime: str,
    top_k: int,
    rank_r: int,
) -> Tuple[EnergyResult, str, List[float]]:
    base = hallucination_energy_svd(claim_vec, evidence_vecs, top_k=top_k, rank_r=rank_r)

    # Robustness probe (same idea you had)
    probe = []
    for k in (8, 12, 20):
        kk = max(1, min(int(k), int(evidence_vecs.shape[0])))
        r = hallucination_energy_svd(claim_vec, evidence_vecs, top_k=kk, rank_r=rank_r)
        probe.append(float(r.energy))

    decision_fixed = apply_policy(base.energy, regime)
    return base, decision_fixed, probe


def apply_policy(energy: float, regime: str, delta: float = 0.10) -> str:
    tau = float(POLICIES[regime])
    if energy <= tau:
        return "accept"
    if energy <= tau + float(delta):
        return "review"
    return "reject"


POLICIES = {
    "editorial": 0.55,
    "standard": 0.45,
    "strict": 0.30,
}

def hallucination_energy_svd(
    claim_vec: np.ndarray,
    evidence_vecs: np.ndarray,
    *,
    top_k: int = 12,
    rank_r: int = 8,
    return_debug: bool = True,
) -> EnergyResult:
    if claim_vec is None or evidence_vecs is None:
        return EnergyResult(
            energy=1.0,
            explained=0.0,
            identity_error=1.0,
            topk=int(top_k),
            rank_r=int(rank_r),
            effective_rank=0,
            sv=None,
            topk_scores=None,
            used_count=0,
        )

    c = _unit_norm(np.asarray(claim_vec, dtype=np.float32))

    E = np.asarray(evidence_vecs, dtype=np.float32)
    if E.ndim != 2 or E.shape[0] == 0:
        return EnergyResult(
            energy=1.0,
            explained=0.0,
            identity_error=1.0,
            topk=int(top_k),
            rank_r=int(rank_r),
            effective_rank=0,
            sv=None,
            topk_scores=None,
            used_count=0,
        )

    E = _unit_norm_rows(E)

    U_r, S, idx = build_evidence_basis(c, E, top_k=top_k, rank_r=rank_r)

    if U_r.shape[1] == 0:
        return EnergyResult(
            energy=1.0,
            explained=0.0,
            identity_error=1.0,
            topk=int(top_k),
            rank_r=int(rank_r),
            effective_rank=0,
            sv=None,
            topk_scores=None,
            used_count=0,
        )

    proj_coords = U_r.T @ c
    explained = float(np.dot(proj_coords, proj_coords))
    energy = 1.0 - explained

    explained = max(0.0, min(1.0, explained))
    energy = max(0.0, min(1.0, energy))

    identity_error = abs(1.0 - (explained + energy))
    effective_rank = int(np.sum(S > 1e-6))

    if not return_debug:
        return EnergyResult(
            energy=energy,
            explained=explained,
            identity_error=identity_error,
            topk=int(top_k),
            rank_r=int(rank_r),
            effective_rank=effective_rank,
            sv=None,
            topk_scores=None,
            used_count=int(len(idx)),
        )

    scores = cosine_scores(c, E)
    topk_scores = scores[idx].tolist() if idx.size else []
    sv_list = S.tolist() if S.size else []

    return EnergyResult(
        energy=energy,
        explained=explained,
        identity_error=identity_error,
        topk=int(top_k),
        rank_r=int(rank_r),
        effective_rank=effective_rank,
        sv=sv_list,
        topk_scores=topk_scores,
        used_count=int(len(idx)),
    )

def cosine_scores(c: np.ndarray, E: np.ndarray) -> np.ndarray:
    # c: (d,) unit, E: (n,d) unit rows
    return (E @ c).astype(np.float32)


def build_evidence_basis(
    c: np.ndarray,
    E: np.ndarray,
    *,
    top_k: int,
    rank_r: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (U_r, S)
      - U_r: (d, r) orthonormal basis vectors (columns)
      - S: singular values
    """
    if E.size == 0:
        return np.zeros((c.shape[0], 0), dtype=np.float32), np.array([], dtype=np.float32)

    top_k = max(1, int(top_k))
    rank_r = max(1, int(rank_r))

    scores = cosine_scores(c, E)
    k = min(top_k, E.shape[0])
    idx = np.argsort(-scores)[:k]
    E_k = E[idx]  # (k,d)

    # Thin SVD: E_k = U * diag(S) * Vt
    _, S, Vt = np.linalg.svd(E_k, full_matrices=False)

    r_full = Vt.shape[0]
    r = min(rank_r, r_full)
    U_r = Vt[:r].T  # (d,r)

    return U_r.astype(np.float32), S.astype(np.float32)

def _unit_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize 1D vector. Rejects (1,d) shapes."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        if x.shape[0] == 1:
            raise ValueError(
                f"_unit_norm received 2D array {x.shape}. Expected 1D vector (d,). "
                "Fix: use embed([text])[0] not embed([text])"
            )
        raise ValueError(f"_unit_norm expects 1D vector, got 2D {x.shape}")
    elif x.ndim != 1:
        raise ValueError(f"_unit_norm expects 1D vector, got {x.ndim}D {x.shape}")
    norm = np.linalg.norm(x)
    return x / max(norm, eps)

def _unit_norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize each row of 2D array."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"_unit_norm_rows expects 2D array, got {X.ndim}D {X.shape}")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms

class HallucinationEnergyComputer:
    def __init__(self, top_k: int = 12, rank_r: int = 8):
        self.top_k = top_k
        self.rank_r = rank_r
    
    def compute(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
    ) -> EnergyResult:
        # ===== STRICT SHAPE VALIDATION =====
        claim_vec = np.asarray(claim_vec, dtype=np.float32)
        evidence_vecs = np.asarray(evidence_vecs, dtype=np.float32)
        
        # Reject (1,d) shapes for claim_vec
        if claim_vec.ndim == 2:
            if claim_vec.shape[0] == 1:
                raise ValueError(
                    f"claim_vec has shape {claim_vec.shape} but must be 1D (d,). "
                    "Fix: use embed([text])[0] instead of embed([text])"
                )
            raise ValueError(f"claim_vec must be 1D vector, got 2D {claim_vec.shape}")
        elif claim_vec.ndim != 1:
            raise ValueError(f"claim_vec must be 1D vector, got {claim_vec.ndim}D {claim_vec.shape}")
        
        # Validate evidence_vecs is 2D
        if evidence_vecs.ndim == 1:
            evidence_vecs = evidence_vecs.reshape(1, -1)
        elif evidence_vecs.ndim != 2:
            raise ValueError(f"evidence_vecs must be 2D array, got {evidence_vecs.ndim}D {evidence_vecs.shape}")
        
        # Dimension compatibility
        if claim_vec.shape[0] != evidence_vecs.shape[1]:
            raise ValueError(
                f"Dimension mismatch: claim_vec dim={claim_vec.shape[0]}, "
                f"evidence_vecs dim={evidence_vecs.shape[1]}"
            )
        # =======================================
        
        if evidence_vecs.size == 0:
            return EnergyResult(
                energy=1.0, explained=0.0, identity_error=1.0,
                topk=0, rank_r=0, effective_rank=0, used_count=0
            )
        
        c = _unit_norm(claim_vec)
        E = _unit_norm_rows(evidence_vecs)
        basis, effective_rank = self._build_evidence_basis(c, E)
        
        projected = basis.T @ c
        explained = float(np.dot(projected, projected))
        energy = 1.0 - explained
        identity_error = abs(1.0 - (explained + energy))
        
        return EnergyResult(
            energy=max(0.0, min(1.0, energy)),
            explained=max(0.0, min(1.0, explained)),
            identity_error=identity_error,
            topk=min(self.top_k, E.shape[0]),
            rank_r=self.rank_r,
            effective_rank=effective_rank,
            used_count=E.shape[0]
        )
    
    def compute_robustness_probe(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
        param_variants: List[Tuple[int, int]] = [(8, 6), (12, 8), (20, 10)]
    ) -> List[float]:
        """Compute energy under parameter variations to detect instability."""
        probes = []
        for top_k, rank_r in param_variants:
            computer = HallucinationEnergyComputer(top_k=top_k, rank_r=rank_r)
            try:
                res = computer.compute(claim_vec, evidence_vecs)
                probes.append(res.energy)
            except Exception:
                probes.append(1.0)  # Fallback on error
        return probes
    
    def _build_evidence_basis(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        if evidence_vecs.shape[0] == 0:
            return np.zeros((claim_vec.shape[0], 0), dtype=np.float32), 0
        
        sims = evidence_vecs @ claim_vec
        k = min(self.top_k, evidence_vecs.shape[0])
        idx = np.argsort(-sims)[:k]
        E_topk = evidence_vecs[idx]
        
        try:
            _, S, Vt = svd(E_topk, full_matrices=False)
        except np.linalg.LinAlgError:
            d = evidence_vecs.shape[1]
            return np.zeros((d, 0), dtype=np.float32), 0
        
        r_full = Vt.shape[0]
        r = min(self.rank_r, r_full)
        basis = Vt[:r].T
        
        effective_rank = int(np.sum(S > 1e-6))
        return basis.astype(np.float32), effective_rank