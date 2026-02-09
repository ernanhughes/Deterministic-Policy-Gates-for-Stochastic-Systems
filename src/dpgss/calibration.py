# src/dpgss/calibration.py
from typing import List, Tuple, Dict, Any
import random
import numpy as np

from dpgss.core_energy import compute_energy_core
from dpgss.energy import HallucinationEnergyComputer
from .embedder import Embedder
from .gate import VerifiabilityGate

class AdaptiveCalibrator:
    """
    Learns percentile thresholds from data distribution.
    Replaces hand-tuned thresholds with data-calibrated policy.
    """
    
    def __init__(self, gate: 'VerifiabilityGate', embedder: Embedder):
        self.gate = gate
        self.embedder = embedder
    
    def _generate_negatives(
        self,
        claims: List[str],
        evidence_sets: List[List[str]],
        mode: str = "deranged",
        offset: int = 37,
        seed: int = 1337,
    ) -> List[Tuple[str, List[str]]]:
        """
        Generate adversarial negative samples by mismatching claims with non-corresponding evidence.
        
        Returns:
            List of (claim, mismatched_evidence) tuples where claim and evidence are deliberately mismatched.
        """
        n = len(claims)
        if n == 0:
            return []
        if n == 1:
            # Cannot create meaningful negative with single sample
            return [(claims[0], evidence_sets[0])]  # Fallback (will be filtered later)
        
        rng = random.Random(seed)
        idx = list(range(n))
        
        # ------------------------------------------------------------
        # Simple permutation modes (deranged/offset/cyclic/permute)
        # ------------------------------------------------------------
        if mode in ("deranged", "offset", "cyclic", "permute"):
            if mode == "cyclic":
                perm = [(i + 1) % n for i in idx]
            elif mode == "offset":
                off = offset % n
                perm = [(i + off) % n for i in idx]
            elif mode == "permute":
                perm = idx[:]
                rng.shuffle(perm)
            else:  # "deranged" (default)
                perm = self._derangement_indices(n, rng)
            
            # Construct mismatched pairs: claim[i] + evidence[perm[i]]
            negatives = [
                (claims[i], evidence_sets[perm[i]])
                for i in idx
                if i != perm[i]  # Skip accidental fixed points (except for permute mode)
            ]
            return negatives
        
        # ------------------------------------------------------------
        # Hard-mined negatives: select evidence most similar to claim
        # (creates hardest possible negatives for calibration)
        # ------------------------------------------------------------
        elif mode == "hard_mined":
            # Compute claim embeddings
            claim_vecs = self.embedder.embed(claims)  # (n, d)
            claim_vecs = self._unit_norm_rows(claim_vecs)
            
            # Compute evidence centroids
            centroids = []
            valid_indices = []
            for i, evidence in enumerate(evidence_sets):
                if not evidence:
                    continue
                ev_vecs = self.embedder.embed(evidence)  # (m, d)
                ev_vecs = self._unit_norm_rows(ev_vecs)
                centroid = self._unit_norm(ev_vecs.mean(axis=0))
                if np.isfinite(centroid).all():
                    centroids.append(centroid)
                    valid_indices.append(i)
            
            if len(centroids) < 2:
                # Fallback to deranged if insufficient valid evidence
                return self._generate_negatives(claims, evidence_sets, mode="deranged", seed=seed)
            
            # Compute similarity matrix: claims × evidence centroids
            centroid_mat = np.stack(centroids, axis=0)  # (k, d)
            centroid_mat = self._unit_norm_rows(centroid_mat)
            sim = claim_vecs @ centroid_mat.T  # (n, k)
            
            # For each claim, find most similar evidence that ISN'T its own
            negatives = []
            for i in range(n):
                if i not in valid_indices:
                    continue
                
                # Exclude self-similarity to force mismatch
                self_pos = valid_indices.index(i) if i in valid_indices else -1
                if self_pos >= 0:
                    sim[i, self_pos] = -np.inf
                
                # Select evidence with highest similarity (hard negative)
                best_idx = int(np.argmax(sim[i]))
                best_evidence_idx = valid_indices[best_idx]
                negatives.append((claims[i], evidence_sets[best_evidence_idx]))
            
            return negatives
        
        else:
            raise ValueError(
                f"Unknown neg_mode: '{mode}'. "
                "Valid modes: deranged, offset, cyclic, permute, hard_mined"
            )
    
    def _derangement_indices(self, n: int, rng: random.Random) -> List[int]:
        """
        Generate a true derangement (permutation with zero fixed points).
        Uses rejection sampling with Sattolo fallback for robustness.
        """
        if n <= 1:
            return list(range(n))
        
        # Rejection sampling (uniform over all derangements)
        for _ in range(1000):  # Practical upper bound
            perm = list(range(n))
            rng.shuffle(perm)
            if all(perm[i] != i for i in range(n)):
                return perm
        
        # Fallback: Sattolo's algorithm (guaranteed derangement for n>1)
        perm = list(range(n))
        for i in range(n - 1, 0, -1):
            j = rng.randrange(i)  # 0 <= j < i
            perm[i], perm[j] = perm[j], perm[i]
        return perm
    
    def _unit_norm(self, x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(x)
        return x / max(norm, eps)
    
    def _unit_norm_rows(self, X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < eps, 1.0, norms)
        return X / norms
    
    def run_sweep(
        self,
        claims: List[str],
        evidence_sets: List[List[str]],
        evidence_vecs: List[np.ndarray],
        percentiles: List[int] = [1, 5, 10, 20, 30],
        neg_mode: str = "deranged",
        neg_offset: int = 37,
        seed: int = 1337,
        claim_vec_cache: Dict[str, np.ndarray] = None,
    ) -> Dict[str, Any]:
        # 1. Compute energies on POSITIVE samples (claim + correct evidence)
        pos_energies = []
        for claim, ev in zip(claims, evidence_vecs):
            energy = compute_energy_core(
                    claim_vec=self.embedder.embed([claim])[0],
                    evidence_vecs=ev,
                    top_k=self.gate.energy_computer.top_k,
                    rank_r=self.gate.energy_computer.rank_r,
                ).energy
            pos_energies.append(energy)
        
        # 2. Generate NEGATIVE samples via adversarial transformation
        neg_samples = self._generate_negatives(
            claims, evidence_sets,
            mode=neg_mode,
            offset=neg_offset,
            seed=seed,
        )
        
        # 3. Compute energies on NEGATIVE samples (claim + mismatched evidence)
        neg_energies = []

        for claim_text, evidence_texts in neg_samples:
            if not evidence_texts:
                continue

            # ---- CLAIM VECTOR (cached) ----
            if claim_text in claim_vec_cache:
                claim_vec = claim_vec_cache[claim_text]
            else:
                claim_vec = self.embedder.embed([claim_text])[0]
                claim_vec_cache[claim_text] = claim_vec

            # ---- EVIDENCE VECTORS (CRITICAL) ----
            # DO NOT re-embed if vectors already exist
            if isinstance(evidence_texts, np.ndarray):
                evidence_vecs = evidence_texts
            else:
                evidence_vecs = self.embedder.embed(evidence_texts)

            # ---- ENERGY COMPUTATION ----
            energy = compute_energy_from_vectors(
                claim_vec=claim_vec,
                evidence_vecs=evidence_vecs,
                energy_computer=self.gate.energy_computer,
            )

            neg_energies.append(float(energy))
        
        if len(neg_energies) < 10:
            raise ValueError(
                f"Insufficient negative samples for calibration ({len(neg_energies)}). "
                "Check evidence quality and negative generation mode."
            )
        
        # 4. Calibrate thresholds DIRECTLY from negative distribution
        tau_by_percentile = {
            p: float(np.percentile(neg_energies, p))
            for p in percentiles
        }
        
        # 5. Compute separation metric (critical for validation)
        separation_delta = (
            np.mean(neg_energies) - np.mean(pos_energies)
            if pos_energies and neg_energies else 0.0
        )
        
        return {
            "tau_by_percentile": tau_by_percentile,
            "pos_energies": pos_energies,
            "neg_energies": neg_energies,
            "separation_delta": separation_delta,
            "sample_count": len(pos_energies),
            "neg_sample_count": len(neg_energies),
            "neg_mode": neg_mode,
        }
    
def compute_energy_from_vectors(
    claim_vec: np.ndarray,
    evidence_vecs: np.ndarray,
    energy_computer: HallucinationEnergyComputer,
) -> float:
    res = energy_computer.compute(
        claim_vec=claim_vec,
        evidence_vecs=evidence_vecs,
    )
    return res.energy
