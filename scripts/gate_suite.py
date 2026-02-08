#!/usr/bin/env python3
"""
gate_suite.py — Hallucination-energy gating with negative-control calibration (v2)

Core idea:
- Energy = 1 - ||U_r^T c||^2  where U_r is evidence subspace basis (SVD on top-k evidence vectors)
- Decision: accept if energy <= tau
- Calibrate tau on shuffled/deranged negatives to bound FAR (false-accept rate)
- Report metrics on a holdout split to avoid "tuned on test"

This evaluates evidence-conditioned groundedness (curated evidence), not raw-document hallucination detection.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# src/verity_gate/dataset.py
from typing import Iterator

def load_feverous(path: Path) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def extract_evidence(example: Dict) -> list[str]:
    texts = []
    for ev in example.get("evidence", []):
        for ctx in ev.get("context", {}).values():
            texts.extend(ctx)
    return list(set(texts))


class HFEmbedder:
    # def __init__(self, model_name="all-MiniLM-L6-v2"):
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )


# -----------------------------------------------------------------------------
# Fixed (editorial) policies: your "hard gate" knobs
# -----------------------------------------------------------------------------
POLICIES = {
    "editorial": 0.55,
    "standard": 0.45,
    "strict": 0.30,
}


def apply_policy(energy: float, regime: str, delta: float = 0.10) -> str:
    tau = float(POLICIES[regime])
    if energy <= tau:
        return "accept"
    if energy <= tau + float(delta):
        return "review"
    return "reject"


# -----------------------------------------------------------------------------
# Hallucination Energy (SVD residual)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class EnergyResult:
    energy: float           # in [0,1]
    explained: float        # ||U_r^T c||^2
    identity_error: float   # |1 - (explained + energy)|
    topk: int
    rank_r: int
    effective_rank: int
    used_count: int


def _unit_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps:
        return x * 0.0
    return x / n


def _unit_norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms


def cosine_scores(c: np.ndarray, E: np.ndarray) -> np.ndarray:
    # c: (d,) unit, E: (n,d) unit rows
    return (E @ c).astype(np.float32)


def build_evidence_basis(
    c: np.ndarray,
    E: np.ndarray,
    *,
    top_k: int,
    rank_r: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (U_r, S, idx)
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

    return U_r.astype(np.float32), S.astype(np.float32), idx.astype(np.int64)


def hallucination_energy_svd(
    claim_vec: np.ndarray,
    evidence_vecs: np.ndarray,
    *,
    top_k: int = 12,
    rank_r: int = 8,
) -> EnergyResult:
    if claim_vec is None or evidence_vecs is None:
        return EnergyResult(
            energy=1.0,
            explained=0.0,
            identity_error=1.0,
            topk=int(top_k),
            rank_r=int(rank_r),
            effective_rank=0,
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
            used_count=int(len(idx)),
        )

    proj_coords = U_r.T @ c
    explained = float(np.dot(proj_coords, proj_coords))
    energy = 1.0 - explained

    # clamp
    explained = max(0.0, min(1.0, explained))
    energy = max(0.0, min(1.0, energy))

    identity_error = abs(1.0 - (explained + energy))
    effective_rank = int(np.sum(S > 1e-6))

    return EnergyResult(
        energy=energy,
        explained=explained,
        identity_error=identity_error,
        topk=int(top_k),
        rank_r=int(rank_r),
        effective_rank=effective_rank,
        used_count=int(len(idx)),
    )


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


# -----------------------------------------------------------------------------
# IO + dataset adapters
# -----------------------------------------------------------------------------
def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def stable_unique(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _pick_first_str(row: dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _pick_evidence_list(row: dict, keys: List[str]) -> Optional[List[str]]:
    for k in keys:
        v = row.get(k)
        if isinstance(v, list) and v:
            out = [str(x).strip() for x in v if str(x).strip()]
            if out:
                return out
        if isinstance(v, str) and v.strip():
            return [v.strip()]
    return None


def load_examples(kind: str, path: Path, n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []

    if kind == "feverous":
        rows = list(load_feverous(path))
        rng.shuffle(rows)
        for r in rows:
            claim = r.get("claim")
            if not isinstance(claim, str) or not claim.strip():
                continue
            ev = extract_evidence(r)
            if not ev:
                continue
            ev = stable_unique([str(x).strip() for x in ev if str(x).strip()])
            if not ev:
                continue
            out.append({"claim": claim.strip(), "evidence": ev, "label": r.get("label")})
            if len(out) >= n:
                break
        return out

    if kind == "jsonl":
        rows = list(iter_jsonl(path))
        rng.shuffle(rows)

        claim_keys = ["claim", "claim_text", "text"]
        evidence_keys = ["evidence_texts", "rationale", "rationale_texts", "evidence_sentence_texts", "evidence_text"]

        for r in rows:
            claim = _pick_first_str(r, claim_keys)
            ev = _pick_evidence_list(r, evidence_keys)
            if not claim or not ev:
                continue
            ev = stable_unique([str(x).strip() for x in ev if str(x).strip()])
            if not ev:
                continue
            out.append({"claim": claim, "evidence": ev, "label": r.get("label")})
            if len(out) >= n:
                break
        return out

    raise ValueError("kind must be: feverous | jsonl")


# -----------------------------------------------------------------------------
# Negatives
# -----------------------------------------------------------------------------
def _derangement_indices(n: int, rng: random.Random, max_tries: int = 200) -> List[int]:
    """
    Returns a permutation p of [0..n-1] with no fixed points (p[i] != i).
    Falls back to a cyclic shift (always a derangement for n>1).
    """
    if n <= 1:
        return list(range(n))

    for _ in range(max_tries):
        p = list(range(n))
        rng.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return p

    # guaranteed derangement for n>1
    return [(i + 1) % n for i in range(n)]


def make_negatives(
    pairs: List[Dict[str, Any]],
    *,
    mode: str,
    seed: int,
    offset: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (neg_pairs, neg_meta)
    """
    rng = random.Random(seed)
    n = len(pairs)

    if mode == "offset":
        off = int(offset) % max(1, n)
        if n > 1 and off == 0:
            off = 1
        neg = [{"claim": pairs[i]["claim"], "evidence": pairs[(i + off) % n]["evidence"]} for i in range(n)]
        return neg, {"mode": "offset", "offset": off, "guarantees": "no fixed points if n>1 and offset% n != 0"}

    if mode == "cyclic":
        if n <= 1:
            neg = pairs[:]
        else:
            neg = [{"claim": pairs[i]["claim"], "evidence": pairs[(i + 1) % n]["evidence"]} for i in range(n)]
        return neg, {"mode": "cyclic", "guarantees": "derangement for n>1"}

    if mode == "permute":
        # WARNING: can contain fixed points. Provided only for ablation.
        evidence_list = [p["evidence"] for p in pairs]
        rng.shuffle(evidence_list)
        neg = [{"claim": p["claim"], "evidence": ev} for p, ev in zip(pairs, evidence_list)]
        return neg, {"mode": "permute", "warning": "may include fixed points"}

    if mode == "deranged":
        p = _derangement_indices(n, rng)
        neg = [{"claim": pairs[i]["claim"], "evidence": pairs[p[i]]["evidence"]} for i in range(n)]
        return neg, {"mode": "deranged", "guarantees": "no fixed points (n>1)"}

    raise ValueError("neg_mode must be: deranged | offset | cyclic | permute")


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def summarize(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"count": 0.0}
    return {
        "count": float(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """
    Rank data with average ranks for ties. Ranks are 1..N.
    """
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")  # stable
    xs = x[order]
    ranks = np.empty_like(xs, dtype=np.float64)

    n = len(xs)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        # average rank in [i..j] with 1-indexing
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[i : j + 1] = avg_rank
        i = j + 1

    out = np.empty_like(ranks)
    out[order] = ranks
    return out


def auc_lower_is_positive(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC where *lower* energy implies positive (better evidence match).

    Implemented by negating energies and computing standard rank-based AUC where
    higher score => positive. (Ties are rare with float energies; we ignore tie averaging.)
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")

    scores = np.concatenate([-pos, -neg])  # higher => more positive
    labels = np.concatenate([np.ones_like(pos, dtype=np.int32), np.zeros_like(neg, dtype=np.int32)])

    order = np.argsort(scores)  # ascending
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)  # 1..N

    pos_ranks = ranks[labels == 1]
    n_pos = float(pos.size)
    n_neg = float(neg.size)

    # Mann–Whitney U / rank-sum AUC
    return float((pos_ranks.sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


    scores = np.concatenate([-pos, -neg])  # higher = more positive
    labels = np.concatenate([np.ones_like(pos, dtype=np.int32), np.zeros_like(neg, dtype=np.int32)])

    ranks = _rankdata_average_ties(scores)
    pos_ranks = ranks[labels == 1]

    n_pos = float(len(pos))
    n_neg = float(len(neg))
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Negative-control calibration for hallucination-energy policy gates (v2)")
    ap.add_argument("--kind", required=True, choices=["feverous", "jsonl"])
    ap.add_argument("--in_path", required=True, type=Path)

    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--regime", type=str, default="standard", choices=["standard", "strict", "editorial"])
    ap.add_argument("--delta", type=float, default=0.10)

    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--rank_r", type=int, default=8)

    ap.add_argument("--far", type=float, default=0.01, help="Target FAR on negatives for rule: accept if energy <= tau")
    ap.add_argument("--cal_frac", type=float, default=0.5, help="Fraction of examples used to calibrate tau; rest is holdout eval")

    ap.add_argument("--neg_mode", type=str, default="deranged", choices=["deranged", "offset", "cyclic", "permute"])
    ap.add_argument("--neg_offset", type=int, default=37)

    ap.add_argument("--out_report", type=Path, default=Path("artifacts/negcal_report.json"))
    ap.add_argument("--out_pos_scored", type=Path, default=None, help="Optional JSONL of scored positives (holdout split)")
    ap.add_argument("--out_neg_scored", type=Path, default=None, help="Optional JSONL of scored negatives (holdout split)")
    ap.add_argument("--plot_png", type=Path, default=None, help="Optional PNG histogram (pos vs neg, holdout split)")
    args = ap.parse_args()

    # -----------------------------
    # Load + split
    # -----------------------------
    pairs = load_examples(args.kind, args.in_path, args.n, args.seed)
    if len(pairs) < 50:
        raise RuntimeError(f"Too few usable examples ({len(pairs)}). Check input format/evidence extraction.")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    cal_frac = float(args.cal_frac)
    cal_frac = max(0.1, min(0.9, cal_frac))
    n_cal = max(1, int(len(pairs) * cal_frac))

    cal_pos = pairs[:n_cal]
    ev_pos = pairs[n_cal:]

    # -----------------------------
    # Negatives (cal + eval)
    # -----------------------------
    cal_neg, neg_meta_cal = make_negatives(
        cal_pos,
        mode=args.neg_mode,
        seed=args.seed + 100,
        offset=args.neg_offset,
    )

    ev_neg, neg_meta_ev = make_negatives(
        ev_pos if ev_pos else cal_pos,  # if no holdout (shouldn't happen), reuse
        mode=args.neg_mode,
        seed=args.seed + 200,
        offset=args.neg_offset,
    )

    # -----------------------------
    # Embedder
    # -----------------------------
    embedder = HFEmbedder()

    def compute_energy_for_pair(ex: Dict[str, Any]) -> Tuple[float, float, int, int, List[float], str]:
        claim_vec = embedder.embed([ex["claim"]])[0]
        ev_vecs = embedder.embed(ex["evidence"])
        base, decision_fixed, probe = evaluate_claim(
            claim_vec,
            ev_vecs,
            regime=args.regime,
            top_k=min(int(args.top_k), len(ex["evidence"])),
            rank_r=int(args.rank_r),
        )
        return (
            float(base.energy),
            float(base.explained),
            int(base.effective_rank),
            int(len(ex["evidence"])),
            [float(x) for x in probe],
            decision_fixed,
        )

    # -----------------------------
    # Compute energies
    # -----------------------------
    cal_pos_e = []
    cal_neg_e = []

    for ex in tqdm(cal_pos, desc="Compute CAL POS energies"):
        e, *_ = compute_energy_for_pair(ex)
        cal_pos_e.append(e)

    for ex in tqdm(cal_neg, desc="Compute CAL NEG energies"):
        e, *_ = compute_energy_for_pair(ex)
        cal_neg_e.append(e)

    cal_pos_e = np.asarray(cal_pos_e, dtype=np.float32)
    cal_neg_e = np.asarray(cal_neg_e, dtype=np.float32)

    # -----------------------------
    # Calibrate tau on CAL negatives
    # -----------------------------
    far = float(args.far)
    far = max(0.0, min(0.5, far))  # sane range
    tau_cal = float(np.percentile(cal_neg_e, far * 100.0))  # accept if energy <= tau

    cal_FAR = float((cal_neg_e <= tau_cal).mean())
    cal_TPR = float((cal_pos_e <= tau_cal).mean())
    cal_AUC = auc_lower_is_positive(cal_pos_e, cal_neg_e)

    # -----------------------------
    # Holdout eval energies
    # -----------------------------
    if ev_pos:
        ev_pos_e = []
        ev_neg_e = []

        for ex in tqdm(ev_pos, desc="Compute EVAL POS energies"):
            e, *_ = compute_energy_for_pair(ex)
            ev_pos_e.append(e)

        for ex in tqdm(ev_neg, desc="Compute EVAL NEG energies"):
            e, *_ = compute_energy_for_pair(ex)
            ev_neg_e.append(e)

        ev_pos_e = np.asarray(ev_pos_e, dtype=np.float32)
        ev_neg_e = np.asarray(ev_neg_e, dtype=np.float32)

        ev_FAR = float((ev_neg_e <= tau_cal).mean())
        ev_TPR = float((ev_pos_e <= tau_cal).mean())
        ev_AUC = auc_lower_is_positive(ev_pos_e, ev_neg_e)
    else:
        ev_pos_e = np.asarray([], dtype=np.float32)
        ev_neg_e = np.asarray([], dtype=np.float32)
        ev_FAR, ev_TPR, ev_AUC = float("nan"), float("nan"), float("nan")

    # -----------------------------
    # Hardest negatives (lowest energy mismatches)
    # -----------------------------
    hardest = []
    if ev_pos:
        # score a small sample of negatives with text for audit
        scored = []
        for i, ex in enumerate(ev_neg[: min(200, len(ev_neg))]):
            e, _, _, _, _, _ = compute_energy_for_pair(ex)
            scored.append((e, ex))
        scored.sort(key=lambda t: t[0])  # lowest energy = most dangerous negative
        for e, ex in scored[:10]:
            hardest.append({
                "energy": float(e),
                "claim": ex["claim"][:200],
                "evidence_0": ex["evidence"][0][:200] if ex["evidence"] else None,
                "evidence_len": len(ex["evidence"]),
            })

    # -----------------------------
    # Report
    # -----------------------------
    report = {
        "kind": args.kind,
        "in_path": str(args.in_path),
        "n_total": int(len(pairs)),
        "split": {
            "cal_frac": cal_frac,
            "n_cal": int(len(cal_pos)),
            "n_eval": int(len(ev_pos)),
        },
        "params": {
            "regime": args.regime,
            "delta": float(args.delta),
            "top_k": int(args.top_k),
            "rank_r": int(args.rank_r),
            "far_target": float(args.far),
            "tau_cal": float(tau_cal),
            "seed": int(args.seed),
            "neg_mode": args.neg_mode,
            "neg_offset": int(args.neg_offset),
            "neg_meta_cal": neg_meta_cal,
            "neg_meta_eval": neg_meta_ev,
        },
        "stats": {
            "cal": {
                "pos": summarize(cal_pos_e),
                "neg": summarize(cal_neg_e),
                "FAR": cal_FAR,
                "TPR": cal_TPR,
                "AUC": cal_AUC,
            },
            "eval": {
                "pos": summarize(ev_pos_e),
                "neg": summarize(ev_neg_e),
                "FAR": ev_FAR,
                "TPR": ev_TPR,
                "AUC": ev_AUC,
            },
        },
        "hardest_negatives_preview": hardest,
        "interpretation": {
            "accept_rule": "accept if energy <= tau",
            "calibration": "tau is set to the FAR-percentile of CAL negative energies",
            "what_this_tests": "evidence-conditioned groundedness (curated evidence + adversarial mismatches)",
            "what_this_does_not_test": "raw-document hallucination detection (uncurated documents)",
        },
    }

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("✅ Negative-control calibration (v2, NO ORACLE)")
    print("=" * 80)
    print(f"tau_cal (accept if energy<=tau): {tau_cal:.4f}")
    print(f"CAL  FAR/TPR/AUC: {cal_FAR:.3%} / {cal_TPR:.3%} / {cal_AUC:.3f}")
    if len(ev_pos) > 0:
        print(f"EVAL FAR/TPR/AUC: {ev_FAR:.3%} / {ev_TPR:.3%} / {ev_AUC:.3f}")
    print(f"Wrote report -> {args.out_report}")

    # -----------------------------
    # Optional: write scored JSONLs (eval split)
    # -----------------------------
    def write_scored(path: Path, data: List[Dict[str, Any]], tau: float, tag: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for ex in tqdm(data, desc=f"Write {tag} scored"):
                e, cov, erank, evlen, probe, decision_fixed = compute_energy_for_pair(ex)
                decision_tau = "accept" if e <= tau else ("review" if e <= tau + float(args.delta) else "reject")
                f.write(json.dumps({
                    "claim": ex["claim"],
                    "label": ex.get("label"),
                    "energy": float(e),
                    "coverage": float(cov),
                    "effective_rank": int(erank),
                    "evidence_len": int(evlen),
                    "probe": probe,
                    "decision_fixed": decision_fixed,
                    "decision_tau": decision_tau,
                    "tau_cal": float(tau),
                }, ensure_ascii=False) + "\n")

    if args.out_pos_scored is not None:
        write_scored(args.out_pos_scored, ev_pos if len(ev_pos) > 0 else cal_pos, tau_cal, "POS")

    if args.out_neg_scored is not None:
        write_scored(args.out_neg_scored, ev_neg if len(ev_neg) > 0 else cal_neg, tau_cal, "NEG")

    # -----------------------------
    # Optional: plot
    # -----------------------------
    if args.plot_png is not None:
        try:
            import matplotlib.pyplot as plt

            pos_plot = ev_pos_e if len(ev_pos_e) > 0 else cal_pos_e
            neg_plot = ev_neg_e if len(ev_neg_e) > 0 else cal_neg_e

            args.plot_png.parent.mkdir(parents=True, exist_ok=True)
            plt.figure()
            plt.hist(pos_plot, bins=40, alpha=0.7, label="pos (real evidence)")
            plt.hist(neg_plot, bins=40, alpha=0.7, label=f"neg ({args.neg_mode})")
            plt.axvline(tau_cal, linestyle="--", linewidth=2, label=f"tau_cal={tau_cal:.3f}")
            plt.xlabel("Hallucination Energy")
            plt.ylabel("Count")
            plt.legend()
            plt.title("Energy separation: real vs negatives (holdout eval)")
            plt.savefig(args.plot_png, dpi=160)
            plt.close()
            print(f"Wrote plot -> {args.plot_png}")
        except Exception as e:
            print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
