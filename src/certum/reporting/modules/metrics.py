"""
certum.reporting.modules.metrics
Research-grade metrics computation for Certum reporting.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

# ============================================================
# Basic Extraction
# ============================================================

def extract_energies(rows):
    return np.array([r["energy"]["value"] for r in rows], dtype=float)


def extract_verdicts(rows):
    return [r["decision"]["verdict"] for r in rows]


def extract_geometry(rows, key_path):
    """
    key_path example:
        ["energy", "geometry", "alignment", "alignment_to_sigma1"]
    """
    values = []
    for r in rows:
        cur = r
        for k in key_path:
            if k not in cur:
                cur = None
                break
            cur = cur[k]
        if cur is not None:
            values.append(cur)
    return np.array(values, dtype=float) if values else np.array([], dtype=float)


# ============================================================
# Core Metrics
# ============================================================

def compute_auc(pos_rows, neg_rows):
    pos = extract_energies(pos_rows)
    neg = extract_energies(neg_rows)

    y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    y_scores = -np.concatenate([pos, neg])  # lower energy = more positive

    return float(roc_auc_score(y_true, y_scores))


def compute_tpr_at_far(pos_rows, neg_rows, tau):
    pos = extract_energies(pos_rows)
    neg = extract_energies(neg_rows)

    tpr = float(np.mean(pos <= tau))
    far = float(np.mean(neg <= tau))

    return {
        "tpr": tpr,
        "far": far
    }


# ============================================================
# Distribution Statistics
# ============================================================

def summarize_distribution(values: np.ndarray):
    if len(values) == 0:
        return {}

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def summarize_verdicts(rows):
    verdicts = extract_verdicts(rows)

    counts = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1

    total = len(verdicts)

    return {
        "counts": counts,
        "fractions": {k: v / total for k, v in counts.items()}
    }


# ============================================================
# Geometry Averages
# ============================================================

def summarize_geometry(rows):
    alignment = extract_geometry(
        rows,
        ["energy", "geometry", "alignment", "alignment_to_sigma1"]
    )

    pr = extract_geometry(
        rows,
        ["energy", "geometry", "spectral", "participation_ratio"]
    )

    sim_margin = extract_geometry(
        rows,
        ["energy", "geometry", "similarity", "sim_margin"]
    )

    sensitivity = extract_geometry(
        rows,
        ["energy", "geometry", "robustness", "sensitivity"]
    )

    entropy_rank = extract_geometry(
        rows,
        ["energy", "geometry", "support", "entropy_rank"]
    )

    return {
        "alignment_mean": float(np.mean(alignment)) if len(alignment) else None,
        "participation_ratio_mean": float(np.mean(pr)) if len(pr) else None,
        "sim_margin_mean": float(np.mean(sim_margin)) if len(sim_margin) else None,
        "sensitivity_mean": float(np.mean(sensitivity)) if len(sensitivity) else None,
        "entropy_rank_mean": float(np.mean(entropy_rank)) if len(entropy_rank) else None,
    }
