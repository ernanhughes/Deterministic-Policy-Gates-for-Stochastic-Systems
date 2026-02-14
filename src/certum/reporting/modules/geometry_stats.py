"""
Advanced geometric and diagnostic statistics
for Certum reporting.
"""

import numpy as np

# ============================================================
# Safe extractor
# ============================================================

def safe_extract(rows, path):
    values = []
    for r in rows:
        cur = r
        for k in path:
            if k not in cur:
                cur = None
                break
            cur = cur[k]
        if cur is not None:
            values.append(cur)
    return np.array(values, dtype=float) if values else np.array([], dtype=float)


# ============================================================
# Correlation helpers
# ============================================================

def correlation(x, y):
    if len(x) == 0 or len(y) == 0:
        return None
    if len(x) != len(y):
        return None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


# ============================================================
# Energy vs Participation
# ============================================================

def energy_vs_participation(rows):
    energy = safe_extract(rows, ["energy", "value"])
    pr = safe_extract(rows, ["energy", "geometry", "spectral", "participation_ratio"])

    return correlation(energy, pr)


# ============================================================
# Alignment vs Correctness
# ============================================================

def alignment_vs_correctness(rows):
    alignment = safe_extract(
        rows,
        ["energy", "geometry", "alignment", "alignment_to_sigma1"]
    )

    verdicts = []
    for r in rows:
        verdict = r["decision"]["verdict"]
        verdicts.append(1 if verdict == "accept" else 0)

    verdicts = np.array(verdicts, dtype=float)

    return correlation(alignment, verdicts)


# ============================================================
# Hard-negative gap distribution
# ============================================================

def hard_negative_gap_distribution(rows):
    gaps = safe_extract(rows, ["decision", "trace", "hard_negative_gap"])
    if len(gaps) == 0:
        return None

    return {
        "mean": float(np.mean(gaps)),
        "std": float(np.std(gaps)),
        "min": float(np.min(gaps)),
        "max": float(np.max(gaps)),
    }


# ============================================================
# Stability fraction
# ============================================================

def stability_fraction(rows):
    stable = 0
    total = len(rows)

    for r in rows:
        if r.get("stability", {}).get("is_stable", False):
            stable += 1

    if total == 0:
        return None

    return float(stable / total)


# ============================================================
# Effectiveness mean
# ============================================================

def effectiveness_mean(rows):
    eff = safe_extract(rows, ["effectiveness"])
    if len(eff) == 0:
        return None
    return float(np.mean(eff))


def hard_negative_gap_per_row(deranged_rows, hard_rows):

    hard_map = {
        r["claim"]: r["energy"]["value"]
        for r in hard_rows
    }

    gaps = []

    for r in deranged_rows:
        claim = r["claim"]
        if claim in hard_map:
            e_der = r["energy"]["value"]
            e_hard = hard_map[claim]
            gaps.append(e_hard - e_der)

    if not gaps:
        return {"mean": None, "n": 0}

    gaps = np.array(gaps)

    return {
        "mean": float(np.mean(gaps)),
        "std": float(np.std(gaps)),
        "p90": float(np.percentile(gaps, 90)),
        "positive_fraction": float(np.mean(gaps > 0)),
        "n": len(gaps)
    }
