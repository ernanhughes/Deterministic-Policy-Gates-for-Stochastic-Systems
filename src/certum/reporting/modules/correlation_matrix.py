"""
Correlation matrix computation and axis collapse detection.
"""

import numpy as np


def safe_extract(rows, path):
    vals = []
    for r in rows:
        cur = r
        for k in path:
            if k not in cur:
                cur = None
                break
            cur = cur[k]
        if cur is not None:
            vals.append(cur)
    return np.array(vals, dtype=float)



def correlation_matrix(rows):
    """
    Compute correlation matrix over selected axes.
    Automatically drops zero-variance features.
    """

    features = {
        "energy": [],
        "participation_ratio": [],
        "alignment": [],
        "sim_margin": [],
        "sensitivity": [],
        "effectiveness": [],
    }

    for r in rows:
        try:
            features["energy"].append(r["energy"]["value"])
            features["participation_ratio"].append(
                r["energy"]["geometry"]["spectral"]["participation_ratio"]
            )
            features["alignment"].append(
                r["energy"]["geometry"]["alignment"]["alignment_to_sigma1"]
            )
            features["sim_margin"].append(
                r["energy"]["geometry"]["similarity"]["sim_margin"]
            )
            features["sensitivity"].append(
                r["energy"]["geometry"]["robustness"]["sensitivity"]
            )
            features["effectiveness"].append(r.get("effectiveness", 0.0))
        except KeyError:
            continue

    # Convert to numpy
    clean_features = {}
    dropped = []

    for k, v in features.items():
        arr = np.array(v, dtype=float)
        if len(arr) == 0:
            continue

        if np.std(arr) < 1e-8:
            dropped.append(k)
            continue

        clean_features[k] = arr

    if len(clean_features) < 2:
        return {}, dropped

    keys = list(clean_features.keys())
    mat = np.vstack([clean_features[k] for k in keys])

    corr = np.corrcoef(mat)

    result = {
        keys[i]: {
            keys[j]: float(corr[i, j])
            for j in range(len(keys))
        }
        for i in range(len(keys))
    }

    return result, dropped


def detect_axis_collapse(corr_matrix, threshold=0.85):
    collapse = []

    for a in corr_matrix:
        for b in corr_matrix[a]:
            if a == b:
                continue

            corr = corr_matrix[a][b]
            if corr is None:
                continue

            if abs(corr) >= threshold:
                collapse.append({
                    "axis_a": a,
                    "axis_b": b,
                    "correlation": corr
                })

    return collapse

def correlation_eigenvalues(corr_matrix_dict):
    import numpy as np

    if not corr_matrix_dict:
        return []

    keys = list(corr_matrix_dict.keys())
    mat = np.array([[corr_matrix_dict[r][c] for c in keys] for r in keys], dtype=float)

    if not np.isfinite(mat).all():
        return []

    eigvals = np.linalg.eigvals(mat)
    eigvals = np.real(eigvals)

    return sorted(eigvals.tolist(), reverse=True)
