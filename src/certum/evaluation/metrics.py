"""
Evaluation metrics utilities.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_auc(
    y_true,
    y_probs,
    n_bootstrap: int = 1000,
    seed: int = 42,
):
    """
    Compute bootstrap confidence interval for AUC.

    Returns:
        mean_auc,
        lower_bound,
        upper_bound
    """

    rng = np.random.RandomState(seed)
    aucs = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, len(y_true), len(y_true))

        # Skip degenerate resamples
        if len(np.unique(y_true[indices])) < 2:
            continue

        auc = roc_auc_score(y_true[indices], y_probs[indices])
        aucs.append(auc)

    if not aucs:
        raise RuntimeError("No valid bootstrap samples generated.")

    mean_auc = float(np.mean(aucs))
    lower = float(np.percentile(aucs, 2.5))
    upper = float(np.percentile(aucs, 97.5))

    return mean_auc, lower, upper
