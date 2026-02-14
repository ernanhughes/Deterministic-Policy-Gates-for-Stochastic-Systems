import matplotlib.pyplot as plt
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


def scatter_plot(rows, x_path, y_path, out_path, title):
    x = safe_extract(rows, x_path)
    y = safe_extract(rows, y_path)

    if len(x) == 0 or len(y) == 0:
        return

    plt.figure(figsize=(5,5))
    plt.scatter(x, y, alpha=0.4, s=8)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
