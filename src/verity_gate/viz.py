# src/verity_gate/viz.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


def write_plots(*, out_dir: Path, energies: Dict[str, List[float]], totals: Dict[str, Dict[str, int]], cfg: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Histogram per policy
    for pol, e in energies.items():
        if not e:
            continue
        arr = np.asarray(e, dtype=np.float32)

        plt.figure(figsize=(10, 4))
        plt.hist(arr, bins=60)
        plt.title(f"Hallucination Energy Distribution — {pol}")
        plt.xlabel("H_E (0..1)")
        plt.ylabel("count")
        p = out_dir / f"energy_hist_{pol}.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        log.info("Wrote: %s", p)

    # 2) Running mean (stability across dataset order)
    for pol, e in energies.items():
        if len(e) < 50:
            continue
        arr = np.asarray(e, dtype=np.float32)
        running = np.cumsum(arr) / (np.arange(len(arr)) + 1)

        plt.figure(figsize=(10, 4))
        plt.plot(running)
        plt.title(f"Running Mean of H_E (Stability) — {pol}")
        plt.xlabel("example index")
        plt.ylabel("running mean H_E")
        p = out_dir / f"energy_running_mean_{pol}.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        log.info("Wrote: %s", p)

    # 3) Rolling “tempo chart” (windowed mean)
    window = int(cfg.get("viz", {}).get("rolling_window", 200))
    for pol, e in energies.items():
        if len(e) < window * 2:
            continue
        arr = np.asarray(e, dtype=np.float32)
        kernel = np.ones(window, dtype=np.float32) / float(window)
        roll = np.convolve(arr, kernel, mode="valid")

        plt.figure(figsize=(10, 4))
        plt.plot(roll)
        plt.title(f"Energy Tempo (rolling mean, window={window}) — {pol}")
        plt.xlabel("example index (windowed)")
        plt.ylabel("rolling mean H_E")
        p = out_dir / f"energy_tempo_{pol}.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        log.info("Wrote: %s", p)


def write_cut_list(*, out_dir: Path, results_jsonl: Path, cfg: Dict[str, Any]) -> None:
    """
    “Cut list” ranked by boredom score.
    For now: boredom_score := (1 - energy) => very “safe” / low novelty sections,
    which are often repetitive / redundant. We’ll refine this later.
    """
    out_path = out_dir / "cut_list.json"
    top_n = int(cfg.get("cuts", {}).get("top_n", 50))

    items = []
    with results_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            energy = float(row.get("energy", 1.0))
            boredom = 1.0 - energy
            items.append(
                {
                    "idx": row.get("idx"),
                    "boredom_score": boredom,
                    "energy": energy,
                    "claim": row.get("claim", "")[:300],
                    "evidence_count": row.get("evidence_count", 0),
                }
            )

    items.sort(key=lambda x: x["boredom_score"], reverse=True)
    items = items[:top_n]

    with out_path.open("w", encoding="utf-8") as w:
        json.dump(items, w, indent=2, ensure_ascii=False)

    log.info("Wrote cut list: %s (top_n=%d)", out_path, top_n)
