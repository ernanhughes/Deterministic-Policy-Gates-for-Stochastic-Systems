import json
from pathlib import Path
import pandas as pd

RUNS = {
    "wiki": Path("artifacts/runs/20260216_221025"),
    "pubmed": Path("artifacts/runs/20260216_220953"),
    "casehold": Path("artifacts/runs/20260216_222353"),
}

def load_report(path: Path):
    with open(path / "report.json", "r", encoding="utf-8") as f:
        return json.load(f)

def summarize_dataset(name, run_path):
    report = load_report(run_path)

    pos = report["positive_summary"]
    neg = report["negative_summary"]
    cal = report.get("calibration", {})
    neg_meta = report.get("neg_meta", {})

    pos_total = pos["total_samples"]
    neg_total = neg["total_samples"]

    pos_accept = pos["verdict_distribution"]["accept"]
    neg_accept = neg["verdict_distribution"]["accept"]

    pos_mean_energy = pos["energy_stats"]["mean"]
    neg_mean_energy = neg["energy_stats"]["mean"]

    pos_std = pos["energy_stats"]["std"]

    tpr = pos_accept / pos_total
    far = neg_accept / neg_total

    delta_mean = neg_mean_energy - pos_mean_energy
    separation_ratio = delta_mean / pos_std if pos_std else None

    return {
        "dataset": name,
        "neg_mode": neg_meta.get("mode"),
        "tau_energy": cal.get("tau_energy"),
        "TPR_eval": round(tpr, 4),
        "FAR_eval": round(far, 4),
        "pos_mean_energy": round(pos_mean_energy, 4),
        "neg_mean_energy": round(neg_mean_energy, 4),
        "delta_mean_energy": round(delta_mean, 4),
        "separation_ratio": round(separation_ratio, 4) if separation_ratio else None,
    }

rows = []

for name, path in RUNS.items():
    rows.append(summarize_dataset(name, path))

df = pd.DataFrame(rows)

print("\nAdversarial Consolidated Results\n")
print(df)

df.to_csv("artifacts/adversarial_consolidated.csv", index=False)
