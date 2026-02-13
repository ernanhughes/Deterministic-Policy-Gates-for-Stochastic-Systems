#!/usr/bin/env python
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# =============================================================
# Utilities
# =============================================================

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_axes(records):
    energy = []
    pr = []
    sens = []
    sim_margin = []
    alignment = []
    effective_rank = []
    entropy_rank = []
    verdict = []

    for r in records:
        g = r["energy"]["geometry"]

        energy.append(r["energy"]["value"])
        pr.append(g["spectral"]["participation_ratio"])
        sens.append(g["robustness"]["sensitivity"])
        sim_margin.append(g["similarity"]["sim_margin"])
        alignment.append(g["alignment"]["alignment_to_sigma1"])
        effective_rank.append(g["support"]["effective_rank"])
        entropy_rank.append(g["support"]["entropy_rank"])
        verdict.append(r["decision"]["verdict"])

    return {
        "energy": np.array(energy),
        "pr": np.array(pr),
        "sens": np.array(sens),
        "sim_margin": np.array(sim_margin),
        "alignment": np.array(alignment),
        "effective_rank": np.array(effective_rank),
        "entropy_rank": np.array(entropy_rank),
        "verdict": np.array(verdict),
    }


def summarize_axis(name, pos, neg):
    print(f"\n{name.upper()}")
    print("  POS:", {"mean": float(np.mean(pos)), "std": float(np.std(pos))})
    print("  NEG:", {"mean": float(np.mean(neg)), "std": float(np.std(neg))})
    print("  Δ:", float(np.mean(neg) - np.mean(pos)))


def compute_roc_auc(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    return fpr, tpr, auc(fpr, tpr)


def print_roc(label, y_true, scores):
    fpr, tpr, roc_auc = compute_roc_auc(y_true, scores)
    tpr_at_1 = float(np.interp(0.01, fpr, tpr))
    print(f"{label} AUC: {roc_auc:.4f}")
    print(f"{label} TPR at FAR=1%: {tpr_at_1:.4f}")
    return fpr, tpr, roc_auc, tpr_at_1


# =============================================================
# Main
# =============================================================

def main():

    run_dirs = sorted(Path("artifacts/runs").glob("*"))
    if not run_dirs:
        raise RuntimeError("No runs found.")

    run_dir = run_dirs[-1]
    print(f"\n📊 Using run: {run_dir}\n")

    pos_file = list(run_dir.glob("pos_hard_mined_v2*.jsonl"))[0]
    neg_file = list(run_dir.glob("neg_hard_mined_v2*.jsonl"))[0]

    print(f"📄 POS file: {pos_file}")
    print(f"📄 NEG file: {neg_file}")

    pos_records = load_jsonl(pos_file)
    neg_records = load_jsonl(neg_file)

    pos = extract_axes(pos_records)
    neg = extract_axes(neg_records)

    # ---------------------------------------------------------
    # Axis Distributions
    # ---------------------------------------------------------

    print("\n================ AXIS DISTRIBUTIONS ================")

    summarize_axis("Energy", pos["energy"], neg["energy"])
    summarize_axis("Participation Ratio", pos["pr"], neg["pr"])
    summarize_axis("Sensitivity", pos["sens"], neg["sens"])
    summarize_axis("Sim Margin", pos["sim_margin"], neg["sim_margin"])
    summarize_axis("Alignment", pos["alignment"], neg["alignment"])

    # ---------------------------------------------------------
    # ROC Preparation
    # ---------------------------------------------------------

    y_true = np.concatenate([
        np.ones(len(pos["energy"])),
        np.zeros(len(neg["energy"]))
    ])

    # Build score arrays (higher score = more hallucinated)
    energy_scores = np.concatenate([pos["energy"], neg["energy"]])
    pr_scores = np.concatenate([pos["pr"], neg["pr"]])
    sens_scores = np.concatenate([pos["sens"], neg["sens"]])
    margin_scores = np.concatenate([-pos["sim_margin"], -neg["sim_margin"]])
    alignment_scores = np.concatenate([-pos["alignment"], -neg["alignment"]])
    eff_rank_scores = np.concatenate([-pos["effective_rank"], -neg["effective_rank"]])
    entropy_scores = np.concatenate([-pos["entropy_rank"], -neg["entropy_rank"]])

    # ---------------------------------------------------------
    # ROC ANALYSIS
    # ---------------------------------------------------------

    print("\n================ ROC ANALYSIS ======================")

    fpr_e, tpr_e, auc_e, tpr1_e = print_roc("Energy", y_true, energy_scores)
    fpr_pr, tpr_pr, auc_pr, tpr1_pr = print_roc("Participation Ratio", y_true, pr_scores)
    fpr_s, tpr_s, auc_s, tpr1_s = print_roc("Sensitivity", y_true, sens_scores)
    fpr_m, tpr_m, auc_m, tpr1_m = print_roc("Sim Margin (inverted)", y_true, margin_scores)
    fpr_a, tpr_a, auc_a, tpr1_a = print_roc("Alignment (inverted)", y_true, alignment_scores)
    fpr_er, tpr_er, auc_er, tpr1_er = print_roc("Effective Rank (inverted)", y_true, eff_rank_scores)
    fpr_ent, tpr_ent, auc_ent, tpr1_ent = print_roc("Entropy Rank (inverted)", y_true, entropy_scores)

    # ---------------------------------------------------------
    # Interaction: Energy × (1 - Margin)
    # ---------------------------------------------------------

    interaction_scores = energy_scores * (1 - np.concatenate([
        pos["sim_margin"], neg["sim_margin"]
    ]))

    fpr_i, tpr_i, auc_i, tpr1_i = print_roc("Energy × (1 - Margin)", y_true, interaction_scores)

    # ---------------------------------------------------------
    # Conditional Energy (low margin)
    # ---------------------------------------------------------

    margins_all = np.concatenate([pos["sim_margin"], neg["sim_margin"]])
    threshold = np.percentile(margins_all, 30)

    mask = margins_all < threshold
    if len(set(y_true[mask])) > 1:
        fpr_c, tpr_c, auc_c, tpr1_c = print_roc(
            "Energy (sim_margin < p30)",
            y_true[mask],
            energy_scores[mask]
        )
    else:
        auc_c = None
        tpr1_c = None

    # ---------------------------------------------------------
    # Policy Performance
    # ---------------------------------------------------------

    tpr_policy = np.mean(pos["verdict"] == "accept")
    far_policy = np.mean(neg["verdict"] == "accept")

    print("\n================ POLICY PERFORMANCE ================")
    print("TPR:", float(tpr_policy))
    print("FAR:", float(far_policy))

    # ---------------------------------------------------------
    # Plots
    # ---------------------------------------------------------

    plots_dir = run_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)

    # ROC Plot
    plt.figure()
    plt.plot(fpr_e, tpr_e, label=f"Energy ({auc_e:.3f})")
    plt.plot(fpr_pr, tpr_pr, label=f"PR ({auc_pr:.3f})")
    plt.plot(fpr_i, tpr_i, label=f"E×(1-M) ({auc_i:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig(plots_dir / "roc_curves.png")
    plt.close()

    # ---------------------------------------------------------
    # Summary JSON
    # ---------------------------------------------------------

    summary = {
        "roc": {
            "energy_auc": auc_e,
            "pr_auc": auc_pr,
            "sensitivity_auc": auc_s,
            "sim_margin_auc": auc_m,
            "alignment_auc": auc_a,
            "effective_rank_auc": auc_er,
            "entropy_rank_auc": auc_ent,
            "interaction_auc": auc_i,
            "conditional_energy_auc": auc_c,
        },
        "policy": {
            "TPR": float(tpr_policy),
            "FAR": float(far_policy),
        }
    }

    with open(run_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPlots saved to: {plots_dir}")
    print("Summary written to analysis_summary.json\n")


if __name__ == "__main__":
    main()
