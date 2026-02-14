#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main():
    run_dirs = sorted(Path("artifacts/runs").glob("*"))
    if not run_dirs:
        raise RuntimeError("No runs found.")

    run_dir = run_dirs[-1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=run_dir, help="Path to artifacts/<run_id> directory")
    args = parser.parse_args()

    run_dir = args.run

    report_path = run_dir / "report_summary.json"
    meta_path = run_dir / "run_meta.json"

    if not report_path.exists():
        raise RuntimeError("report_summary.json not found")

    if not meta_path.exists():
        raise RuntimeError("run_meta.json not found")

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    proof = {
        "run_id": meta["run_id"],
        "dataset": meta["dataset"],
        "model": meta["embedding"]["model"],
        "policy": meta["policy"],
        "modes": {}
    }

    for mode, stats in report["modes"].items():
        proof["modes"][mode] = {
            "auc": stats["auc"],
            "tpr_at_tau": stats["tpr_at_tau"],
            "far_at_tau": stats["far_at_tau"],
            "axis_collapse_detected": len(stats.get("axis_collapse_pos", [])) > 0,
            "stability_fraction_pos": stats.get("stability_fraction_pos"),
            "hard_negative_gap_pos_mean":
                stats.get("hard_negative_gap_pos", {}).get("mean")
        }

    proof_json_path = run_dir / "proof.json"
    proof_md_path = run_dir / "proof.md"

    with proof_json_path.open("w", encoding="utf-8") as f:
        json.dump(proof, f, indent=2)

    # Markdown
    lines = []
    lines.append("# Certum Proof Summary\n")

    for mode, m in proof["modes"].items():
        lines.append(f"## Mode: {mode}")
        lines.append(f"- AUC: {m['auc']:.4f}")
        lines.append(f"- TPR@tau: {m['tpr_at_tau']:.4f}")
        lines.append(f"- FAR@tau: {m['far_at_tau']:.4f}")
        lines.append(f"- Axis Collapse Detected: {m['axis_collapse_detected']}")
        lines.append(f"- Stability Fraction (POS): {m['stability_fraction_pos']}")
        lines.append(f"- Hard Negative Gap Mean (POS): {m['hard_negative_gap_pos_mean']}")
        lines.append("")

    with proof_md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nProof written to: {proof_json_path}")
    print(f"Markdown proof: {proof_md_path}")


if __name__ == "__main__":
    main()
