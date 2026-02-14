def build_markdown(summary: dict) -> str:
    lines = []
    lines.append("# Certum Run Report\n")

    for mode, stats in summary["modes"].items():
        lines.append(f"## Mode: {mode}")
        lines.append("")
        lines.append(f"- AUC: {stats['auc']:.4f}")
        lines.append(f"- TPR@tau: {stats['tpr_at_tau']:.4f}")
        lines.append(f"- FAR@tau: {stats['far_at_tau']:.4f}")
        lines.append("")

    return "\n".join(lines)
