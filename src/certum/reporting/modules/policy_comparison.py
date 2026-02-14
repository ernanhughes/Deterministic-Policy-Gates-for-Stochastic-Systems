import numpy as np

from certum.axes.bundle import AxisBundle
from certum.policy.energy_only import EnergyOnlyPolicy


def evaluate_policy(pos_rows, neg_rows, policy):
    """
    Compute TPR and FAR for arbitrary policy.
    """
    pos_accept = []
    neg_accept = []

    for r in pos_rows:
        axes = AxisBundle.from_trace(r)
        verdict = policy.decide(axes, r.get("effectiveness", 0.0))
        pos_accept.append(verdict.name.lower() == "accept")

    for r in neg_rows:
        axes = AxisBundle.from_trace(r)
        verdict = policy.decide(axes, r.get("effectiveness", 0.0))
        neg_accept.append(verdict.name.lower() == "accept")

    return {
        "tpr": float(np.mean(pos_accept)),
        "far": float(np.mean(neg_accept)),
    }

def equal_far_comparison(pos_rows, neg_rows, adaptive_policy):
    """
    Compare adaptive vs energy-only at equal FAR.
    """

    # 1️⃣ Measure adaptive performance
    adaptive_metrics = evaluate_policy(pos_rows, neg_rows, adaptive_policy)
    target_far = adaptive_metrics["far"]

    # 2️⃣ Compute tau_energy that matches that FAR
    neg_energies = np.array([r["energy"]["value"] for r in neg_rows])

    if target_far <= 0.0:
        tau_energy_equal = np.min(neg_energies) - 1e-6
    else:
        tau_energy_equal = float(np.quantile(neg_energies, target_far))

    energy_policy = EnergyOnlyPolicy(tau_energy=tau_energy_equal)
    energy_metrics = evaluate_policy(pos_rows, neg_rows, energy_policy)

    return {
        "target_far": target_far,
        "adaptive_tpr": adaptive_metrics["tpr"],
        "energy_equal_far_tpr": energy_metrics["tpr"],
        "tpr_gain": adaptive_metrics["tpr"] - energy_metrics["tpr"],
        "tau_energy_equal": tau_energy_equal,
    }


def compare_energy_vs_policy(pos_rows, neg_rows, tau_energy):
    """
    Compare energy-only decision vs adaptive policy decision.
    """

    # --- Energy-only decisions ---
    pos_energy = np.array([r["energy"]["value"] for r in pos_rows])
    neg_energy = np.array([r["energy"]["value"] for r in neg_rows])

    energy_accept_pos = pos_energy <= tau_energy
    energy_accept_neg = neg_energy <= tau_energy

    # --- Adaptive decisions ---
    policy_accept_pos = np.array(
        [r["decision"]["verdict"] == "accept" for r in pos_rows]
    )
    policy_accept_neg = np.array(
        [r["decision"]["verdict"] == "accept" for r in neg_rows]
    )

    results = {
        "energy_only": {
            "tpr": float(np.mean(energy_accept_pos)),
            "far": float(np.mean(energy_accept_neg)),
        },
        "adaptive_policy": {
            "tpr": float(np.mean(policy_accept_pos)),
            "far": float(np.mean(policy_accept_neg)),
        },
    }

    # Add delta
    results["delta"] = {
        "tpr_gain": results["adaptive_policy"]["tpr"] - results["energy_only"]["tpr"],
        "far_change": results["adaptive_policy"]["far"] - results["energy_only"]["far"],
    }

    return results


def evaluate_with_policy(rows, policy):
    results = []

    for r in rows:
        axes = AxisBundle.from_trace(r)
        eff = r.get("effectiveness", 0.0)
        verdict = policy.decide(axes, eff)
        results.append(verdict.name.lower() == "accept")

    return results

def sweep_policy_curve(pos_rows, neg_rows, policy_builder, taus):
    """
    Sweep tau values and compute (TPR, FAR) curve for given policy.

    policy_builder: function that takes tau -> policy instance
    taus: iterable of tau values
    """

    curve = []

    for tau in taus:
        policy = policy_builder(tau)

        pos_accept = evaluate_with_policy(pos_rows, policy)
        neg_accept = evaluate_with_policy(neg_rows, policy)

        tpr = float(np.mean(pos_accept))
        far = float(np.mean(neg_accept))

        curve.append({
            "tau": float(tau),
            "tpr": tpr,
            "far": far
        })

    return curve

