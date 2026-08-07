"""Extended figure generation for large-scale experiments.

Generates publication-quality figures for Experiments 1-6.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

SOLVER_STYLES = {
    "sa":              {"color": "#e74c3c", "marker": "s", "label": "SA (QUBO)"},
    "admm":            {"color": "#3498db", "marker": "D", "label": "ADMM (SENTINEL)"},
    "random":          {"color": "#95a5a6", "marker": "x", "label": "Random"},
    "greedy_rate":     {"color": "#f39c12", "marker": "^", "label": "Greedy-Rate"},
    "greedy_aoii":     {"color": "#e67e22", "marker": "v", "label": "Greedy-AoII"},
    "greedy_aoii_ls":  {"color": "#d35400", "marker": "p", "label": "Greedy-AoII+LS"},
    "exact":           {"color": "#2ecc71", "marker": "o", "label": "Exact"},
    "qaoa":            {"color": "#9b59b6", "marker": "P", "label": "QAOA"},
    "admm_raw":        {"color": "#85c1e9", "marker": "d", "label": "ADMM (raw)"},
    "admm_sa_sub":     {"color": "#2980b9", "marker": "D", "label": "ADMM (SA sub)"},
    "admm_qaoa_sub":   {"color": "#8e44ad", "marker": "H", "label": "ADMM (QAOA sub)"},
}


def _aggregate(results, metric, solver):
    by_key: Dict[str, List[float]] = {}
    for r in results:
        key = str(r.get("num_ues", "?"))
        if solver not in r.get("solvers", {}):
            continue
        val = r["solvers"][solver].get(metric)
        if val is not None:
            by_key.setdefault(key, []).append(float(val))
    return by_key


def _agg_by_field(results, metric, solver, field):
    by_key: Dict[str, List[float]] = {}
    for r in results:
        key = str(r.get(field, "?"))
        if solver not in r.get("solvers", {}):
            continue
        val = r["solvers"][solver].get(metric)
        if val is not None:
            by_key.setdefault(key, []).append(float(val))
    return by_key


def _plot_lines(ax, results, metric, ylabel, title, solvers, x_field="num_ues",
                x_label="Number of UEs"):
    for s in solvers:
        if x_field == "num_ues":
            by_key = _aggregate(results, metric, s)
        else:
            by_key = _agg_by_field(results, metric, s, x_field)
        if not by_key:
            continue
        keys = sorted(by_key.keys(), key=lambda k: (len(k), k))
        try:
            x_vals = [int(k) for k in keys]
        except ValueError:
            x_vals = list(range(len(keys)))
        means = [np.mean(by_key[k]) for k in keys]
        stds = [np.std(by_key[k]) for k in keys]
        style = SOLVER_STYLES.get(s, {})
        ax.errorbar(
            x_vals, means, yerr=stds,
            fmt=f"-{style.get('marker', 'o')}",
            color=style.get("color"),
            label=style.get("label", s),
            capsize=3, linewidth=1.5,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Experiment-specific figure generators
# ---------------------------------------------------------------------------

GAP_SOLVERS = ["greedy_aoii", "greedy_aoii_ls", "sa", "admm_raw", "admm"]

# Categorical sweep values that must not be sorted alphabetically.
CATEGORY_ORDER = {"mild": 0, "moderate": 1, "severe": 2, "extreme": 3}


def _plot_optimality_gap(results, output_dir, x_field, x_label, title, fname):
    """Plot each solver's percentage gap to the exact optimum.

    ``x_field`` names the instance field to sweep along; it may be numeric
    (num_ues, num_scenarios) or categorical (severity).  Returns the saved path,
    or None when no instance carries an exact optimum.
    """
    by_solver: Dict[str, Dict] = {s: {} for s in GAP_SOLVERS}
    have_exact = False

    for r in results:
        exact = r["solvers"].get("exact", {}).get("worst_aoii")
        if exact is None or exact <= 0:
            continue
        have_exact = True
        key = r.get(x_field)
        for s in GAP_SOLVERS:
            val = r["solvers"].get(s, {}).get("worst_aoii")
            if val is not None:
                by_solver[s].setdefault(key, []).append((val - exact) / exact * 100.0)

    if not have_exact:
        print(f"    [no figure] {fname}: no exact optimum in results")
        return None

    all_keys = {k for d in by_solver.values() for k in d}
    numeric = all(isinstance(k, (int, float)) for k in all_keys)
    if numeric:
        keys = sorted(all_keys)
        xs = keys
    else:
        # Severity is ordinal, not alphabetical — sorting by name would put
        # "extreme" first and invert the trend the figure is meant to show.
        keys = sorted(all_keys, key=lambda k: (CATEGORY_ORDER.get(k, 99), str(k)))
        xs = list(range(len(keys)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for s in GAP_SOLVERS:
        data = by_solver[s]
        if not data:
            continue
        pts = [(x, np.mean(data[k])) for x, k in zip(xs, keys) if k in data]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=s)

    ax.axhline(0.0, color="green", linestyle="--", label="exact optimum")
    if not numeric:
        ax.set_xticks(xs)
        ax.set_xticklabels([str(k) for k in keys])
    ax.set_ylabel("Optimality gap vs exact (%)")
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    p = os.path.join(output_dir, fname)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_exp1(results, output_dir):
    """Exp 1: Small-instance validation figures."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    solvers = ["exact", "sa", "admm", "admm_raw", "greedy_aoii_ls", "greedy_aoii"]

    # Fig: Worst-case AoII comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_lines(ax, results, "worst_aoii", "Worst-case AoII",
                "Exp 1: Small-Instance Solver Validation", solvers)
    fig.tight_layout()
    p = os.path.join(output_dir, "exp1_worst_aoii.png")
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # Fig: optimality gap vs the exact ground truth
    p = _plot_optimality_gap(
        results, output_dir, "num_ues", "Number of UEs",
        "Exp 1: Solver Gap to Exact Minimax Optimum",
        "exp1_optimality_gap.png",
    )
    if p:
        paths.append(p)

    # Fig: QAOA approximation ratio — only when QAOA actually produced data.
    # Previously this was saved unconditionally and emitted a blank chart
    # whenever QAOA was skipped, which hid the fact that it never ran.
    by_ue: Dict[int, List[float]] = {}
    for r in results:
        n = r["num_ues"]
        exact_e = r["solvers"].get("exact_qubo", {}).get("energy")
        qaoa_e = r["solvers"].get("qaoa", {}).get("energy")
        if exact_e is not None and qaoa_e is not None and exact_e != 0:
            by_ue.setdefault(n, []).append(qaoa_e / exact_e)

    if by_ue:
        fig, ax = plt.subplots(figsize=(8, 5))
        ues = sorted(by_ue.keys())
        means = [np.mean(by_ue[n]) for n in ues]
        stds = [np.std(by_ue[n]) for n in ues]
        ax.bar([str(n) for n in ues], means, yerr=stds,
               color="#9b59b6", edgecolor="black", capsize=4)
        ax.axhline(1.0, color="green", linestyle="--", label="Exact optimum")
        ax.set_ylabel("QAOA Energy / Exact Energy")
        ax.set_xlabel("Number of UEs")
        ax.set_title("Exp 1: QAOA Approximation Ratio")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(output_dir, "exp1_qaoa_ratio.png")
        fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    else:
        skipped = [r.get("skipped", {}).get("qaoa") for r in results]
        skipped = [s for s in skipped if s]
        reason = skipped[0] if skipped else "QAOA not run for these instances"
        print(f"    [no figure] exp1_qaoa_ratio: {reason}")

    return paths


def plot_exp2(results, output_dir):
    """Exp 2: Medium-scale solver comparison figures."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    solvers = ["exact", "sa", "admm", "admm_raw", "greedy_aoii_ls", "greedy_rate", "greedy_aoii", "random"]

    for metric, ylabel, title, fname in [
        ("worst_aoii", "Worst-case AoII", "Worst-case AoII vs UEs", "exp2_worst_aoii.png"),
        ("avg_aoii", "Average AoII", "Average AoII vs UEs", "exp2_avg_aoii.png"),
        ("timing_s", "Time (s)", "Computation Time vs UEs", "exp2_timing.png"),
        ("delivery_ratio", "Delivery Ratio", "Delivery Ratio vs UEs", "exp2_delivery.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_lines(ax, results, metric, ylabel, f"Exp 2: {title}", solvers)
        if metric == "timing_s":
            ax.set_yscale("log")
        fig.tight_layout()
        p = os.path.join(output_dir, fname)
        fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    p = _plot_optimality_gap(
        results, output_dir, "num_ues", "Number of UEs",
        "Exp 2: Solver Gap to Exact Minimax Optimum",
        "exp2_optimality_gap.png",
    )
    if p:
        paths.append(p)

    return paths


def plot_exp3(results, output_dir):
    """Exp 3: Large-scale scalability figures."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    solvers = ["admm", "admm_raw", "greedy_aoii_ls", "greedy_rate", "greedy_aoii", "random"]

    for metric, ylabel, title, fname in [
        ("worst_aoii", "Worst-case AoII", "Worst-case AoII (Large Scale)", "exp3_worst_aoii.png"),
        ("avg_aoii", "Average AoII", "Average AoII (Large Scale)", "exp3_avg_aoii.png"),
        ("timing_s", "Time (s)", "Computation Time (Large Scale)", "exp3_timing.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_lines(ax, results, metric, ylabel, f"Exp 3: {title}", solvers)
        if metric == "timing_s":
            ax.set_yscale("log")
        fig.tight_layout()
        p = os.path.join(output_dir, fname)
        fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # Domain size breakdown
    fig, ax = plt.subplots(figsize=(8, 5))
    by_ue: Dict[int, List] = {}
    for r in results:
        ds = r.get("solvers", {}).get("admm", {}).get("domain_sizes")
        if ds:
            by_ue.setdefault(r["num_ues"], []).append(ds)
    if by_ue:
        ues = sorted(by_ue.keys())
        total_vars = []
        max_domain = []
        for n in ues:
            all_ds = by_ue[n]
            total_vars.append(np.mean([sum(d) for d in all_ds]))
            max_domain.append(np.mean([max(d) for d in all_ds]))
        ax.bar([str(n) for n in ues], total_vars, color="#3498db",
               edgecolor="black", label="Total QUBO vars", alpha=0.7)
        ax.bar([str(n) for n in ues], max_domain, color="#e74c3c",
               edgecolor="black", label="Max domain size", alpha=0.7)
        ax.set_xlabel("Number of UEs")
        ax.set_ylabel("Variables")
        ax.set_title("Exp 3: ADMM Domain Decomposition")
        ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(output_dir, "exp3_domain_sizes.png")
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    return paths


def plot_exp4(results, output_dir):
    """Exp 4: Disruption severity sweep figures."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    solvers = ["exact", "sa", "admm", "greedy_aoii_ls", "greedy_rate", "random"]

    sev_order = {"mild": 0, "moderate": 1, "severe": 2, "extreme": 3}

    for metric, ylabel, fname in [
        ("worst_aoii", "Worst-case AoII", "exp4_worst_aoii.png"),
        ("avg_aoii", "Average AoII", "exp4_avg_aoii.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for s in solvers:
            by_sev: Dict[str, List[float]] = {}
            for r in results:
                sev = r.get("severity", "?")
                val = r.get("solvers", {}).get(s, {}).get(metric)
                if val is not None:
                    by_sev.setdefault(sev, []).append(float(val))
            if not by_sev:
                continue
            keys = sorted(by_sev.keys(), key=lambda k: sev_order.get(k, 99))
            x_vals = [sev_order.get(k, 99) for k in keys]
            means = [np.mean(by_sev[k]) for k in keys]
            stds = [np.std(by_sev[k]) for k in keys]
            style = SOLVER_STYLES.get(s, {})
            ax.errorbar(x_vals, means, yerr=stds,
                        fmt=f"-{style.get('marker', 'o')}",
                        color=style.get("color"),
                        label=style.get("label", s),
                        capsize=3, linewidth=1.5)
        ax.set_xticks(list(sev_order.values()))
        ax.set_xticklabels(list(sev_order.keys()))
        ax.set_xlabel("Disruption Severity")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Exp 4: {ylabel} vs Disruption Severity")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = os.path.join(output_dir, fname)
        fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # Improvement of ADMM over greedy_aoii_ls by severity
    fig, ax = plt.subplots(figsize=(8, 5))
    by_sev_imp: Dict[str, List[float]] = {}
    for r in results:
        sev = r.get("severity", "?")
        admm_w = r.get("solvers", {}).get("admm", {}).get("worst_aoii")
        gls_w = r.get("solvers", {}).get("greedy_aoii_ls", {}).get("worst_aoii")
        if admm_w is not None and gls_w is not None and gls_w > 0:
            by_sev_imp.setdefault(sev, []).append((gls_w - admm_w) / gls_w * 100)
    if by_sev_imp:
        keys = sorted(by_sev_imp.keys(), key=lambda k: sev_order.get(k, 99))
        means = [np.mean(by_sev_imp[k]) for k in keys]
        stds = [np.std(by_sev_imp[k]) for k in keys]
        colors = ["#27ae60", "#f39c12", "#e74c3c", "#8e44ad"]
        ax.bar(keys, means, yerr=stds, color=colors[:len(keys)],
               edgecolor="black", capsize=4)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Disruption Severity")
    ax.set_ylabel("ADMM Improvement over Greedy+LS (%)")
    ax.set_title("Exp 4: SENTINEL Advantage by Disruption Severity")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(output_dir, "exp4_improvement.png")
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    p = _plot_optimality_gap(
        results, output_dir, "severity", "Disruption Severity",
        "Exp 4: Solver Gap to Exact Optimum by Severity",
        "exp4_optimality_gap.png",
    )
    if p:
        paths.append(p)

    return paths


def plot_exp5(results, output_dir):
    """Exp 5: Multi-scenario robustness figures."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    solvers = ["exact", "sa", "admm", "greedy_aoii_ls", "greedy_rate"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for s in solvers:
        by_sc: Dict[int, List[float]] = {}
        for r in results:
            n_sc = r.get("num_scenarios", 1)
            val = r.get("solvers", {}).get(s, {}).get("worst_aoii")
            if val is not None:
                by_sc.setdefault(n_sc, []).append(float(val))
        if not by_sc:
            continue
        keys = sorted(by_sc.keys())
        means = [np.mean(by_sc[k]) for k in keys]
        stds = [np.std(by_sc[k]) for k in keys]
        style = SOLVER_STYLES.get(s, {})
        ax.errorbar(keys, means, yerr=stds,
                    fmt=f"-{style.get('marker', 'o')}",
                    color=style.get("color"),
                    label=style.get("label", s),
                    capsize=3, linewidth=1.5)
    ax.set_xlabel("Number of Disruption Scenarios (S)")
    ax.set_ylabel("Worst-case AoII")
    ax.set_title("Exp 5: Worst-case AoII vs Number of Scenarios")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(output_dir, "exp5_scenarios.png")
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    p = _plot_optimality_gap(
        results, output_dir, "num_scenarios", "Number of Disruption Scenarios (S)",
        "Exp 5: Solver Gap to Exact Optimum vs Scenario Count",
        "exp5_optimality_gap.png",
    )
    if p:
        paths.append(p)

    return paths


def plot_exp6(results, output_dir):
    """Exp 6: QAOA-in-ADMM comparison figures."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    solvers = ["admm_sa_sub", "admm_qaoa_sub", "greedy_aoii_ls"]

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_lines(ax, results, "worst_aoii", "Worst-case AoII",
                "Exp 6: QAOA-in-ADMM vs SA-in-ADMM", solvers)
    fig.tight_layout()
    p = os.path.join(output_dir, "exp6_qaoa_admm.png")
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # Timing comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_lines(ax, results, "timing_s", "Time (s)",
                "Exp 6: Runtime Comparison", solvers)
    ax.set_yscale("log")
    fig.tight_layout()
    p = os.path.join(output_dir, "exp6_timing.png")
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    return paths


def generate_all_extended_figures(exp_results: Dict[str, List], output_dir: str):
    """Generate figures for all experiments that have results."""
    os.makedirs(output_dir, exist_ok=True)
    all_paths = []

    plotters = {
        "exp1": plot_exp1,
        "exp2": plot_exp2,
        "exp3": plot_exp3,
        "exp4": plot_exp4,
        "exp5": plot_exp5,
        "exp6": plot_exp6,
    }

    for key, results in exp_results.items():
        if results and key in plotters:
            try:
                paths = plotters[key](results, output_dir)
                all_paths.extend(paths)
                print(f"  {key}: {len(paths)} figures generated")
            except Exception as e:
                print(f"  {key}: FAILED - {e}")

    return all_paths
