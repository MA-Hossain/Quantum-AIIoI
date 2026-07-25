"""Generate publication-quality figures from experiment results.

Produces Figures 2-12 covering worst-case AoII, average AoII, sum rate,
computation time, delivery ratio, fresh-update ratio, latency,
ADMM convergence, solver comparison bars, per-UE AoII distribution,
and a summary heatmap.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------

SOLVER_STYLES = {
    "sa":      {"color": "#e74c3c", "marker": "s", "label": "SA (neal)"},
    "admm":    {"color": "#3498db", "marker": "D", "label": "ADMM"},
    "random":  {"color": "#95a5a6", "marker": "x", "label": "Random"},
    "greedy":  {"color": "#f39c12", "marker": "^", "label": "Greedy"},
    "exact":   {"color": "#2ecc71", "marker": "o", "label": "Exact"},
    "qaoa":    {"color": "#9b59b6", "marker": "P", "label": "QAOA"},
}


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _aggregate(results: List[Dict], metric: str, solver: str):
    """Group metric values by num_ues."""
    by_ues: Dict[int, List[float]] = {}
    for r in results:
        n = r["num_ues"]
        if solver not in r["solvers"]:
            continue
        val = r["solvers"][solver].get(metric)
        if val is not None:
            by_ues.setdefault(n, []).append(val)
    return by_ues


def _mean_std(by_ues):
    ues = sorted(by_ues.keys())
    means = [np.mean(by_ues[n]) for n in ues]
    stds = [np.std(by_ues[n]) for n in ues]
    return ues, means, stds


def _line_plot(ax, results, metric, ylabel, title, solvers=None):
    if solvers is None:
        solvers = ["sa", "admm", "greedy", "random"]
    for s in solvers:
        by_ues = _aggregate(results, metric, s)
        if not by_ues:
            continue
        ues, means, stds = _mean_std(by_ues)
        style = SOLVER_STYLES.get(s, {})
        ax.errorbar(
            ues, means, yerr=stds,
            fmt=f"-{style.get('marker', 'o')}",
            color=style.get("color"),
            label=style.get("label", s),
            capsize=3, linewidth=1.5,
        )
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# -----------------------------------------------------------------------
# Individual figures
# -----------------------------------------------------------------------

def plot_worst_aoii(results, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    _line_plot(ax, results, "worst_aoii", "Worst-case AoII",
               "Fig 2: Worst-case AoII vs Number of UEs")
    fig.tight_layout()
    path = os.path.join(output_dir, "fig02_worst_aoii.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_avg_aoii(results, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    _line_plot(ax, results, "avg_aoii", "Average AoII",
               "Fig 3: Average AoII vs Number of UEs")
    fig.tight_layout()
    path = os.path.join(output_dir, "fig03_avg_aoii.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sum_rate(results, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    _line_plot(ax, results, "sum_rate", "Sum Rate (bps)",
               "Fig 4: Sum Rate vs Number of UEs")
    fig.tight_layout()
    path = os.path.join(output_dir, "fig04_sum_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_timing(results, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    _line_plot(ax, results, "timing_s", "Computation Time (s)",
               "Fig 5: Computation Time vs Number of UEs")
    ax.set_yscale("log")
    fig.tight_layout()
    path = os.path.join(output_dir, "fig05_timing.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_delivery_ratio(results, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    _line_plot(ax, results, "delivery_ratio", "Delivery Ratio",
               "Fig 6: Delivery Ratio vs Number of UEs")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig06_delivery_ratio.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_fresh_ratio(results, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    _line_plot(ax, results, "fresh_ratio", "Fresh-Update Ratio (Y<=2)",
               "Fig 7: Fresh-Update Ratio vs Number of UEs")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig07_fresh_ratio.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_latency(results, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    _line_plot(ax, results, "avg_latency", "Average Latency (slots)",
               "Fig 8: Average Latency vs Number of UEs")
    fig.tight_layout()
    path = os.path.join(output_dir, "fig08_latency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_admm_convergence(results, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    seen = set()
    for r in results:
        n = r["num_ues"]
        if n in seen:
            continue
        seen.add(n)
        admm = r["solvers"].get("admm", {})
        residuals = admm.get("primal_residuals", [])
        if residuals:
            ax.plot(range(1, len(residuals) + 1), residuals,
                    label=f"{n} UEs", linewidth=1.5)
    ax.set_xlabel("ADMM Iteration")
    ax.set_ylabel("Primal Residual")
    ax.set_title("Fig 9: ADMM Convergence")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig09_admm_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_solver_bars(results, output_dir):
    max_ues = max(r["num_ues"] for r in results)
    subset = [r for r in results if r["num_ues"] == max_ues]
    solvers = ["random", "greedy", "admm", "sa"]

    fig, ax = plt.subplots(figsize=(7, 5))
    names, means, stds, colors = [], [], [], []
    for s in solvers:
        vals = [r["solvers"][s]["worst_aoii"]
                for r in subset if s in r["solvers"]]
        if vals:
            style = SOLVER_STYLES.get(s, {})
            names.append(style.get("label", s))
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            colors.append(style.get("color", "#333"))

    ax.bar(names, means, yerr=stds, color=colors, edgecolor="black",
           linewidth=0.5, capsize=4)
    ax.set_ylabel("Worst-case AoII")
    ax.set_title(f"Fig 10: Solver Comparison ({max_ues} UEs)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig10_solver_bars.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_per_ue_aoii(results, output_dir):
    max_ues = max(r["num_ues"] for r in results)
    r = next(r for r in results if r["num_ues"] == max_ues)

    solvers = ["random", "greedy", "admm", "sa"]
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels, colors_list = [], [], []
    for s in solvers:
        if s in r["solvers"] and "per_ue_aoii" in r["solvers"][s]:
            data.append(r["solvers"][s]["per_ue_aoii"])
            style = SOLVER_STYLES.get(s, {})
            labels.append(style.get("label", s))
            colors_list.append(style.get("color", "#333"))

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Per-UE AoII")
    ax.set_title(f"Fig 11: Per-UE AoII Distribution ({max_ues} UEs)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig11_per_ue_aoii.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_summary_heatmap(results, output_dir):
    max_ues = max(r["num_ues"] for r in results)
    subset = [r for r in results if r["num_ues"] == max_ues]
    solvers = ["random", "greedy", "admm", "sa"]
    metrics = ["worst_aoii", "avg_aoii", "delivery_ratio", "fresh_ratio"]
    metric_labels = ["Worst AoII", "Avg AoII", "Delivery Ratio", "Fresh Ratio"]

    data = np.zeros((len(solvers), len(metrics)))
    for si, s in enumerate(solvers):
        for mi, met in enumerate(metrics):
            vals = [r["solvers"][s][met]
                    for r in subset if s in r["solvers"]]
            data[si, mi] = np.mean(vals) if vals else 0

    solver_labels = [SOLVER_STYLES.get(s, {}).get("label", s) for s in solvers]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="YlOrRd_r", aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_yticks(range(len(solvers)))
    ax.set_yticklabels(solver_labels, fontsize=9)

    for si in range(len(solvers)):
        for mi in range(len(metrics)):
            ax.text(mi, si, f"{data[si, mi]:.2f}",
                    ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax)
    ax.set_title(f"Fig 12: Performance Summary ({max_ues} UEs)")
    fig.tight_layout()
    path = os.path.join(output_dir, "fig12_summary_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -----------------------------------------------------------------------
# Generate all figures
# -----------------------------------------------------------------------

def generate_all_figures(results: List[Dict], output_dir: str) -> List[str]:
    """Generate Figures 2-12 and return list of file paths."""
    os.makedirs(output_dir, exist_ok=True)
    return [
        plot_worst_aoii(results, output_dir),
        plot_avg_aoii(results, output_dir),
        plot_sum_rate(results, output_dir),
        plot_timing(results, output_dir),
        plot_delivery_ratio(results, output_dir),
        plot_fresh_ratio(results, output_dir),
        plot_latency(results, output_dir),
        plot_admm_convergence(results, output_dir),
        plot_solver_bars(results, output_dir),
        plot_per_ue_aoii(results, output_dir),
        plot_summary_heatmap(results, output_dir),
    ]
