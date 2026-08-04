#!/usr/bin/env python3
"""SENTINEL Paper — Experiment Runner

Usage:
    python3 run_all_experiments.py --tier quick     # ~5 min, smoke test
    python3 run_all_experiments.py --tier medium    # ~1-2 hours
    python3 run_all_experiments.py --tier full      # ~4-8 hours (550 UEs)
    python3 run_all_experiments.py --experiment 3   # run only experiment 3
    python3 run_all_experiments.py --experiment 1 2 # run experiments 1 and 2
    python3 run_all_experiments.py --plots-only     # regenerate figures from saved results

Tiers:
    quick  — Experiments 1 only (small validation, 3-8 UEs, 2 seeds)
    medium — Experiments 1-2, 4-6 (up to 100 UEs, 3 seeds)
    full   — All experiments 1-6 (up to 550 UEs, 5 seeds)

Results are saved to output/large_scale/ as JSON files.
Figures are saved to output/large_scale/figures/.
"""

import argparse
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.run_large_scale import (
    experiment_1_small_validation,
    experiment_2_solver_comparison,
    experiment_3_large_scale,
    experiment_4_disruption_severity,
    experiment_5_multi_scenario,
    experiment_6_qaoa_in_admm,
    load_results,
)
from experiments.plots_extended import generate_all_extended_figures


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "output", "large_scale")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")


def run_experiments(experiments: list, seeds_override: list = None):
    """Run selected experiments and generate figures."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    exp_results = {}
    total_t0 = time.perf_counter()

    runners = {
        1: ("exp1", experiment_1_small_validation),
        2: ("exp2", experiment_2_solver_comparison),
        3: ("exp3", experiment_3_large_scale),
        4: ("exp4", experiment_4_disruption_severity),
        5: ("exp5", experiment_5_multi_scenario),
        6: ("exp6", experiment_6_qaoa_in_admm),
    }

    for exp_num in experiments:
        if exp_num not in runners:
            print(f"Unknown experiment: {exp_num}")
            continue

        key, runner = runners[exp_num]
        t0 = time.perf_counter()
        try:
            kwargs = {"output_dir": OUTPUT_DIR}
            if seeds_override is not None:
                kwargs["seeds"] = seeds_override
            results = runner(**kwargs)
            exp_results[key] = results
            elapsed = time.perf_counter() - t0
            print(f"  Experiment {exp_num} completed in {elapsed:.1f}s "
                  f"({len(results)} instances)\n")
        except Exception as e:
            print(f"  Experiment {exp_num} FAILED: {e}\n")
            import traceback
            traceback.print_exc()

    # Generate figures
    if exp_results:
        print("=" * 60)
        print("Generating figures...")
        print("=" * 60)
        paths = generate_all_extended_figures(exp_results, FIG_DIR)
        print(f"  Total: {len(paths)} figures saved to {FIG_DIR}/")

    total_elapsed = time.perf_counter() - total_t0
    print(f"\nTotal runtime: {total_elapsed:.1f}s "
          f"({total_elapsed/60:.1f} min)")


def plots_only():
    """Regenerate figures from saved JSON results."""
    import json
    exp_results = {}
    result_files = {
        "exp1": "exp1_small_validation.json",
        "exp2": "exp2_solver_comparison.json",
        "exp3": "exp3_large_scale.json",
        "exp4": "exp4_disruption_severity.json",
        "exp5": "exp5_multi_scenario.json",
        "exp6": "exp6_qaoa_in_admm.json",
    }
    for key, fname in result_files.items():
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            exp_results[key] = load_results(path)
            print(f"  Loaded {key}: {len(exp_results[key])} instances")

    if not exp_results:
        print("No saved results found. Run experiments first.")
        return

    paths = generate_all_extended_figures(exp_results, FIG_DIR)
    print(f"  Total: {len(paths)} figures generated")


def main():
    parser = argparse.ArgumentParser(
        description="SENTINEL Paper Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tier", choices=["quick", "medium", "full"],
        help="Run a predefined tier of experiments",
    )
    parser.add_argument(
        "--experiment", nargs="+", type=int,
        help="Run specific experiments (1-6)",
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Regenerate figures from saved results",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        help="Override number of seeds (e.g., --seeds 0 1 2)",
    )

    args = parser.parse_args()

    if args.plots_only:
        plots_only()
        return

    if args.experiment:
        experiments = args.experiment
    elif args.tier == "quick":
        experiments = [1]
        if args.seeds is None:
            args.seeds = [0, 1]
    elif args.tier == "medium":
        experiments = [1, 2, 4, 5, 6]
        if args.seeds is None:
            args.seeds = [0, 1, 2]
    elif args.tier == "full":
        experiments = [1, 2, 3, 4, 5, 6]
    else:
        parser.print_help()
        print("\nExample: python3 run_all_experiments.py --tier quick")
        return

    print("=" * 60)
    print("SENTINEL Paper — Experiment Runner")
    print(f"Experiments: {experiments}")
    print(f"Seeds: {args.seeds or 'default (5)'}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    print()

    run_experiments(experiments, seeds_override=args.seeds)


if __name__ == "__main__":
    main()
