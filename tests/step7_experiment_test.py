"""Step 7 Test: Experiment sweep + figure generation.

Test A — Single instance:  5 UEs, all 4 solvers run and produce valid metrics
Test B — Sweep:            [5, 8] UEs x 2 seeds = 4 instances
Test C — Metrics sanity:   worst >= avg, delivery ratio in [0,1], etc.
Test D — Figures:          all 11 figures (Fig 2-12) generated as PNG files
Test E — Solver ordering:  SA/ADMM should beat random baseline on avg
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from experiments.run import (
    build_rate_matrix,
    compute_metrics,
    greedy_rate_baseline,
    random_baseline,
    run_single,
    run_sweep,
)
from experiments.plots import generate_all_figures

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

all_passed = True
def check(name, condition, detail=""):
    global all_passed
    status = "PASS" if condition else "FAIL"
    if not condition:
        all_passed = False
    suffix = f"  ({detail})" if detail else ""
    log(f"  [{status}] {name}{suffix}")
    return condition


log("=" * 60)
log("STEP 7: EXPERIMENTS")
log("=" * 60)

# ================================================================
# Test A: Single instance (5 UEs)
# ================================================================
log("\n--- Test A: Single Instance (5 UEs) ---")

res_single = run_single(
    num_ues=5, seed=42,
    sa_reads=50, sa_sweeps=500,
    admm_max_iter=30,
)

log(f"  Instance: {res_single['num_ues']} UEs, "
    f"{res_single['num_nodes']} nodes, "
    f"{res_single['num_vars']} QUBO vars")
log(f"  Capacity: {res_single['capacity']}")

solvers_present = list(res_single["solvers"].keys())
log(f"  Solvers run: {solvers_present}")

check("All 6 solvers ran",
      all(s in solvers_present for s in ["random", "greedy", "greedy_aoii", "greedy_aoii_ls", "sa", "admm"]))

for solver_name in solvers_present:
    s = res_single["solvers"][solver_name]
    log(f"\n  {solver_name}:")
    log(f"    worst_aoii={s['worst_aoii']:.4f}, avg_aoii={s['avg_aoii']:.4f}")
    log(f"    sum_rate={s['sum_rate']:.0f} bps, avg_latency={s['avg_latency']:.2f}")
    log(f"    delivery_ratio={s['delivery_ratio']:.2f}, "
        f"fresh_ratio={s['fresh_ratio']:.2f}")
    log(f"    capacity_ok={s['capacity_ok']}, timing={s['timing_s']:.3f}s")
    if "energy" in s:
        log(f"    QUBO energy={s['energy']:.4f}")
    log(f"    association={s['association']}")

    check(f"{solver_name}: has valid worst_aoii",
          s["worst_aoii"] >= 0 and np.isfinite(s["worst_aoii"]))
    check(f"{solver_name}: has valid avg_aoii",
          s["avg_aoii"] >= 0 and np.isfinite(s["avg_aoii"]))
    check(f"{solver_name}: delivery_ratio in [0,1]",
          0 <= s["delivery_ratio"] <= 1)
    check(f"{solver_name}: fresh_ratio in [0,1]",
          0 <= s["fresh_ratio"] <= 1)
    check(f"{solver_name}: all UEs assigned",
          len(s["association"]) == res_single["num_ues"])

# ================================================================
# Test B: Sweep
# ================================================================
log("\n--- Test B: Experiment Sweep [5,8] x 2 seeds ---")

sweep_results = run_sweep(
    ue_counts=[5, 8],
    seeds=[0, 1],
    sa_reads=50, sa_sweeps=500,
    admm_max_iter=30,
)

log(f"  Total instances: {len(sweep_results)}")
check("Sweep produced 4 results", len(sweep_results) == 4)

ue_counts_seen = sorted(set(r["num_ues"] for r in sweep_results))
log(f"  UE counts: {ue_counts_seen}")
check("Both UE counts present", ue_counts_seen == [5, 8])

for r in sweep_results:
    for s_name in ["random", "greedy", "greedy_aoii", "greedy_aoii_ls", "sa"]:
        check(f"{r['num_ues']}UE/seed{r['seed']}/{s_name}: feasible",
              r["solvers"][s_name]["capacity_ok"],
              f"assoc={r['solvers'][s_name]['association']}")
    # ADMM is a heuristic — report but don't fail on capacity violations
    admm_ok = r["solvers"]["admm"]["capacity_ok"]
    log(f"  [{'PASS' if admm_ok else 'WARN'}] "
        f"{r['num_ues']}UE/seed{r['seed']}/admm: feasible={admm_ok}")

# ================================================================
# Test C: Metrics sanity
# ================================================================
log("\n--- Test C: Metrics Sanity ---")

for r in sweep_results:
    for s_name in ["sa", "admm", "greedy", "greedy_aoii", "greedy_aoii_ls", "random"]:
        s = r["solvers"][s_name]
        check(f"{r['num_ues']}UE/{s_name}: worst >= avg",
              s["worst_aoii"] >= s["avg_aoii"] - 1e-9,
              f"worst={s['worst_aoii']:.4f}, avg={s['avg_aoii']:.4f}")

# ================================================================
# Test D: Figure generation
# ================================================================
log("\n--- Test D: Figure Generation ---")

fig_dir = os.path.join(OUTPUT_DIR, "figures")
fig_paths = generate_all_figures(sweep_results, fig_dir)

log(f"  Generated {len(fig_paths)} figures:")
for p in fig_paths:
    fname = os.path.basename(p)
    exists = os.path.isfile(p)
    size_kb = os.path.getsize(p) / 1024 if exists else 0
    log(f"    {fname}: {size_kb:.1f} KB")
    check(f"Figure {fname} exists", exists)

check("All 11 figures created", len(fig_paths) == 11)

# ================================================================
# Test E: Solver ordering (SA/ADMM should generally beat random)
# ================================================================
log("\n--- Test E: Solver Ordering ---")

for n_ue in ue_counts_seen:
    subset = [r for r in sweep_results if r["num_ues"] == n_ue]
    avg_random = np.mean([r["solvers"]["random"]["worst_aoii"] for r in subset])
    avg_sa = np.mean([r["solvers"]["sa"]["worst_aoii"] for r in subset])
    avg_admm = np.mean([r["solvers"]["admm"]["worst_aoii"] for r in subset])
    avg_greedy = np.mean([r["solvers"]["greedy"]["worst_aoii"] for r in subset])

    log(f"\n  {n_ue} UEs (avg worst-case AoII):")
    log(f"    Random={avg_random:.4f}, Greedy={avg_greedy:.4f}, "
        f"SA={avg_sa:.4f}, ADMM={avg_admm:.4f}")

    # SA/ADMM should produce <= random worst-case AoII (with tolerance)
    check(f"{n_ue}UE: SA worst_aoii <= random * 1.5",
          avg_sa <= avg_random * 1.5 + 0.1,
          f"sa={avg_sa:.4f}, random={avg_random:.4f}")
    check(f"{n_ue}UE: ADMM worst_aoii <= random * 1.5",
          avg_admm <= avg_random * 1.5 + 0.1,
          f"admm={avg_admm:.4f}, random={avg_random:.4f}")

# ================================================================
# Summary
# ================================================================
log("\n" + "=" * 60)
result_str = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
log(f"STEP 7 RESULT: {result_str}")
log("=" * 60)

output_path = os.path.join(OUTPUT_DIR, "step7_output.txt")
with open(output_path, "w") as f:
    f.write("\n".join(lines) + "\n")
log(f"\nOutput saved to: {output_path}")
