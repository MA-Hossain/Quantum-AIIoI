"""Step 8 Test: exact minimax ground truth and solver-skip reporting.

Test A — Cross-validation: fast bottleneck solver vs brute-force enumeration
Test B — Feasibility:      returned assignments respect node capacities
Test C — Exp 1 wiring:     run_instance actually produces an "exact" entry
Test D — Skip reporting:   oversized solvers are recorded, never silent

Regression guard for the bug where Experiment 1 requested exact and QAOA
solvers, both were silently skipped on every instance because the QUBO was
larger than their gates, and the experiment still reported success.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sagin_research_sim.solver.solve_exact_assoc import (
    solve_exact_assoc,
    solve_exact_assoc_bruteforce,
)
from experiments.run_large_scale import run_instance

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

all_passed = True


def check(condition, label):
    global all_passed
    status = "PASS" if condition else "FAIL"
    if not condition:
        all_passed = False
    log(f"  [{status}] {label}")
    return condition


# ================================================================
# Test A — fast solver agrees with brute force
# ================================================================
log("=" * 60)
log("TEST A: bottleneck solver vs brute-force enumeration")
log("=" * 60)

rng = np.random.default_rng(12345)
mismatches = 0
trials = 200

for _ in range(trials):
    I = int(rng.integers(2, 8))
    M = int(rng.integers(2, 5))
    cost = np.round(rng.uniform(0.0, 20.0, size=(I, M)), 3)
    cap = rng.integers(1, I + 1, size=M)
    if cap.sum() < I:
        cap[np.argmin(cap)] += I - cap.sum()

    fast = solve_exact_assoc(cost, cap)
    brute = solve_exact_assoc_bruteforce(cost, cap)
    if abs(fast["worst_aoii"] - brute["worst_aoii"]) > 1e-9:
        mismatches += 1

check(mismatches == 0,
      f"{trials} random instances, {mismatches} objective mismatches")


# ================================================================
# Test B — returned assignments are capacity-feasible and complete
# ================================================================
log("\n" + "=" * 60)
log("TEST B: exact assignments respect capacity")
log("=" * 60)

violations = 0
incomplete = 0

for _ in range(trials):
    I = int(rng.integers(2, 10))
    M = int(rng.integers(2, 6))
    cost = np.round(rng.uniform(0.0, 20.0, size=(I, M)), 3)
    cap = rng.integers(1, I + 1, size=M)
    if cap.sum() < I:
        cap[np.argmin(cap)] += I - cap.sum()

    res = solve_exact_assoc(cost, cap)
    assoc = res["association"]
    if len(assoc) != I:
        incomplete += 1
        continue
    load = np.bincount([assoc[i] for i in range(I)], minlength=M)
    if np.any(load > cap):
        violations += 1

check(incomplete == 0, f"all {trials} assignments cover every UE")
check(violations == 0, f"all {trials} assignments respect node capacity")


# ================================================================
# Test C — Experiment 1 wiring produces exact ground truth
# ================================================================
log("\n" + "=" * 60)
log("TEST C: run_instance emits an 'exact' solver entry")
log("=" * 60)

res = run_instance(
    num_ues=5, seed=0, num_scenarios=2, severity="moderate",
    run_sa=True, run_admm=True, run_exact=True, run_qaoa=False,
    sa_reads=50, sa_sweeps=200, admm_restarts=2,
)

check("exact" in res["solvers"], "'exact' present in solver results")

if "exact" in res["solvers"]:
    opt = res["solvers"]["exact"]["worst_aoii"]
    log(f"       exact worst-case AoII = {opt:.6f}")
    check(res["solvers"]["exact"]["capacity_ok"],
          "exact solution is capacity-feasible")

    # No heuristic may beat the exact optimum.
    beaten = [
        name for name, m in res["solvers"].items()
        if name != "exact" and m["worst_aoii"] < opt - 1e-9
    ]
    check(not beaten,
          f"no solver beats the exact optimum (offenders: {beaten})")


# ================================================================
# Test D — oversized solvers are reported, not silently dropped
# ================================================================
log("\n" + "=" * 60)
log("TEST D: skipped solvers are recorded with a reason")
log("=" * 60)

check("skipped" in res, "'skipped' field present in results")

res_q = run_instance(
    num_ues=3, seed=0, num_scenarios=2, severity="moderate",
    run_sa=False, run_admm=False, run_exact=False, run_qaoa=True,
    qaoa_depth=1, qaoa_shots=256,
)

n_vars = res_q.get("num_vars")
log(f"       3-UE instance QUBO size: N = {n_vars}")

if n_vars is not None and n_vars > 20:
    check("qaoa" in res_q["skipped"],
          "oversized QAOA run recorded in 'skipped'")
    check("qaoa" not in res_q["solvers"],
          "oversized QAOA produced no phantom result")
    log(f"       reason: {res_q['skipped'].get('qaoa')}")
else:
    check("qaoa" in res_q["solvers"], "QAOA ran within the qubit limit")


# ================================================================
# Summary
# ================================================================
log("\n" + "=" * 60)
result = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
log(f"STEP 8 RESULT: {result}")
log("=" * 60)

output_path = os.path.join(OUTPUT_DIR, "step8_output.txt")
with open(output_path, "w") as f:
    f.write("\n".join(lines) + "\n")
log(f"\nOutput saved to: {output_path}")

sys.exit(0 if all_passed else 1)
