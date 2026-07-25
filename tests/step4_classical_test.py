"""Step 4 Test: Classical solvers — exact brute-force and simulated annealing.

Test A — Exact solver on 2-UE / 2-node (matches Step 3 brute-force)
Test B — SA (neal) on 2-UE / 2-node (finds optimal or near-optimal)
Test C — SA (numpy fallback) on 2-UE / 2-node
Test D — SA on 5-UE / 4-node toy instance (feasible, lower than bad solution)
Test E — Exact vs SA consistency on 2-UE / 2-node
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sagin_research_sim.aoii.renewal import precompute_psi
from sagin_research_sim.solver.qubo import (
    build_vanguard_qubo,
    compute_aoii_cost_matrix,
    evaluate_qubo,
    extract_solution,
)
from sagin_research_sim.solver.solve_exact import solve_exact
from sagin_research_sim.solver.solve_classical import solve_sa, has_neal

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
log("STEP 4: CLASSICAL SOLVERS")
log("=" * 60)
log(f"  D-Wave neal available: {has_neal()}")

# ---------- Shared setup ----------
P2 = np.array([[0.8, 0.2],
               [0.3, 0.7]])
PSI = precompute_psi(P2, Y_max=20)

# 2-UE / 2-node instance (same as Step 3 Test A)
rate_A = np.array([
    [1000.0, 500.0],
    [500.0, 1000.0],
])
cap_A = np.array([2, 2])

Q_A, idx_A = build_vanguard_qubo(
    num_ues=2, num_nodes=2,
    psi_table=PSI, rate_matrix=rate_A,
    capacity=cap_A,
    packet_size_bits=1000.0,
    num_levels=5, level_step=0.3,
    penalty_single_assoc=10.0,
    penalty_capacity=8.0,
    penalty_aoii=6.0,
    penalty_epigraph=10.0,
)

# 5-UE / 4-node instance (same as Step 3 Test B)
rate_B = np.array([
    [1000, 400, 300, 200],
    [350, 1000, 250, 300],
    [200, 300, 1000, 400],
    [250, 200, 350, 1000],
    [800, 600, 400, 300],
], dtype=float)
rate_B_disrupted = rate_B.copy()
rate_B_disrupted[:, 2] *= 0.3
rate_B_disrupted[:, 3] *= 0.5
cap_B = np.array([3, 3, 2, 2])

Q_B, idx_B = build_vanguard_qubo(
    num_ues=5, num_nodes=4,
    psi_table=PSI, rate_matrix=rate_B,
    capacity=cap_B,
    packet_size_bits=1000.0,
    num_levels=8, level_step=0.3,
    penalty_single_assoc=10.0,
    penalty_capacity=8.0,
    penalty_aoii=6.0,
    penalty_epigraph=10.0,
    scenario_rate_matrices=[rate_B_disrupted],
)

# ================================================================
# Test A: Exact solver on 2-UE / 2-node
# ================================================================
log("\n--- Test A: Exact Solver (2-UE / 2-Node) ---")

res_exact = solve_exact(Q_A, index_map=idx_A, level_step=0.3, capacity=cap_A)

log(f"  Energy:      {res_exact['energy']:.4f}")
log(f"  Association: {res_exact['association']}")
log(f"  eta:         {res_exact['eta']:.2f}")
log(f"  Feasible:    single_assoc={res_exact['single_assoc']}, "
    f"capacity_ok={res_exact['capacity_ok']}")
log(f"  Evaluated:   {res_exact['num_evaluated']} solutions")
log(f"  Time:        {res_exact['timing_s']:.3f}s")

check("Exact: optimal is feasible",
      res_exact["single_assoc"] and res_exact["capacity_ok"])
check("Exact: UE0 -> Node0",
      res_exact["association"].get(0) == 0,
      f"got {res_exact['association']}")
check("Exact: UE1 -> Node1",
      res_exact["association"].get(1) == 1,
      f"got {res_exact['association']}")
check("Exact: energy matches Step 3",
      abs(res_exact["energy"] - (-93.7)) < 0.01,
      f"got {res_exact['energy']:.4f}, expected -93.7")

# ================================================================
# Test B: SA (neal) on 2-UE / 2-node
# ================================================================
log("\n--- Test B: SA Neal (2-UE / 2-Node) ---")

if has_neal():
    res_neal = solve_sa(
        Q_A, index_map=idx_A, level_step=0.3, capacity=cap_A,
        num_reads=200, num_sweeps=1000, seed=42, backend="neal",
    )
    log(f"  Energy:      {res_neal['energy']:.4f}")
    log(f"  Association: {res_neal['association']}")
    log(f"  eta:         {res_neal['eta']:.2f}")
    log(f"  Feasible:    single_assoc={res_neal['single_assoc']}, "
        f"capacity_ok={res_neal['capacity_ok']}")
    log(f"  Backend:     {res_neal['backend']}")
    log(f"  Time:        {res_neal['timing_s']:.3f}s")

    check("SA neal: finds optimal energy",
          abs(res_neal["energy"] - res_exact["energy"]) < 0.01,
          f"SA={res_neal['energy']:.4f} vs exact={res_exact['energy']:.4f}")
    check("SA neal: optimal is feasible",
          res_neal["single_assoc"] and res_neal["capacity_ok"])
    check("SA neal: correct association",
          res_neal["association"] == res_exact["association"],
          f"got {res_neal['association']}")
else:
    log("  [SKIP] D-Wave neal not installed")

# ================================================================
# Test C: SA (numpy fallback) on 2-UE / 2-node
# ================================================================
log("\n--- Test C: SA NumPy Fallback (2-UE / 2-Node) ---")

res_np = solve_sa(
    Q_A, index_map=idx_A, level_step=0.3, capacity=cap_A,
    num_reads=200, num_sweeps=1000, seed=42, backend="numpy",
)

log(f"  Energy:      {res_np['energy']:.4f}")
log(f"  Association: {res_np['association']}")
log(f"  eta:         {res_np['eta']:.2f}")
log(f"  Feasible:    single_assoc={res_np['single_assoc']}, "
    f"capacity_ok={res_np['capacity_ok']}")
log(f"  Backend:     {res_np['backend']}")
log(f"  Time:        {res_np['timing_s']:.3f}s")

check("SA numpy: finds optimal energy",
      abs(res_np["energy"] - res_exact["energy"]) < 0.01,
      f"SA={res_np['energy']:.4f} vs exact={res_exact['energy']:.4f}")
check("SA numpy: optimal is feasible",
      res_np["single_assoc"] and res_np["capacity_ok"])
check("SA numpy: correct association",
      res_np["association"] == res_exact["association"],
      f"got {res_np['association']}")

# ================================================================
# Test D: SA on 5-UE / 4-node toy instance
# ================================================================
log("\n--- Test D: SA on 5-UE / 4-Node Toy Instance ---")

# Reference: known-good energy from Step 3
x_good = np.zeros(idx_B.num_vars)
good_assoc = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0}
for (ue, node), flat in idx_B._assoc.items():
    if good_assoc.get(ue) == node:
        x_good[flat] = 1
aoii_B = compute_aoii_cost_matrix(5, 4, rate_B, PSI, 1000.0,
                                  scenario_rate_matrices=[rate_B_disrupted])
max_aoii_d = 0
for ue, node in good_assoc.items():
    aoii_d = int(np.round(aoii_B[ue, node] / 0.3))
    max_aoii_d = max(max_aoii_d, aoii_d)
max_aoii_d = min(max_aoii_d, 7)
x_good[idx_B.epigraph(max_aoii_d)] = 1
cap_slack_vars = [v for v in idx_B.variables if v.var_type == "cap_slack"]
for m in range(4):
    load = sum(1 for u, n in good_assoc.items() if n == m)
    slack_val = int(cap_B[m]) - load
    nbits = len([v for v in cap_slack_vars if v.m == m])
    for b in range(nbits):
        if slack_val & (1 << b):
            x_good[idx_B.cap_slack(m, b)] = 1
aoii_slack_nbits = max(1, int(np.ceil(np.log2(max(8, 2)))))
for ue, node in good_assoc.items():
    aoii_d = int(np.round(aoii_B[ue, node] / 0.3))
    slack_val = max_aoii_d - aoii_d
    for b in range(aoii_slack_nbits):
        if slack_val & (1 << b):
            x_good[idx_B.aoii_slack(ue, b)] = 1
e_good_ref = evaluate_qubo(Q_B, x_good)
log(f"  Reference good-solution energy: {e_good_ref:.4f}")

# Neal SA
if has_neal():
    res_d_neal = solve_sa(
        Q_B, index_map=idx_B, level_step=0.3, capacity=cap_B,
        num_reads=500, num_sweeps=2000, seed=42, backend="neal",
    )
    log(f"\n  Neal SA:")
    log(f"    Energy:      {res_d_neal['energy']:.4f}")
    log(f"    Association: {res_d_neal['association']}")
    log(f"    eta:         {res_d_neal['eta']:.2f}")
    log(f"    Feasible:    single_assoc={res_d_neal['single_assoc']}, "
        f"capacity_ok={res_d_neal['capacity_ok']}")
    log(f"    Time:        {res_d_neal['timing_s']:.3f}s")

    check("SA neal 5-UE: solution is feasible",
          res_d_neal["single_assoc"] and res_d_neal["capacity_ok"])
    check("SA neal 5-UE: energy <= reference",
          res_d_neal["energy"] <= e_good_ref + 0.01,
          f"SA={res_d_neal['energy']:.4f} vs ref={e_good_ref:.4f}")

# NumPy SA
res_d_np = solve_sa(
    Q_B, index_map=idx_B, level_step=0.3, capacity=cap_B,
    num_reads=500, num_sweeps=2000, seed=42, backend="numpy",
)
log(f"\n  NumPy SA:")
log(f"    Energy:      {res_d_np['energy']:.4f}")
log(f"    Association: {res_d_np['association']}")
log(f"    eta:         {res_d_np['eta']:.2f}")
log(f"    Feasible:    single_assoc={res_d_np['single_assoc']}, "
    f"capacity_ok={res_d_np['capacity_ok']}")
log(f"    Time:        {res_d_np['timing_s']:.3f}s")

check("SA numpy 5-UE: solution is feasible",
      res_d_np["single_assoc"] and res_d_np["capacity_ok"])
check("SA numpy 5-UE: energy <= reference",
      res_d_np["energy"] <= e_good_ref + 0.01,
      f"SA={res_d_np['energy']:.4f} vs ref={e_good_ref:.4f}")

# ================================================================
# Test E: Exact vs SA consistency on 2-UE / 2-node
# ================================================================
log("\n--- Test E: Solver Consistency ---")

log(f"  Exact energy:     {res_exact['energy']:.4f}")
if has_neal():
    log(f"  SA (neal) energy: {res_neal['energy']:.4f}")
log(f"  SA (numpy) energy: {res_np['energy']:.4f}")

if has_neal():
    check("Neal matches exact",
          abs(res_neal["energy"] - res_exact["energy"]) < 0.01)
check("NumPy matches exact",
      abs(res_np["energy"] - res_exact["energy"]) < 0.01)

# Approximation ratios on 5-UE instance
if has_neal():
    ratio_neal = res_d_neal["energy"] / e_good_ref if e_good_ref != 0 else 0
    log(f"  5-UE neal  approx ratio (vs ref): {ratio_neal:.4f}")
ratio_np = res_d_np["energy"] / e_good_ref if e_good_ref != 0 else 0
log(f"  5-UE numpy approx ratio (vs ref): {ratio_np:.4f}")

# ================================================================
# Summary
# ================================================================
log("\n" + "=" * 60)
result = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
log(f"STEP 4 RESULT: {result}")
log("=" * 60)

output_path = os.path.join(OUTPUT_DIR, "step4_output.txt")
with open(output_path, "w") as f:
    f.write("\n".join(lines) + "\n")
log(f"\nOutput saved to: {output_path}")
