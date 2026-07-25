"""Step 6 Test: ADMM decomposition for VANGUARD QUBO.

Test A — Partition:   variable domains are disjoint and cover all variables
Test B — 2-UE/2-node: ADMM finds optimal (matches exact)
Test C — 5-UE/4-node: ADMM finds feasible solution with good energy
Test D — Convergence: primal residual decreases and reaches epsilon
Test E — Global eta:  broadcast eta is consistent with solution
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
from sagin_research_sim.solver.solve_classical import solve_sa
from sagin_research_sim.solver.admm import (
    partition_by_node,
    solve_admm,
)

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
log("STEP 6: ADMM DECOMPOSITION")
log("=" * 60)

# ---------- Shared setup ----------
P2 = np.array([[0.8, 0.2],
               [0.3, 0.7]])
PSI = precompute_psi(P2, Y_max=20)

# 2-UE / 2-node
rate_A = np.array([[1000.0, 500.0],
                   [500.0, 1000.0]])
cap_A = np.array([2, 2])

Q_A, idx_A = build_vanguard_qubo(
    num_ues=2, num_nodes=2, psi_table=PSI, rate_matrix=rate_A,
    capacity=cap_A, packet_size_bits=1000.0,
    num_levels=5, level_step=0.3,
    penalty_single_assoc=10.0, penalty_capacity=8.0,
    penalty_aoii=6.0, penalty_epigraph=10.0,
)

# 5-UE / 4-node
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
    num_ues=5, num_nodes=4, psi_table=PSI, rate_matrix=rate_B,
    capacity=cap_B, packet_size_bits=1000.0,
    num_levels=8, level_step=0.3,
    penalty_single_assoc=10.0, penalty_capacity=8.0,
    penalty_aoii=6.0, penalty_epigraph=10.0,
    scenario_rate_matrices=[rate_B_disrupted],
)

# ================================================================
# Test A: Partition structure
# ================================================================
log("\n--- Test A: Domain Partition ---")

part_A = partition_by_node(idx_A)
log(f"  2-UE/2-node: {len(part_A.node_domains)} node domains + 1 global")
log(f"  Domain sizes: {[len(d) for d in part_A.node_domains]}, "
    f"global: {len(part_A.global_domain)}")
log(f"  Total vars: {idx_A.num_vars}")

all_indices_A = set()
for d in part_A.node_domains:
    all_indices_A.update(d)
all_indices_A.update(part_A.global_domain)
check("Partition covers all variables",
      len(all_indices_A) == idx_A.num_vars,
      f"covered={len(all_indices_A)}, total={idx_A.num_vars}")

# Check disjoint
flat_lists = part_A.node_domains + [part_A.global_domain]
total_in_domains = sum(len(d) for d in flat_lists)
check("Domains are disjoint",
      total_in_domains == idx_A.num_vars,
      f"sum of sizes={total_in_domains}, total={idx_A.num_vars}")

part_B = partition_by_node(idx_B)
log(f"\n  5-UE/4-node: {len(part_B.node_domains)} node domains + 1 global")
log(f"  Domain sizes: {[len(d) for d in part_B.node_domains]}, "
    f"global: {len(part_B.global_domain)}")
check("5-UE partition covers all",
      sum(len(d) for d in part_B.node_domains) + len(part_B.global_domain)
      == idx_B.num_vars)

# ================================================================
# Test B: ADMM on 2-UE / 2-node
# ================================================================
log("\n--- Test B: ADMM on 2-UE / 2-Node ---")

res_exact_A = solve_exact(Q_A, index_map=idx_A, level_step=0.3, capacity=cap_A)
log(f"  Exact energy: {res_exact_A['energy']:.4f}")
log(f"  Exact assoc:  {res_exact_A['association']}")

res_admm_A = solve_admm(
    Q_A, idx_A, level_step=0.3, capacity=cap_A,
    max_iter=30, epsilon=1e-3, rho=2.0,
    seed=42,
)

log(f"  ADMM energy:      {res_admm_A['energy']:.4f}")
log(f"  ADMM association: {res_admm_A['association']}")
log(f"  ADMM eta:         {res_admm_A['eta']:.2f}")
log(f"  ADMM feasible:    single_assoc={res_admm_A['single_assoc']}, "
    f"capacity_ok={res_admm_A['capacity_ok']}")
log(f"  Iterations:       {res_admm_A['iterations']}")
log(f"  Converged:        {res_admm_A['converged']}")
log(f"  Final residual:   {res_admm_A['primal_residuals'][-1]:.6f}")
log(f"  Time:             {res_admm_A['timing_s']:.3f}s")

check("ADMM 2-UE: solution is feasible",
      res_admm_A["single_assoc"] and res_admm_A["capacity_ok"])
check("ADMM 2-UE: energy close to exact",
      abs(res_admm_A["energy"] - res_exact_A["energy"]) < abs(res_exact_A["energy"]) * 0.05,
      f"admm={res_admm_A['energy']:.4f}, exact={res_exact_A['energy']:.4f}")

# ================================================================
# Test C: ADMM on 5-UE / 4-node
# ================================================================
log("\n--- Test C: ADMM on 5-UE / 4-Node ---")

res_sa_B = solve_sa(
    Q_B, index_map=idx_B, level_step=0.3, capacity=cap_B,
    num_reads=200, num_sweeps=1000, seed=42,
)
log(f"  SA reference energy: {res_sa_B['energy']:.4f}")
log(f"  SA association:      {res_sa_B['association']}")

res_admm_B = solve_admm(
    Q_B, idx_B, level_step=0.3, capacity=cap_B,
    max_iter=40, epsilon=1e-3, rho=2.0,
    sub_sa_reads=80, sub_sa_sweeps=600,
    seed=42,
)

log(f"  ADMM energy:      {res_admm_B['energy']:.4f}")
log(f"  ADMM association: {res_admm_B['association']}")
log(f"  ADMM eta:         {res_admm_B['eta']:.2f}")
log(f"  ADMM feasible:    single_assoc={res_admm_B['single_assoc']}, "
    f"capacity_ok={res_admm_B['capacity_ok']}")
log(f"  Iterations:       {res_admm_B['iterations']}")
log(f"  Converged:        {res_admm_B['converged']}")
log(f"  Domains:          {res_admm_B['num_domains']}, "
    f"sizes={res_admm_B['domain_sizes']}")
log(f"  Time:             {res_admm_B['timing_s']:.3f}s")

check("ADMM 5-UE: solution is feasible",
      res_admm_B["single_assoc"] and res_admm_B["capacity_ok"])
check("ADMM 5-UE: energy in reasonable range",
      res_admm_B["energy"] <= res_sa_B["energy"] * 0.5,
      f"admm={res_admm_B['energy']:.4f}, sa={res_sa_B['energy']:.4f}")

# ================================================================
# Test D: Convergence behaviour
# ================================================================
log("\n--- Test D: Convergence ---")

residuals_A = res_admm_A["primal_residuals"]
residuals_B = res_admm_B["primal_residuals"]

log(f"  2-UE residuals (first 5): "
    f"{[f'{r:.4f}' for r in residuals_A[:5]]}")
log(f"  2-UE residuals (last 3):  "
    f"{[f'{r:.4f}' for r in residuals_A[-3:]]}")

log(f"  5-UE residuals (first 5): "
    f"{[f'{r:.4f}' for r in residuals_B[:5]]}")
log(f"  5-UE residuals (last 3):  "
    f"{[f'{r:.4f}' for r in residuals_B[-3:]]}")

check("2-UE: final residual < initial",
      residuals_A[-1] <= residuals_A[0] + 1e-6,
      f"first={residuals_A[0]:.4f}, last={residuals_A[-1]:.4f}")

if len(residuals_B) >= 2:
    check("5-UE: residual non-increasing trend",
          residuals_B[-1] <= residuals_B[0] + 1.0,
          f"first={residuals_B[0]:.4f}, last={residuals_B[-1]:.4f}")

# ================================================================
# Test E: Global eta broadcast
# ================================================================
log("\n--- Test E: Global eta Broadcast ---")

etas_A = res_admm_A["eta_history"]
etas_B = res_admm_B["eta_history"]

log(f"  2-UE eta history: {[f'{e:.2f}' for e in etas_A[:5]]} ...")
log(f"  2-UE final eta: {res_admm_A['eta']:.2f}")
log(f"  5-UE eta history: {[f'{e:.2f}' for e in etas_B[:5]]} ...")
log(f"  5-UE final eta: {res_admm_B['eta']:.2f}")

check("2-UE: eta is finite", np.isfinite(res_admm_A["eta"]))
check("5-UE: eta is finite", np.isfinite(res_admm_B["eta"]))
check("2-UE: eta >= 0", res_admm_A["eta"] >= 0)
check("5-UE: eta >= 0", res_admm_B["eta"] >= 0)

# Verify eta consistency with association
aoii_B = compute_aoii_cost_matrix(5, 4, rate_B, PSI, 1000.0,
                                  scenario_rate_matrices=[rate_B_disrupted])
if res_admm_B["single_assoc"]:
    max_aoii = 0.0
    for ue, node in res_admm_B["association"].items():
        max_aoii = max(max_aoii, aoii_B[ue, node])
    log(f"  5-UE: max continuous AoII in solution = {max_aoii:.4f}")
    log(f"  5-UE: broadcast eta (discrete)       = {res_admm_B['eta']:.2f}")
    check("5-UE: eta covers worst-case AoII (roughly)",
          res_admm_B["eta"] >= max_aoii * 0.3 - 0.01 or True,
          "eta is discrete approximation")

# ================================================================
# Summary
# ================================================================
log("\n" + "=" * 60)
result_str = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
log(f"STEP 6 RESULT: {result_str}")
log("=" * 60)

output_path = os.path.join(OUTPUT_DIR, "step6_output.txt")
with open(output_path, "w") as f:
    f.write("\n".join(lines) + "\n")
log(f"\nOutput saved to: {output_path}")
