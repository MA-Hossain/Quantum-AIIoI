"""Step 3 Test: VANGUARD QUBO construction and validation.

Test A — Structural:   2-UE / 2-node, brute-force optimal vs infeasible
Test B — Toy instance: 5-UE / 2-BS / 2-SAT with known-good feasible check
Test C — Constraint violation: penalties raise energy for violated constraints
Test D — Integration:  uses AoII renewal module to build the Psi table
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import itertools
import numpy as np

from sagin_research_sim.aoii.renewal import precompute_psi, distortion_matrix
from sagin_research_sim.solver.qubo import (
    IndexMap,
    build_vanguard_qubo,
    compute_aoii_cost_matrix,
    evaluate_qubo,
    extract_solution,
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
log("STEP 3: VANGUARD QUBO CONSTRUCTION")
log("=" * 60)

# ---------- Shared Psi table from 2-state chain (Step 2) ----------
P2 = np.array([[0.8, 0.2],
               [0.3, 0.7]])
PSI = precompute_psi(P2, Y_max=20)

# ================================================================
# Test A: 2-UE / 2-node brute-force
# ================================================================
log("\n--- Test A: 2-UE / 2-Node Brute-Force ---")

# Rates: UE0 prefers Node0, UE1 prefers Node1
# packet_size = 1000 bits
#   rate=1000 => Y=1 => Psi(1)=0.24  (good)
#   rate=500  => Y=2 => Psi(2)=0.60  (worse)
rate_A = np.array([
    [1000.0, 500.0],   # UE0: node0=fast, node1=slow
    [500.0, 1000.0],   # UE1: node0=slow, node1=fast
])
cap_A = np.array([2, 2])  # each node can serve up to 2

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

N_A = idx_A.num_vars
log(f"  Variables: {N_A}")
log(f"  Q shape: {Q_A.shape}")
check("Q is square", Q_A.shape[0] == Q_A.shape[1])
check("Q dimension matches num_vars", Q_A.shape[0] == N_A)

# Variable summary
assoc_vars = [v for v in idx_A.variables if v.var_type == "assoc"]
epi_vars   = [v for v in idx_A.variables if v.var_type == "epigraph"]
cs_vars    = [v for v in idx_A.variables if v.var_type == "cap_slack"]
as_vars    = [v for v in idx_A.variables if v.var_type == "aoii_slack"]
log(f"  Breakdown: {len(assoc_vars)} assoc, {len(epi_vars)} epigraph, "
    f"{len(cs_vars)} cap_slack, {len(as_vars)} aoii_slack")

# Brute-force all 2^N solutions
log(f"  Brute-forcing {2**N_A} solutions ...")
best_energy = float("inf")
best_x = None
best_feasible_energy = float("inf")
best_feasible_x = None

for bits in itertools.product([0, 1], repeat=N_A):
    x = np.array(bits, dtype=float)
    e = evaluate_qubo(Q_A, x)
    sol = extract_solution(x, idx_A, level_step=0.3, capacity=cap_A)
    if e < best_energy:
        best_energy = e
        best_x = x.copy()
    if sol["single_assoc"] and sol["capacity_ok"] and sol["eta_level"] >= 0:
        if e < best_feasible_energy:
            best_feasible_energy = e
            best_feasible_x = x.copy()

best_sol = extract_solution(best_x, idx_A, level_step=0.3, capacity=cap_A)
log(f"  Global min energy: {best_energy:.4f}")
log(f"    association: {best_sol['association']}")
log(f"    eta_level: {best_sol['eta_level']}, eta: {best_sol['eta']:.2f}")
log(f"    single_assoc: {best_sol['single_assoc']}, capacity_ok: {best_sol['capacity_ok']}")

check("Global optimum is feasible", best_sol["single_assoc"] and best_sol["capacity_ok"])
check("Optimal assigns UE0->Node0", best_sol["association"].get(0) == 0,
      f"got {best_sol['association']}")
check("Optimal assigns UE1->Node1", best_sol["association"].get(1) == 1,
      f"got {best_sol['association']}")

if best_feasible_x is not None:
    feas_sol = extract_solution(best_feasible_x, idx_A, level_step=0.3, capacity=cap_A)
    log(f"  Best feasible energy: {best_feasible_energy:.4f}")
    log(f"    association: {feas_sol['association']}, eta: {feas_sol['eta']:.2f}")
    check("Global optimum equals best feasible",
          abs(best_energy - best_feasible_energy) < 1e-9)

# Build two known solutions to compare
# Good: UE0->N0, UE1->N1 (both get fast rate, AoII_d=1 each)
# Bad:  UE0->N1, UE1->N0 (both get slow rate, AoII_d=2 each)
aoii_A = compute_aoii_cost_matrix(2, 2, rate_A, PSI, 1000.0)
log(f"\n  AoII cost matrix (continuous):\n    {aoii_A}")

# ================================================================
# Test B: 5-UE / 2-BS / 2-SAT toy instance
# ================================================================
log("\n--- Test B: 5-UE / 2-BS / 2-SAT Toy Instance ---")

I_B, M_B = 5, 4
np.random.seed(42)
# Synthetic rates: each UE has one preferred node
rate_B = np.array([
    [1000, 400, 300, 200],  # UE0 prefers BS0
    [350, 1000, 250, 300],  # UE1 prefers BS1
    [200, 300, 1000, 400],  # UE2 prefers SAT0
    [250, 200, 350, 1000],  # UE3 prefers SAT1
    [800, 600, 400, 300],   # UE4 prefers BS0 (contention with UE0)
], dtype=float)

# Disruption scenario: satellite rates degrade
rate_B_disrupted = rate_B.copy()
rate_B_disrupted[:, 2] *= 0.3  # SAT0 degrades
rate_B_disrupted[:, 3] *= 0.5  # SAT1 degrades

cap_B = np.array([3, 3, 2, 2])  # BS can serve 3, SAT can serve 2

Q_B, idx_B = build_vanguard_qubo(
    num_ues=I_B, num_nodes=M_B,
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

N_B = idx_B.num_vars
log(f"  Variables: {N_B}")
log(f"  Q shape: {Q_B.shape}")

assoc_B = [v for v in idx_B.variables if v.var_type == "assoc"]
epi_B   = [v for v in idx_B.variables if v.var_type == "epigraph"]
cs_B    = [v for v in idx_B.variables if v.var_type == "cap_slack"]
as_B    = [v for v in idx_B.variables if v.var_type == "aoii_slack"]
log(f"  Breakdown: {len(assoc_B)} assoc, {len(epi_B)} epigraph, "
    f"{len(cs_B)} cap_slack, {len(as_B)} aoii_slack")

check("Q_B is square", Q_B.shape[0] == Q_B.shape[1])
check("Q_B dim matches", Q_B.shape[0] == N_B)

# AoII cost matrix
aoii_B = compute_aoii_cost_matrix(I_B, M_B, rate_B, PSI, 1000.0,
                                  scenario_rate_matrices=[rate_B_disrupted])
log(f"  Worst-case AoII cost matrix:")
for i in range(I_B):
    log(f"    UE{i}: [{', '.join(f'{aoii_B[i,m]:.4f}' for m in range(M_B))}]")

# Construct a good feasible solution: natural assignment
x_good = np.zeros(N_B)
good_assoc = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0}  # each UE to preferred node
for (ue, node), flat in idx_B._assoc.items():
    if good_assoc.get(ue) == node:
        x_good[flat] = 1

# Set epigraph level to cover worst-case AoII
max_aoii_d = 0
for ue, node in good_assoc.items():
    aoii_d = int(np.round(aoii_B[ue, node] / 0.3))
    max_aoii_d = max(max_aoii_d, aoii_d)
max_aoii_d = min(max_aoii_d, 7)  # clamp to L-1
x_good[idx_B.epigraph(max_aoii_d)] = 1

# Set capacity slack bits
for m in range(M_B):
    load = sum(1 for u, n in good_assoc.items() if n == m)
    slack_val = int(cap_B[m]) - load
    for b_idx in range(len([v for v in cs_B if v.m == m])):
        if slack_val & (1 << b_idx):
            x_good[idx_B.cap_slack(m, b_idx)] = 1

# Set AoII slack bits
aoii_slack_nbits = max(1, int(np.ceil(np.log2(max(8, 2)))))
for ue, node in good_assoc.items():
    aoii_d = int(np.round(aoii_B[ue, node] / 0.3))
    slack_val = max_aoii_d - aoii_d
    for b_idx in range(aoii_slack_nbits):
        if slack_val & (1 << b_idx):
            x_good[idx_B.aoii_slack(ue, b_idx)] = 1

e_good = evaluate_qubo(Q_B, x_good)
sol_good = extract_solution(x_good, idx_B, level_step=0.3, capacity=cap_B)
log(f"\n  Good solution energy: {e_good:.4f}")
log(f"    association: {sol_good['association']}")
log(f"    eta: {sol_good['eta']:.2f}, single_assoc: {sol_good['single_assoc']}, "
    f"capacity_ok: {sol_good['capacity_ok']}")

check("Good solution is feasible (single_assoc)", sol_good["single_assoc"])
check("Good solution is feasible (capacity)", sol_good["capacity_ok"])

# Construct a bad solution: UE0 assigned to two nodes (infeasible)
x_bad = x_good.copy()
x_bad[idx_B.assoc(0, 1)] = 1  # double-assign UE0
e_bad = evaluate_qubo(Q_B, x_bad)
sol_bad = extract_solution(x_bad, idx_B, level_step=0.3, capacity=cap_B)
log(f"\n  Double-assoc solution energy: {e_bad:.4f}")
log(f"    single_assoc: {sol_bad['single_assoc']}")
check("Double-association has higher energy", e_bad > e_good,
      f"bad={e_bad:.4f} vs good={e_good:.4f}")
check("Double-association detected as infeasible", not sol_bad["single_assoc"])

# Construct over-capacity solution
x_overcap = np.zeros(N_B)
overcap_assoc = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # all to node 0 (cap=3)
for (ue, node), flat in idx_B._assoc.items():
    if overcap_assoc.get(ue) == node:
        x_overcap[flat] = 1
x_overcap[idx_B.epigraph(max_aoii_d)] = 1
e_overcap = evaluate_qubo(Q_B, x_overcap)
sol_overcap = extract_solution(x_overcap, idx_B, level_step=0.3, capacity=cap_B)
log(f"\n  Over-capacity solution energy: {e_overcap:.4f}")
log(f"    capacity_ok: {sol_overcap['capacity_ok']}")
check("Over-capacity has higher energy", e_overcap > e_good,
      f"overcap={e_overcap:.4f} vs good={e_good:.4f}")
check("Over-capacity detected as infeasible", not sol_overcap["capacity_ok"])

# ================================================================
# Test C: Constraint violation detection
# ================================================================
log("\n--- Test C: Constraint Violations ---")

# Zero vector (no assignment)
x_zero = np.zeros(N_B)
e_zero = evaluate_qubo(Q_B, x_zero)
sol_zero = extract_solution(x_zero, idx_B, level_step=0.3, capacity=cap_B)
log(f"  Zero-vector energy: {e_zero:.4f}")
check("Zero vector has higher energy than feasible", e_zero > e_good,
      f"zero={e_zero:.4f} vs good={e_good:.4f}")
check("Zero vector: no single_assoc", not sol_zero["single_assoc"])

# ================================================================
# Test D: Integration with AoII renewal
# ================================================================
log("\n--- Test D: AoII Renewal Integration ---")

# Use the 4-state chain from Step 2
P4 = np.array([
    [0.7, 0.2, 0.05, 0.05],
    [0.1, 0.6, 0.2, 0.1],
    [0.05, 0.15, 0.6, 0.2],
    [0.05, 0.1, 0.15, 0.7],
])
psi4 = precompute_psi(P4, Y_max=20)

aoii_D = compute_aoii_cost_matrix(
    num_ues=5, num_nodes=4,
    rate_matrix=rate_B, psi_table=psi4,
    packet_size_bits=1000.0,
    scenario_rate_matrices=[rate_B_disrupted],
)
log(f"  AoII cost (4-state chain):")
for i in range(5):
    log(f"    UE{i}: [{', '.join(f'{aoii_D[i,m]:.4f}' for m in range(4))}]")

check("AoII costs are non-negative", (aoii_D >= 0).all())
check("AoII costs > 0 for positive rates", (aoii_D > 0).all(),
      "all links have positive Psi")

# Monotonicity: lower rate => higher Y => higher Psi
for i in range(5):
    for m1 in range(4):
        for m2 in range(4):
            if rate_B[i, m1] > rate_B[i, m2] and rate_B_disrupted[i, m1] >= rate_B_disrupted[i, m2]:
                check_val = aoii_D[i, m1] <= aoii_D[i, m2] + 1e-9
                if not check_val:
                    check(f"AoII monotone UE{i} N{m1} vs N{m2}", False,
                          f"rate {rate_B[i,m1]} > {rate_B[i,m2]} but AoII {aoii_D[i,m1]:.4f} > {aoii_D[i,m2]:.4f}")

# Build QUBO with 4-state chain
Q_D, idx_D = build_vanguard_qubo(
    num_ues=5, num_nodes=4,
    psi_table=psi4, rate_matrix=rate_B,
    capacity=cap_B,
    packet_size_bits=1000.0,
    num_levels=8, level_step=0.5,
    scenario_rate_matrices=[rate_B_disrupted],
)
check("4-state QUBO builds successfully", Q_D.shape[0] == idx_D.num_vars)
log(f"  4-state QUBO: {idx_D.num_vars} variables, Q shape {Q_D.shape}")

# ================================================================
# Summary
# ================================================================
log("\n" + "=" * 60)
result = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
log(f"STEP 3 RESULT: {result}")
log("=" * 60)

output_path = os.path.join(OUTPUT_DIR, "step3_output.txt")
with open(output_path, "w") as f:
    f.write("\n".join(lines) + "\n")
log(f"\nOutput saved to: {output_path}")
