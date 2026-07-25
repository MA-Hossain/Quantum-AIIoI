"""Step 5 Test: QAOA solver — QUBO→Ising, circuit, COBYLA, depth sweep, figure.

Test A — Ising conversion: energy equivalence on random binary vectors
Test B — QAOA p=1 on 16-qubit instance
Test C — Depth sweep p=1,2,3 with approximation ratios vs exact
Test D — Comparison figure: QAOA vs SA vs Exact
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sagin_research_sim.aoii.renewal import precompute_psi
from sagin_research_sim.solver.qubo import (
    build_vanguard_qubo,
    evaluate_qubo,
    extract_solution,
)
from sagin_research_sim.solver.solve_exact import solve_exact
from sagin_research_sim.solver.solve_classical import solve_sa, has_neal
from sagin_research_sim.solver.solve_qaoa import (
    qubo_to_ising,
    solve_qaoa,
    sweep_depths,
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
log("STEP 5: QAOA SOLVER")
log("=" * 60)

# ---------- Build a 16-qubit QUBO (2-UE / 2-node, L=4) ----------
P2 = np.array([[0.8, 0.2],
               [0.3, 0.7]])
PSI = precompute_psi(P2, Y_max=20)

rate_A = np.array([
    [1000.0, 500.0],
    [500.0, 1000.0],
])
cap_A = np.array([2, 2])

Q, idx_map = build_vanguard_qubo(
    num_ues=2, num_nodes=2,
    psi_table=PSI, rate_matrix=rate_A,
    capacity=cap_A,
    packet_size_bits=1000.0,
    num_levels=4, level_step=0.3,
    penalty_single_assoc=10.0,
    penalty_capacity=8.0,
    penalty_aoii=6.0,
    penalty_epigraph=10.0,
)
N = idx_map.num_vars
log(f"  QUBO: {N} variables (qubits), Q shape {Q.shape}")

# ================================================================
# Test A: QUBO → Ising conversion
# ================================================================
log("\n--- Test A: QUBO -> Ising Conversion ---")

h, J, offset = qubo_to_ising(Q)
log(f"  Ising: {N} spins, {len(J)} couplings, offset={offset:.4f}")

rng = np.random.default_rng(42)
max_err = 0.0
for _ in range(1000):
    x = rng.integers(0, 2, size=N).astype(float)
    qubo_e = float(x @ Q @ x)

    s = 1.0 - 2.0 * x
    ising_e = offset
    for i in range(N):
        ising_e += h[i] * s[i]
    for (i, j), jij in J.items():
        ising_e += jij * s[i] * s[j]

    max_err = max(max_err, abs(qubo_e - ising_e))

log(f"  Max |QUBO - Ising| over 1000 random x: {max_err:.2e}")
check("Ising conversion correct", max_err < 1e-9)

# ================================================================
# Test B: QAOA p=1
# ================================================================
log("\n--- Test B: QAOA p=1 ---")

res_qaoa1 = solve_qaoa(
    Q, index_map=idx_map, level_step=0.3, capacity=cap_A,
    p=1, shots=2048, maxiter=80, seed=42,
)

log(f"  Energy:      {res_qaoa1['energy']:.4f}")
log(f"  Opt energy:  {res_qaoa1['opt_energy']:.4f}")
log(f"  Association: {res_qaoa1['association']}")
log(f"  eta:         {res_qaoa1['eta']:.2f}")
log(f"  Feasible:    single_assoc={res_qaoa1['single_assoc']}, "
    f"capacity_ok={res_qaoa1['capacity_ok']}")
log(f"  COBYLA evals: {res_qaoa1['opt_nfev']}")
log(f"  Time:        {res_qaoa1['timing_s']:.1f}s")

check("QAOA p=1: returns a solution", res_qaoa1["energy"] < float("inf"))
check("QAOA p=1: energy is finite", np.isfinite(res_qaoa1["energy"]))

# ================================================================
# Exact reference
# ================================================================
log("\n--- Exact Reference ---")

res_exact = solve_exact(Q, index_map=idx_map, level_step=0.3, capacity=cap_A)
E_exact = res_exact["energy"]
log(f"  Exact energy:      {E_exact:.4f}")
log(f"  Exact association: {res_exact['association']}")
log(f"  Time:              {res_exact['timing_s']:.3f}s")

# ================================================================
# SA reference
# ================================================================
log("\n--- SA Reference ---")

res_sa = solve_sa(
    Q, index_map=idx_map, level_step=0.3, capacity=cap_A,
    num_reads=200, num_sweeps=1000, seed=42,
)
log(f"  SA energy:      {res_sa['energy']:.4f}")
log(f"  SA association: {res_sa['association']}")
log(f"  SA backend:     {res_sa['backend']}")
log(f"  Time:           {res_sa['timing_s']:.3f}s")

# ================================================================
# Test C: Depth sweep p = 1, 2, 3
# ================================================================
log("\n--- Test C: QAOA Depth Sweep p=1,2,3 ---")

qaoa_results = [res_qaoa1]  # reuse p=1

for p_depth in [2, 3]:
    log(f"\n  Running QAOA p={p_depth} ...")
    res_p = solve_qaoa(
        Q, index_map=idx_map, level_step=0.3, capacity=cap_A,
        p=p_depth, shots=2048, maxiter=120, seed=42,
    )
    qaoa_results.append(res_p)
    log(f"    Energy:      {res_p['energy']:.4f}")
    log(f"    Opt energy:  {res_p['opt_energy']:.4f}")
    log(f"    Association: {res_p['association']}")
    log(f"    Feasible:    single_assoc={res_p['single_assoc']}, "
        f"capacity_ok={res_p['capacity_ok']}")
    log(f"    COBYLA evals: {res_p['opt_nfev']}")
    log(f"    Time:        {res_p['timing_s']:.1f}s")

log(f"\n  Approximation ratios (energy / exact, closer to 1.0 = better):")
for res_p in qaoa_results:
    ratio = res_p["energy"] / E_exact if E_exact != 0 else 0
    log(f"    QAOA p={res_p['depth']}: energy={res_p['energy']:.4f}, "
        f"ratio={ratio:.4f}")

ratio_sa = res_sa["energy"] / E_exact if E_exact != 0 else 0
log(f"    SA:          energy={res_sa['energy']:.4f}, ratio={ratio_sa:.4f}")
log(f"    Exact:       energy={E_exact:.4f}, ratio=1.0000")

# Check QAOA finds reasonable solutions
best_qaoa = min(qaoa_results, key=lambda r: r["energy"])
best_ratio = best_qaoa["energy"] / E_exact if E_exact != 0 else 0
check("Best QAOA achieves ratio >= 0.5",
      best_ratio >= 0.5,
      f"ratio={best_ratio:.4f}")
check("Best QAOA energy <= 0 (negative = feasible regime)",
      best_qaoa["energy"] < 0,
      f"energy={best_qaoa['energy']:.4f}")

# Check higher depth tends to improve (or at least not degrade much)
energies_by_depth = [r["energy"] for r in qaoa_results]
check("QAOA p=3 energy <= QAOA p=1 energy + tolerance",
      energies_by_depth[2] <= energies_by_depth[0] + abs(E_exact) * 0.1,
      f"p1={energies_by_depth[0]:.4f}, p3={energies_by_depth[2]:.4f}")

# ================================================================
# Test D: Comparison figure
# ================================================================
log("\n--- Test D: Comparison Figure ---")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Left panel: QAOA convergence ---
for res_p in qaoa_results:
    p_val = res_p["depth"]
    energies = res_p["history"]["energies"]
    ax1.plot(range(1, len(energies) + 1), energies,
             label=f"QAOA p={p_val}", linewidth=1.5)

ax1.axhline(E_exact, color="black", linestyle="--", linewidth=1.2, label="Exact")
ax1.axhline(res_sa["energy"], color="tab:red", linestyle=":",
            linewidth=1.2, label=f"SA ({res_sa['backend']})")
ax1.set_xlabel("COBYLA Iteration")
ax1.set_ylabel("Expected QUBO Energy")
ax1.set_title("QAOA Convergence")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Right panel: approximation ratio bar chart ---
solver_names = ["Exact", f"SA\n({res_sa['backend']})",
                "QAOA\np=1", "QAOA\np=2", "QAOA\np=3"]
ratios = [1.0, ratio_sa]
for res_p in qaoa_results:
    r = res_p["energy"] / E_exact if E_exact != 0 else 0
    ratios.append(r)

colors = ["#2ecc71", "#e74c3c", "#3498db", "#2980b9", "#1a5276"]
bars = ax2.bar(solver_names, ratios, color=colors, edgecolor="black", linewidth=0.5)

ax2.set_ylabel("Approximation Ratio (energy / exact)")
ax2.set_title(f"Solver Comparison ({N} qubits)")
ax2.set_ylim(0, max(1.2, max(ratios) * 1.1))
ax2.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax2.grid(True, axis="y", alpha=0.3)

for bar, ratio in zip(bars, ratios):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{ratio:.3f}", ha="center", va="bottom", fontsize=9)

fig.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "step5_qaoa_comparison.png")
fig.savefig(fig_path, dpi=150)
plt.close(fig)
log(f"  Figure saved to: {fig_path}")
check("Figure file created", os.path.isfile(fig_path))

# ================================================================
# Summary
# ================================================================
log("\n" + "=" * 60)
result_str = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
log(f"STEP 5 RESULT: {result_str}")
log("=" * 60)

output_path = os.path.join(OUTPUT_DIR, "step5_output.txt")
with open(output_path, "w") as f:
    f.write("\n".join(lines) + "\n")
log(f"\nOutput saved to: {output_path}")
