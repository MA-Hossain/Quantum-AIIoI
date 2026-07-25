"""Step 2 Test: AoII renewal computation.

Hand-computed reference for 2-state chain:
    P = [[0.8, 0.2],
         [0.3, 0.7]]
    pi = [0.6, 0.4]
    F  = [[0, 1], [1, 0]]

    Y=1: q(1) = 0.6*0.2*1 + 0.4*0.3*1 = 0.24          => Psi(1) = 0.24
    Y=2: P^2 = [[0.70,0.30],[0.45,0.55]]
          q(2) = 0.6*0.30 + 0.4*0.45 = 0.36            => Psi(2) = 0.60
    Y=3: P^3 = [[0.65,0.35],[0.525,0.475]]
          q(3) = 0.6*0.35 + 0.4*0.525 = 0.42           => Psi(3) = 1.02
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sagin_research_sim.aoii.renewal import (
    stationary_distribution,
    distortion_matrix,
    psi,
    precompute_psi,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

all_passed = True
def check(name, got, expected, tol=1e-9):
    global all_passed
    ok = abs(got - expected) < tol
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_passed = False
    log(f"  [{status}] {name}: got={got:.10f}, expected={expected:.10f}")
    return ok


log("=" * 60)
log("STEP 2: AoII RENEWAL COMPUTATION")
log("=" * 60)

# ---- 2-state chain ----
log("\n--- Test A: 2-State Markov Chain ---")
P2 = np.array([[0.8, 0.2],
               [0.3, 0.7]])
F2 = distortion_matrix(2)

log(f"  P = {P2.tolist()}")
log(f"  F = {F2.tolist()}")

pi2 = stationary_distribution(P2)
log(f"  pi = [{pi2[0]:.6f}, {pi2[1]:.6f}]")
check("pi[0]", pi2[0], 0.6)
check("pi[1]", pi2[1], 0.4)

# Check pi @ P = pi
pi_check = pi2 @ P2
check("pi@P[0]", pi_check[0], pi2[0], tol=1e-12)
check("pi@P[1]", pi_check[1], pi2[1], tol=1e-12)

check("psi(Y=1)", psi(P2, F2, 1), 0.24)
check("psi(Y=2)", psi(P2, F2, 2), 0.60)
check("psi(Y=3)", psi(P2, F2, 3), 1.02)

# Test precompute matches individual calls
table2 = precompute_psi(P2, Y_max=10)
log(f"\n  Precomputed Psi table (Y=0..10):")
for y in range(11):
    psi_y = psi(P2, F2, y) if y > 0 else 0.0
    match = abs(table2[y] - psi_y) < 1e-12
    if not match:
        all_passed = False
    log(f"    Y={y:2d}: Psi={table2[y]:.6f}  (direct={psi_y:.6f}, match={match})")

# ---- 4-state chain (default for the project) ----
log("\n--- Test B: 4-State Markov Chain ---")
P4 = np.array([
    [0.7, 0.2, 0.05, 0.05],
    [0.1, 0.6, 0.2, 0.1],
    [0.05, 0.15, 0.6, 0.2],
    [0.05, 0.1, 0.15, 0.7],
])
F4 = distortion_matrix(4)
log(f"  P = {P4.tolist()}")
log(f"  F (distortion |i-j|):\n    {F4.tolist()}")

pi4 = stationary_distribution(P4)
log(f"  pi = [{', '.join(f'{p:.6f}' for p in pi4)}]")

# Verify stationary
pi4_check = pi4 @ P4
for s in range(4):
    check(f"pi@P[{s}]", pi4_check[s], pi4[s], tol=1e-10)

# Verify sum to 1
check("sum(pi)", pi4.sum(), 1.0, tol=1e-12)

# Precompute and show table
table4 = precompute_psi(P4, Y_max=20)
log(f"\n  Precomputed Psi table (Y=0..20):")
for y in range(21):
    log(f"    Y={y:2d}: Psi={table4[y]:.6f}")

# Verify monotonicity (Psi should be non-decreasing)
monotonic = all(table4[y] <= table4[y+1] for y in range(20))
status = "PASS" if monotonic else "FAIL"
if not monotonic:
    all_passed = False
log(f"\n  [{status}] Monotonicity check: Psi(Y) <= Psi(Y+1)")

# Verify Psi(0) = 0
check("Psi(0)", table4[0], 0.0)

# ---- Default f=|i-j| auto-generation ----
log("\n--- Test C: Auto-distortion (f=None) ---")
table_auto = precompute_psi(P2, Y_max=5)
table_explicit = precompute_psi(P2, Y_max=5, f=F2)
match_all = np.allclose(table_auto, table_explicit)
status = "PASS" if match_all else "FAIL"
if not match_all:
    all_passed = False
log(f"  [{status}] precompute_psi(f=None) matches precompute_psi(f=|i-j|)")

log("\n" + "=" * 60)
result = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
log(f"STEP 2 RESULT: {result}")
log("=" * 60)

output_path = os.path.join(OUTPUT_DIR, "step2_output.txt")
with open(output_path, "w") as f:
    f.write("\n".join(lines) + "\n")
log(f"\nOutput saved to: {output_path}")
