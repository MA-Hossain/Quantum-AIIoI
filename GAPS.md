# Current Gaps and Limitations

## Claims That CAN Be Supported by Experiments

| Claim | How | Status |
|-------|-----|--------|
| SENTINEL reduces robust AoII under disruption | Exp 2, 4: ADMM beats greedy baselines | Implemented |
| Renewal decomposition reproduces simulated AoII | Step 2: hand-computed chain validation | Done (Step 2) |
| QUBO matches original optimization (small) | Exp 1: exact vs QUBO on 3-8 UE instances | Implemented |
| ADMM decomposition reduces subproblem size | Exp 3: domain size breakdown at 550 UEs | Implemented |
| QAOA solves small sub-QUBOs near-exact | Exp 1: QAOA approx ratio vs exact | Implemented |
| QAOA integrates into ADMM pipeline | Exp 6: QAOA as sub-solver within ADMM | Implemented |
| Classical distributed framework scales | Exp 3: 100-550 UEs with ADMM + baselines | Implemented |

## Claims That CANNOT Be Supported

| Claim | Why |
|-------|-----|
| Quantum advantage over classical methods | Qiskit Aer is a classical simulator; QAOA gets 0.9968 vs SA's 1.0 |
| Universal finite-depth QAOA approximation guarantee | Requires a mathematical proof, not simulation evidence |
| Global convergence of nonconvex binary ADMM | Binary ADMM is nonconvex; we show empirical convergence only |
| Superiority over every classical heuristic | Only compared against implemented baselines |

## Known Technical Gaps

### 1. ADMM vs Greedy+LS Differentiation

**Problem**: In Step 7 results, ADMM ties greedy+LS on 24/25 instances. The local
search post-processing equalizes the solvers — ADMM's QUBO decomposition provides
no measurable benefit over greedy+LS in current experiments.

**Expected fix**: Experiments 4-5 (disruption severity + multi-scenario) should create
harder instances where greedy+LS cannot easily find the global optimum and ADMM's
systematic exploration provides genuine improvement. Under severe disruption with
multiple scenarios, the minimax landscape becomes more complex.

**Status**: Implemented in experiments, needs lab execution to verify.

### 2. SA Degrades at Scale

**Problem**: SA on the full QUBO becomes worse than greedy+LS at 15+ UEs because
the QUBO penalty landscape is ill-conditioned — neal's single-bit-flip SA struggles
with the complex constraint encoding.

**Expected**: SA is not run at 550 UEs (too slow). ADMM decomposes the problem into
smaller sub-QUBOs where SA performs well. This is actually a positive result for the
paper: it motivates decomposition.

**Status**: Documented and handled by experimental design.

### 3. Formulation Completeness

**Current formulation implements**:
- UE-to-BS/SAT association (binary x_{i,m})
- Worst-case AoII minimization (minimax via epigraph)
- Capacity constraints
- Multiple disruption scenarios (robust optimization)

**Paper may claim but NOT yet implemented**:
- Update-generation-rate decisions (q variables)
- Discrete bandwidth allocation levels
- Discrete computing allocation levels
- Freshness-budget constraints (hard AoII threshold per UE)
- UE heterogeneous source models (currently all UEs share one P matrix)

**Impact**: If the paper claims joint optimization over all these variables, the
simulation only validates the association sub-problem. This gap must be reconciled
in the manuscript or the formulation extended.

### 4. Monte Carlo AoII Validation

**Problem**: Step 2 validates the renewal formula against hand-computed values for
2-state and 4-state chains, but does NOT compare against full Monte Carlo trajectory
simulation (generating actual Markov source trajectories and measuring AoII directly).

**Fix**: Add Monte Carlo simulation to verify that Psi(Y) matches average AoII from
10K+ trajectory samples. This is a straightforward addition.

**Status**: Not implemented. Low effort, high value.

### 5. QAOA Circuit Metrics

**Problem**: Step 5 reports approximation ratio but does NOT report:
- Circuit depth (after transpilation)
- Total gate count (1Q + 2Q)
- Feasibility rate of sampled bitstrings
- Distribution of solution quality across shots

**Fix**: Add detailed QAOA circuit analysis to Experiment 1 and 6 outputs.

**Status**: Partially implemented (qubit count and depth reported).

### 6. Statistical Rigor

**Current**: 5 seeds per configuration, mean +/- std reported.

**Needed for publication**:
- 95% confidence intervals
- Paired statistical tests (Wilcoxon signed-rank for same-instance comparison)
- Win/tie/loss tables
- Negative results reported honestly

**Status**: Not implemented. Add to post-processing after lab runs complete.

### 7. Heterogeneous Sources

**Problem**: All UEs use the same 4-state Markov source P4. Real SAGIN networks
have UEs with different source dynamics (fast-changing video, slow-changing sensors).

**Fix**: Generate per-UE transition matrices with varying volatility. High-volatility
sources benefit more from low-latency BS links; slow sources tolerate SAT links.
This would make the optimization problem more interesting and differentiate solvers.

**Status**: Not implemented. Medium effort.

### 8. Comparison with Robust Classical Optimization

**Problem**: No comparison with established robust optimization methods:
- Scenario-based stochastic programming
- Distributionally robust optimization
- Classical branch-and-bound on the original problem
- MILP solver (e.g., Gurobi) on the original formulation

**Impact**: Without these, we cannot claim SENTINEL is superior to classical robust
optimization — only that it beats greedy heuristics.

**Status**: Not implemented. Would strengthen the paper significantly.

## Recommended Priority

### High Priority (do before submission)
1. Run Experiments 1-6 on lab machine
2. Verify ADMM advantage appears under severe disruption (Exp 4-5)
3. Monte Carlo AoII validation
4. Add confidence intervals and statistical tests

### Medium Priority (strengthens paper)
5. Heterogeneous source models
6. Extended QAOA circuit metrics
7. Comparison with Gurobi/MILP on small instances

### Low Priority (honest but not blocking)
8. Full formulation with bandwidth/computing variables
9. Comparison with robust optimization baselines
10. Real quantum hardware execution (IBM Quantum)
