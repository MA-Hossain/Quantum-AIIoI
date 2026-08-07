# Current Gaps and Limitations

## Claims That CAN Be Supported by Experiments

| Claim | How | Status |
|-------|-----|--------|
| SENTINEL reduces robust AoII under disruption | Exp 2, 4: ADMM beats greedy baselines | Implemented |
| Renewal decomposition reproduces simulated AoII | Step 2: hand-computed chain validation | Done (Step 2) |
| Solvers approach the exact minimax optimum | Exp 1: exact bottleneck assignment vs all solvers | Implemented |
| ADMM decomposition reduces subproblem size | Exp 3: domain size breakdown at 550 UEs | Implemented |
| QAOA solves small sub-QUBOs near-exact | Exp 6 only: QAOA as ADMM sub-solver | Implemented |
| QAOA integrates into ADMM pipeline | Exp 6: QAOA as sub-solver within ADMM | Implemented |
| Classical distributed framework scales | Exp 3: 100-550 UEs with ADMM + baselines | Implemented |

## Claims That CANNOT Be Supported

| Claim | Why |
|-------|-----|
| Quantum advantage over classical methods | Qiskit Aer is a classical simulator; QAOA gets 0.9968 vs SA's 1.0 |
| QAOA solves the full monolithic QUBO | Needs 42+ qubits at the smallest meaningful instance — see Gap 9 |
| QUBO global optimum equals the original optimum | Requires brute force over 2^N; N >= 42 makes this unverifiable |
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

**New evidence**: with exact ground truth now available (Gap 9), the tie can be
measured against the optimum rather than inferred from a solver-vs-solver
comparison. Experiments 1, 2, 4 and 5 now compute the exact optimum for every
instance. Reduced-seed runs (1-2 seeds; NOT publication statistics) show:

| Experiment | ADMM mean gap | Greedy+LS mean gap | Identical? |
|-----------|---------------|--------------------|------------|
| Exp 1 (3-8 UEs)   | 2.17%  | 2.17%  | yes, 6/6 instances |
| Exp 2 (10-100 UEs)| 6.93%  | 6.93%  | yes, 5/5 instances |
| Exp 4 (severity)  | 4.30%  | 4.30%  | yes, 4/4 instances |
| Exp 5 (S=1,3,5)   | 16.18% | 16.18% | yes, 6/6 instances |

**This does not support the expected fix.** The hypothesis was that severe
disruption and multi-scenario minimax would produce instances where greedy+LS
fails and ADMM's systematic exploration wins. Across every severity level
(mild through extreme) and every scenario count (S=1,3,5), ADMM and greedy+LS
returned *identical* worst-case AoII. Harder instances made both solvers worse
together, not ADMM better.

Two further observations:

- Both solvers leave real headroom — up to 28% above the optimum at S=3. The
  problem is not that greedy+LS is already optimal; it is that ADMM is not
  finding anything greedy+LS misses.
- **`admm_raw` returns the greedy warm start unchanged.** Comparing the actual
  association dictionaries (not just objective values), `admm_raw` is identical
  to `greedy_aoii` on **21 of 21** instances across Experiments 1, 2, 4 and 5.
  ADMM is warm-started from greedy-AoII (`warm_start=greedy_init` in
  `run_instance`) and is giving that warm start straight back. The final `admm`
  number is therefore produced entirely by the local-search post-processing,
  which is the same post-processing applied to `greedy_aoii_ls` — which is
  exactly why the two tie.

  This makes the tie a likely *implementation* problem rather than a negative
  scientific result: the QUBO decomposition is contributing nothing measurable.
  Diagnose before drawing any conclusion about ADMM's merit. Check whether
  `solve_admm` converges immediately, whether its sub-QUBO solutions are being
  written back, and whether the returned energy actually improves on the warm
  start.
- SA degrades badly at 50 UEs (up to 122% gap), consistent with Gap 2.

**Status**: Measurable at last. Needs a full-seed run to confirm, but the
reduced-seed signal is consistent across four independent experiments.

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

### 9. QUBO Encoding Overhead Blocks Full-Problem Quantum Validation

**Problem**: The monolithic QUBO needs

```
N = I*M  +  L  +  sum_m ceil(log2(C_m+1))  +  I*ceil(log2 L)
    assoc   epigraph   capacity slack         AoII slack
```

variables. At the smallest experiment instance (3 UEs, 4 nodes, L=10) that is
N=42; at 8 UEs it is 82. The epigraph and slack encodings dominate, so shrinking
the topology barely helps — the smallest configuration reaching 20 qubits is
either 2 nodes or L=2, both of which destroy what the experiment is meant to
test.

**Consequence**: QAOA and 2^N brute force cannot run on the full QUBO at any
defensible configuration. This was previously hidden: Experiment 1 requested
both solvers, both were silently skipped on every instance, and the experiment
still reported success while writing a blank `exp1_qaoa_ratio.png`.

**Fix applied**: Exact ground truth now comes from a bottleneck-assignment
solver on the original problem (threshold search + bipartite max-flow), which
is exact and polynomial-time, so it scales to any instance size. QAOA is scoped
to Experiment 6. All solver skips are recorded with reasons in the results JSON
and printed to stdout. Regression guard: `tests/step8_exact_validation_test.py`.

**Remaining**: a tighter QUBO encoding (e.g. unary-to-binary epigraph, or
dropping per-UE AoII slack via a different epigraph formulation) would reduce N
substantially and could bring small instances into QAOA range. Not attempted.

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
