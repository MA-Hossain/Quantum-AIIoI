# Results — Exact-Optimum Validation Run

Date: 2026-08-06
Machine: 24-core Linux workstation, 62 GB RAM, Python 3.10.12
Commit baseline: `0fbca2f` ("ready for lab testbed")

> **Statistical caveat, read first.** Every number below comes from a
> **reduced-seed** run (1–2 seeds per configuration), executed to validate the
> new measurement machinery — not to produce publication statistics. The
> tables show effect direction and magnitude, not confidence intervals. A
> full-seed run is still required before any of this goes in the manuscript.
> The one exception is the warm-start finding in Section 4, which is an exact
> structural identity (21/21 instances) and does not depend on seed count.

---

## 1. What prompted this run

Experiment 1 was reporting success while validating nothing. It requested an
exact solver and a QAOA solver; both were gated on QUBO size, both exceeded
their gate on every instance, and both were skipped **silently**. The
experiment printed "completed", exited 0 in 2.2 s, and wrote a blank
`exp1_qaoa_ratio.png` containing empty axes.

Root cause: the monolithic QUBO is far larger than the gates assumed.

```
N = I*M  +  L  +  sum_m ceil(log2(C_m+1))  +  I*ceil(log2 L)
    assoc   epigraph   capacity slack         AoII slack
```

At the smallest experiment instance (3 UEs, 4 nodes, L=10) that is **N = 42**,
against gates of 25 (exact) and 20 (QAOA). At 8 UEs it is 82.

## 2. Why the QUBO cannot be shrunk into QAOA range

The overhead is structural — epigraph and slack encodings dominate, so
shrinking the topology barely helps:

| Configuration | N | Verdict |
|---|---|---|
| I=3, M=4, L=10 (as shipped) | 42 | far over |
| I=3, M=4, L=4 | 30 | still over |
| I=3, M=4, L=2 | 25 | still over, and L=2 destroys the epigraph |
| I=3, M=3, L=2 | 20 | fits, but degenerate |
| I=3, M=2, L=4 | 20 | fits, but 2 nodes is barely a network |

**Conclusion:** QAOA cannot run on the full monolithic QUBO at any
scientifically defensible configuration. This is a property of the encoding,
not a tuning problem. QAOA validation therefore belongs in Experiment 6, where
ADMM decomposition produces genuinely small sub-QUBOs.

Recorded as Gap 9 in [GAPS.md](GAPS.md).

## 3. Exact ground truth is now available at any scale

The 2^N QUBO brute force was the wrong tool. The claim being tested — "the
solvers approach the true optimum" — concerns the **original** problem:

```
minimise    max_i  aoii_cost[i, a(i)]
subject to  |{ i : a(i) = m }|  <=  capacity[m]
```

`aoii_cost` already folds in the worst case across disruption scenarios, so
this is exactly the robust objective the metrics report.

This is a **bottleneck assignment** problem. It is solved exactly by binary
searching the distinct cost values and testing feasibility at each threshold
with a bipartite max-flow. Implementation:
[`sagin_research_sim/solver/solve_exact_assoc.py`](sagin_research_sim/solver/solve_exact_assoc.py).

**Correctness.** Cross-validated against a brute-force `M^I` enumeration on
500 random instances: 0 objective mismatches, 0 capacity violations, all
assignments complete.

**Cost.** Polynomial, and in practice negligible — 5 max-flow solves regardless
of instance size:

| Instance | Nodes | Time |
|---|---|---|
| 50 UEs | 5 | 88.1 ms |
| 100 UEs | 8 | 2.0 ms |
| 200 UEs | 12 | 3.3 ms |
| 550 UEs | 15 | 7.7 ms |

Exact ground truth is therefore available for **every** experiment, not just
enumerable toy instances. It is now enabled in Experiments 1, 2, 4 and 5.

## 4. Principal finding: ADMM returns its warm start unchanged

ADMM is warm-started from the greedy-AoII solution
(`warm_start=greedy_init` in `run_instance`). Comparing the actual
`association` dictionaries — not merely the objective values — `admm_raw` is
**identical to `greedy_aoii` on 21 of 21 instances**:

| Results file | Identical association | Identical objective |
|---|---|---|
| `exp1_small_validation.json` | 6/6 | 6/6 |
| `exp2_solver_comparison.json` | 5/5 | 5/5 |
| `exp4_disruption_severity.json` | 4/4 | 4/4 |
| `exp5_multi_scenario.json` | 6/6 | 6/6 |

ADMM is handing back the solution it was given. The reported `admm` figure is
produced entirely by the `_local_search` post-processing — which is the *same*
post-processing applied to `greedy_aoii_ls`.

**This explains the long-standing ADMM/greedy+LS tie recorded as Gap 1.** The
two solvers tie because, after the warm start is returned unchanged, they are
running the identical local search on the identical starting point.

The QUBO decomposition is contributing nothing measurable. This should be
treated as an implementation defect to diagnose, **not** as a negative
scientific result about ADMM's merit. Until it is resolved, no comparison
involving ADMM in this repository carries meaning.

Suggested diagnosis path: does `solve_admm` hit its stopping criterion
immediately; are sub-QUBO solutions written back into the global solution; does
the returned `energy` ever improve on the warm start's energy; is the
sub-solver actually invoked.

## 5. Optimality gaps against exact ground truth

Percentage above the exact minimax optimum. Lower is better; 0.00% is optimal.

### Aggregate

| Experiment | greedy_aoii | greedy_aoii_ls | sa | admm_raw | admm |
|---|---|---|---|---|---|
| Exp 1 (3–8 UEs, 2 seeds) | 27.72% | **2.17%** | 12.49% | 27.72% | **2.17%** |
| Exp 2 (10–100 UEs, 1 seed) | 22.65% | **6.93%** | 34.10% | 22.65% | **6.93%** |
| Exp 4 (severity sweep, 1 seed) | 21.91% | **4.30%** | 55.11% | 21.91% | **4.30%** |
| Exp 5 (S=1,3,5, 2 seeds) | 40.10% | **16.18%** | 60.91% | 40.10% | **16.18%** |

`admm` and `greedy_aoii_ls` match to the last digit in every row, as do
`admm_raw` and `greedy_aoii` — the identity described in Section 4.

### Experiment 2 — by network size

| UEs | greedy_aoii | greedy_aoii_ls | sa | admm_raw | admm |
|---|---|---|---|---|---|
| 10 | 0.00% | 0.00% | 26.07% | 0.00% | 0.00% |
| 25 | 34.69% | 0.00% | 34.69% | 34.69% | 0.00% |
| 50 | 20.75% | 0.00% | 41.52% | 20.75% | 0.00% |
| 75 | 34.69% | 11.55% | not run | 34.69% | 11.55% |
| 100 | 23.12% | 23.12% | not run | 23.12% | 23.12% |

Note the gap growing with network size — at 100 UEs even the local search is
23% off optimal, and greedy+LS has stopped helping at all.

### Experiment 4 — by disruption severity

| Severity | greedy_aoii | greedy_aoii_ls | sa | admm_raw | admm |
|---|---|---|---|---|---|
| mild | 26.07% | 0.00% | 78.42% | 26.07% | 0.00% |
| moderate | 20.75% | 0.00% | 41.52% | 20.75% | 0.00% |
| severe | 34.42% | 17.21% | 17.21% | 34.42% | 17.21% |
| extreme | 6.40% | 0.00% | 83.27% | 6.40% | 0.00% |

**The Gap 1 hypothesis is not supported.** The expectation was that severe
disruption would produce instances where greedy+LS fails and ADMM's systematic
exploration wins. ADMM and greedy+LS are identical at every severity level.
Given Section 4, this is unsurprising — but it means Experiment 4 currently
provides no evidence for the SENTINEL advantage claim.

### Experiment 5 — by scenario count

| S | greedy_aoii | greedy_aoii_ls | sa | admm_raw | admm |
|---|---|---|---|---|---|
| 1 | 13.79% | 8.60% | 39.76% | 13.79% | 8.60% |
| 3 | 54.87% | 22.72% | 69.81% | 54.87% | 22.72% |
| 5 | 51.64% | 17.21% | 73.16% | 51.64% | 17.21% |

Multi-scenario minimax does produce genuinely harder instances — the gap roughly
doubles from S=1 to S=3. There is real headroom here (up to 28% above optimal on
individual instances). It is simply not being captured by ADMM.

## 6. Simulated annealing degrades badly at scale

SA reaches up to **122% above optimal** at 50 UEs with S=3 (Exp 5), and 83% at
extreme severity (Exp 4). It is beaten by plain greedy-AoII in most
configurations at 50+ UEs.

This corroborates Gap 2 and, as that entry argues, is arguably a *positive*
result for the paper: it is a concrete motivation for decomposition, since the
monolithic QUBO's penalty landscape is demonstrably too ill-conditioned for
single-bit-flip SA. Worth stating quantitatively in the manuscript now that the
distance from optimal can be measured rather than guessed.

One counter-observation: on the Exp 1 instance where both ADMM and greedy+LS
missed the optimum by 13%, SA found it. SA is erratic, not uniformly worse.

## 7. Changes made

### New files
- `sagin_research_sim/solver/solve_exact_assoc.py` — exact bottleneck-assignment
  solver (threshold search + max-flow) plus a brute-force reference used to
  validate it.
- `tests/step8_exact_validation_test.py` — regression guard: cross-validates
  both exact solvers, checks capacity feasibility, asserts Experiment 1 emits an
  `exact` entry, asserts no solver ever beats the optimum, and asserts oversized
  solvers are recorded rather than silently dropped.

### Modified
- `experiments/run_large_scale.py`
  - Exact ground truth wired into Experiments 1, 2, 4, 5.
  - `run_exact` (original problem) split from `run_exact_qubo` (2^N brute
    force) so large experiments do not emit meaningless skip messages.
  - New `skipped` field in the results JSON: every solver that cannot run is
    recorded with a reason and printed to stdout. **Silent skips are gone.**
  - Optimality-gap tables printed per experiment, grouped by the swept variable
    (aggregating across the sweep would hide the effect being measured).
  - QAOA removed from Experiment 1, with the reason stored per-instance.
- `experiments/plots_extended.py`
  - New `exp{1,2,4,5}_optimality_gap.png` figures from a shared helper handling
    numeric and categorical axes.
  - `exact` added to the plotted solver sets.
  - Fixed: `exp1_qaoa_ratio.png` was written unconditionally, outside its own
    `if` guard, producing a blank chart whenever QAOA was skipped.
  - Fixed: severity axis sorted alphabetically (`extreme, mild, moderate,
    severe`), which inverted the apparent trend. Now ordinal.
- `GAPS.md` — Gap 1 updated with measured evidence and the warm-start finding;
  Gap 9 added for the QUBO encoding overhead; claims table corrected.
- `README.md` — experiment descriptions, solver lists, figure inventory.

## 8. Verification status

| Check | Result |
|---|---|
| `tests/step1_smoke_test.py` | pass |
| `tests/step2_aoii_test.py` | pass |
| `tests/step3_qubo_test.py` | pass |
| `tests/step4_classical_test.py` | pass |
| `tests/step5_qaoa_test.py` | pass |
| `tests/step6_admm_test.py` | **fail — pre-existing** |
| `tests/step7_experiment_test.py` | pass |
| `tests/step8_exact_validation_test.py` | pass (new) |
| Exact solver vs brute force, 500 random instances | 0 mismatches |
| `--tier quick` | pass |
| Experiments 2, 4, 5 (reduced seeds) | pass |

`step6_admm_test.py` fails with `KeyError: 'eta_history'` at line 226. This was
verified to fail identically on a clean checkout (via `git stash`) and is
unrelated to this work: the key is referenced only in the test and never
produced by `solve_admm`.

## 9. Recommended next steps

1. **Diagnose the ADMM warm-start defect (Section 4) before anything else.**
   Until it is fixed, no ADMM comparison in this repository is meaningful, and
   a full-seed run would only reproduce the same tie with tighter error bars.
2. Fix `step6_admm_test.py` — it is in the same module and may be related.
3. Re-run Experiments 1, 2, 4, 5 at full seed count once ADMM is fixed.
4. Optionally enable exact ground truth in Experiment 3 (one line; costs ~8 ms
   per instance at 550 UEs).
5. Add confidence intervals and paired significance tests (Gap 6) — now
   straightforward, since an exact reference value exists per instance.
