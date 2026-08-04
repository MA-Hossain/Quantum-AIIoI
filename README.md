# SENTINEL / VANGUARD — SAGIN AoII Optimization

Simulation framework for the SENTINEL paper: robust minimax Age of Incorrect Information (AoII) optimization in Space-Air-Ground Integrated Networks (SAGIN) via QUBO formulation, ADMM decomposition, and QAOA-compatible quantum pipeline.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run a quick smoke test (~5 min)
python3 run_all_experiments.py --tier quick

# 3. Check results
ls output/large_scale/
ls output/large_scale/figures/
```

## Project Structure

```
sagin_research_sim/           # Core simulation package
  aoii/renewal.py             # AoII Markov renewal theory (Psi computation)
  solver/qubo.py              # VANGUARD QUBO builder (minimax AoII formulation)
  solver/admm.py              # SENTINEL ADMM decomposition (vectorized)
  solver/solve_classical.py   # SA solver (D-Wave neal + numpy fallback)
  solver/solve_qaoa.py        # QAOA solver (Qiskit Aer)
  solver/solve_exact.py       # Brute-force exact solver (small instances)
  channels/                   # Physical layer (path loss, fading, rates)
  nodes/                      # Network nodes (BS, SAT, UAV, UE)
  config/                     # Configuration classes
  radio/                      # Radio parameter profiles

experiments/                  # Experiment runners and plotting
  run.py                      # Original experiment runner (Steps 1-7)
  run_large_scale.py          # Large-scale runner (Experiments 1-6, 550+ UEs)
  plots.py                    # Original figure generation
  plots_extended.py           # Extended figures for all 6 experiments

tests/                        # Validation and diagnostic scripts
output/                       # Results (JSON) and figures (PNG)

run_all_experiments.py        # CLI entry point
requirements.txt              # Python dependencies
```

## Installation

### Requirements

- Python 3.10+ (tested with 3.12)
- ~4 GB RAM for experiments up to 100 UEs
- ~16 GB RAM for 550 UE experiments
- Multi-core CPU recommended for large-scale runs

### Install

```bash
# Option A: System-wide (if using Homebrew Python)
pip install -r requirements.txt

# Option B: Virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option C: Conda
conda create -n sentinel python=3.12
conda activate sentinel
pip install -r requirements.txt
```

### Verify Installation

```bash
python3 -c "
import numpy, scipy, matplotlib, neal, dimod
print('Core deps OK')
try:
    import qiskit, qiskit_aer
    print('Qiskit OK (QAOA experiments available)')
except ImportError:
    print('Qiskit not installed (QAOA experiments will be skipped)')
"
```

## Running Experiments

### Tiers

| Tier | Command | Time | UEs | Experiments |
|------|---------|------|-----|-------------|
| Quick | `--tier quick` | ~5 min | 3-8 | 1 only (validation) |
| Medium | `--tier medium` | ~1-2 hr | 3-100 | 1,2,4,5,6 |
| Full | `--tier full` | ~4-8 hr | 3-550 | All (1-6) |

### Run Commands

```bash
# Quick smoke test
python3 run_all_experiments.py --tier quick

# Medium tier (recommended first run)
python3 run_all_experiments.py --tier medium

# Full scale (550 UEs — needs ~16 GB RAM, several hours)
python3 run_all_experiments.py --tier full

# Run specific experiments
python3 run_all_experiments.py --experiment 1 2
python3 run_all_experiments.py --experiment 3    # large-scale only
python3 run_all_experiments.py --experiment 4 5  # disruption analysis

# Custom seeds
python3 run_all_experiments.py --tier full --seeds 0 1 2 3 4

# Regenerate figures from saved results
python3 run_all_experiments.py --plots-only
```

## Experiment Descriptions

### Experiment 1: Small-Instance Validation (3-8 UEs)

Validates correctness by comparing all solvers on instances small enough for exact enumeration.

- Solvers: Exact, QAOA (p=2), SA, ADMM, Greedy baselines
- Validates: QUBO matches original problem, QAOA approximation ratio
- Output: `exp1_small_validation.json`, `exp1_worst_aoii.png`, `exp1_qaoa_ratio.png`

### Experiment 2: Solver Comparison (10-100 UEs)

Compares solver quality and runtime at moderate scale.

- Solvers: SA (up to 50 UEs), ADMM, Greedy, Greedy+LS, Random
- 3 disruption scenarios, moderate severity
- Output: `exp2_solver_comparison.json`, `exp2_*.png`

### Experiment 3: Large-Scale Scalability (100-550 UEs)

Demonstrates that the distributed ADMM framework scales to large networks.

- Solvers: ADMM + baselines (SA too slow at this scale)
- Topology auto-scales: 5-10 BS, 3-5 SAT
- Output: `exp3_large_scale.json`, `exp3_*.png`

### Experiment 4: Disruption Severity Sweep (50 UEs)

Shows that SENTINEL's advantage grows as disruption severity increases.

- Severity levels: mild, moderate, severe, extreme
- 3 disruption scenarios per level
- Output: `exp4_disruption_severity.json`, `exp4_*.png`

### Experiment 5: Multi-Scenario Robustness (50 UEs)

Shows that minimax over more scenarios improves robustness.

- Scenario counts: S=1, 3, 5
- Severe disruption
- Output: `exp5_multi_scenario.json`, `exp5_*.png`

### Experiment 6: QAOA-in-ADMM Pipeline (5-10 UEs)

Demonstrates QAOA solving actual sub-QUBOs within the ADMM pipeline.

- Compares: ADMM with SA sub-solver vs ADMM with QAOA sub-solver
- Reports: quality gap, runtime, qubit count
- Output: `exp6_qaoa_in_admm.json`, `exp6_*.png`

## Output Structure

```
output/large_scale/
  exp1_small_validation.json    # Raw results (all metrics, per-instance)
  exp2_solver_comparison.json
  exp3_large_scale.json
  exp4_disruption_severity.json
  exp5_multi_scenario.json
  exp6_qaoa_in_admm.json
  figures/
    exp1_worst_aoii.png         # Solver comparison (small)
    exp1_qaoa_ratio.png         # QAOA vs exact
    exp2_worst_aoii.png         # Solver comparison (medium)
    exp2_avg_aoii.png
    exp2_timing.png
    exp2_delivery.png
    exp3_worst_aoii.png         # Scalability
    exp3_timing.png
    exp3_domain_sizes.png       # ADMM decomposition analysis
    exp4_worst_aoii.png         # Disruption severity
    exp4_improvement.png        # SENTINEL advantage by severity
    exp5_scenarios.png          # Multi-scenario robustness
    exp6_qaoa_admm.png          # QAOA-in-ADMM quality
    exp6_timing.png             # QAOA-in-ADMM runtime
```

## Running Previous Experiments (Steps 1-7)

The original step-by-step validation is still available:

```bash
# Individual step tests
python3 tests/step2_aoii_test.py      # AoII renewal validation
python3 tests/step3_qubo_test.py      # QUBO correctness
python3 tests/step4_classical_test.py # SA solver
python3 tests/step5_qaoa_test.py      # QAOA depth sweep
python3 tests/step6_admm_test.py      # ADMM decomposition
python3 tests/step7_experiment_test.py # Full sweep (5-20 UEs)
```

## Troubleshooting

### Out of memory at 550 UEs

Reduce the number of ADMM restarts or use fewer seeds:

```bash
python3 run_all_experiments.py --experiment 3 --seeds 0 1
```

### Qiskit not installed

Experiments 1 and 6 (QAOA) will skip QAOA tests gracefully. Install with:

```bash
pip install qiskit qiskit-aer
```

### D-Wave neal not installed

SA solver falls back to pure-NumPy implementation (slower but functional):

```bash
pip install dwave-neal dimod
```

### Slow ADMM at large scale

The vectorized ADMM solver is optimized for large instances but may still take
10-30 minutes per instance at 550 UEs. This is expected — the experiment is
designed to run on a lab workstation, not a laptop.

## Architecture Notes

- **QUBO auto-calibration**: `level_step = max_aoii / (L-1)`, penalty scaling prevents
  ill-conditioned penalty landscape
- **Constrained SA**: warm-started from greedy-AoII solution
- **ADMM vectorized**: sub-QUBO construction uses numpy matrix operations instead
  of Python loops (required for 550+ UE feasibility)
- **QAOA-in-ADMM**: sub-solver parameter allows QAOA to solve domain sub-QUBOs
  within the ADMM pipeline (only for small decomposed problems, <=20 qubits)
- **Local search**: minimax hill climbing (move + swap) applied as post-processing
  to all optimization solvers for fair comparison
