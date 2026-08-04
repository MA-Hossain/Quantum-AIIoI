"""Large-scale experiment runner for SENTINEL/VANGUARD paper.

Designed for lab execution at 550+ UEs. Structured as independent
experiments that can be run individually or together.

Experiment 1 — Small-instance validation (3-10 UEs)
    Exact + QAOA + SA + ADMM. Validates QUBO correctness and QAOA quality.

Experiment 2 — Solver comparison (10-100 UEs)
    SA + ADMM + baselines. Shows SENTINEL vs classical methods.

Experiment 3 — Large-scale scalability (100-550 UEs)
    ADMM + baselines only. Demonstrates framework scales.

Experiment 4 — Disruption severity sweep (50 UEs)
    Vary disruption from mild to severe. Shows robustness benefit.

Experiment 5 — Multi-scenario robustness (50 UEs)
    Vary number of disruption scenarios S=1,3,5. Shows minimax value.

Experiment 6 — QAOA-in-ADMM (5-10 UEs)
    QAOA as ADMM sub-solver vs SA sub-solver. Demonstrates quantum pipeline.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from sagin_research_sim.aoii.renewal import precompute_psi
from sagin_research_sim.solver.admm import solve_admm
from sagin_research_sim.solver.qubo import (
    build_vanguard_qubo,
    compute_aoii_cost_matrix,
    extract_solution,
)
from sagin_research_sim.solver.solve_classical import solve_sa

# ---------------------------------------------------------------------------
# Default 4-state Markov source
# ---------------------------------------------------------------------------

P4 = np.array([
    [0.7, 0.2, 0.05, 0.05],
    [0.1, 0.6, 0.2, 0.1],
    [0.05, 0.15, 0.6, 0.2],
    [0.1, 0.1, 0.2, 0.6],
])


# ---------------------------------------------------------------------------
# Topology scaling: choose BS/SAT count by UE count
# ---------------------------------------------------------------------------

def auto_topology(num_ues: int) -> Dict:
    """Return (num_bs, num_sat) scaled to UE count."""
    if num_ues <= 15:
        return {"num_bs": 2, "num_sat": 2}
    elif num_ues <= 50:
        return {"num_bs": 3, "num_sat": 2}
    elif num_ues <= 150:
        return {"num_bs": 5, "num_sat": 3}
    elif num_ues <= 350:
        return {"num_bs": 8, "num_sat": 4}
    else:
        return {"num_bs": 10, "num_sat": 5}


# ---------------------------------------------------------------------------
# Rate-matrix generation with configurable disruption
# ---------------------------------------------------------------------------

def generate_rate_matrices(
    num_ues: int,
    num_nodes: int,
    num_bs: int,
    packet_size_bits: float,
    rng: np.random.Generator,
    num_scenarios: int = 1,
    severity: str = "moderate",
) -> tuple:
    """Generate nominal + disrupted rate matrices.

    severity: "mild" | "moderate" | "severe" | "extreme"
    num_scenarios: number of disruption scenarios (each with different RNG draw)

    Returns (rate_nominal, [rate_disrupted_1, ..., rate_disrupted_S])
    """
    severity_params = {
        "mild":     {"bs_deg": (1.5, 3.0), "sat_deg": (1.0, 1.1)},
        "moderate": {"bs_deg": (2.5, 5.0), "sat_deg": (1.0, 1.3)},
        "severe":   {"bs_deg": (4.0, 8.0), "sat_deg": (1.1, 1.5)},
        "extreme":  {"bs_deg": (6.0, 15.0), "sat_deg": (1.2, 2.0)},
    }
    params = severity_params.get(severity, severity_params["moderate"])

    # Nominal rates
    rate_matrix = np.zeros((num_ues, num_nodes))
    for i in range(num_ues):
        for m in range(num_bs):
            y_target = rng.uniform(1.0, 8.0)
            rate_matrix[i, m] = packet_size_bits / y_target
        for m in range(num_bs, num_nodes):
            y_target = rng.uniform(2.0, 10.0)
            rate_matrix[i, m] = packet_size_bits / y_target

    # Generate multiple disruption scenarios
    scenario_matrices = []
    for s in range(num_scenarios):
        rate_disrupted = rate_matrix.copy()
        for i in range(num_ues):
            for m in range(num_bs):
                nominal_y = packet_size_bits / rate_matrix[i, m]
                if nominal_y < 2.5:
                    deg = rng.uniform(*params["bs_deg"])
                elif nominal_y < 5.0:
                    deg = rng.uniform(
                        max(1.5, params["bs_deg"][0] * 0.6),
                        params["bs_deg"][1] * 0.7,
                    )
                else:
                    deg = rng.uniform(1.2, max(2.0, params["bs_deg"][0] * 0.4))
                rate_disrupted[i, m] = rate_matrix[i, m] / deg

            for m in range(num_bs, num_nodes):
                deg = rng.uniform(*params["sat_deg"])
                rate_disrupted[i, m] = rate_matrix[i, m] / deg

        scenario_matrices.append(rate_disrupted)

    return rate_matrix, scenario_matrices


# ---------------------------------------------------------------------------
# Baselines (imported from run.py logic, self-contained here)
# ---------------------------------------------------------------------------

def _random_assoc(num_ues, num_nodes, capacity, rng):
    assoc = {}
    load = np.zeros(num_nodes, dtype=int)
    order = list(range(num_ues))
    rng.shuffle(order)
    for i in order:
        cands = [m for m in range(num_nodes) if load[m] < capacity[m]]
        if not cands:
            cands = list(range(num_nodes))
        m = int(rng.choice(cands))
        assoc[i] = m
        load[m] += 1
    return assoc


def _greedy_rate_assoc(num_ues, num_nodes, rate_matrix, capacity):
    assoc = {}
    load = np.zeros(num_nodes, dtype=int)
    for i in range(num_ues):
        sorted_nodes = np.argsort(-rate_matrix[i])
        best_m = int(sorted_nodes[0])
        for m_int in sorted_nodes:
            m = int(m_int)
            if load[m] < capacity[m]:
                best_m = m
                break
        assoc[i] = best_m
        load[best_m] += 1
    return assoc


def _greedy_aoii_assoc(num_ues, num_nodes, aoii_cost, capacity):
    assoc = {}
    load = np.zeros(num_nodes, dtype=int)
    ue_order = sorted(range(num_ues), key=lambda i: -float(np.min(aoii_cost[i])))
    for i in ue_order:
        sorted_nodes = np.argsort(aoii_cost[i])
        best_m = int(sorted_nodes[0])
        for m_int in sorted_nodes:
            m = int(m_int)
            if load[m] < capacity[m]:
                best_m = m
                break
        assoc[i] = best_m
        load[best_m] += 1
    return assoc


def _local_search(association, aoii_cost, capacity, max_iters=200):
    num_ues, num_nodes = aoii_cost.shape
    assoc = dict(association)
    load = np.zeros(num_nodes, dtype=int)
    for m_val in assoc.values():
        load[m_val] += 1

    for _ in range(max_iters):
        per_ue = np.array([aoii_cost[i, assoc[i]] for i in range(num_ues)])
        worst_i = int(np.argmax(per_ue))
        worst_val = per_ue[worst_i]
        old_m = assoc[worst_i]

        best_m = old_m
        best_new_worst = worst_val
        for new_m in range(num_nodes):
            if new_m == old_m or load[new_m] >= capacity[new_m]:
                continue
            cand = per_ue.copy()
            cand[worst_i] = aoii_cost[worst_i, new_m]
            cand_worst = float(np.max(cand))
            if cand_worst < best_new_worst - 1e-10:
                best_new_worst = cand_worst
                best_m = new_m

        if best_m != old_m:
            assoc[worst_i] = best_m
            load[old_m] -= 1
            load[best_m] += 1
            continue

        swapped = False
        for j in range(num_ues):
            if j == worst_i or assoc[j] == old_m:
                continue
            partner_m = assoc[j]
            cand = per_ue.copy()
            cand[worst_i] = aoii_cost[worst_i, partner_m]
            cand[j] = aoii_cost[j, old_m]
            if float(np.max(cand)) < worst_val - 1e-10:
                assoc[worst_i] = partner_m
                assoc[j] = old_m
                swapped = True
                break
        if not swapped:
            break

    return assoc


# ---------------------------------------------------------------------------
# QUBO warm-start encoding
# ---------------------------------------------------------------------------

def _encode_assoc_for_qubo(association, aoii_cost, index_map,
                           num_levels, level_step, capacity):
    N = index_map.num_vars
    x = np.zeros(N)
    num_ues, num_nodes = aoii_cost.shape

    for i, m in association.items():
        x[index_map.assoc(i, m)] = 1.0

    aoii_cost_d = np.clip(
        np.round(aoii_cost / level_step).astype(int), 0, num_levels - 1,
    )
    worst_level = 0
    for i, m in association.items():
        worst_level = max(worst_level, int(aoii_cost_d[i, m]))
    worst_level = min(worst_level, num_levels - 1)

    x[index_map.epigraph(worst_level)] = 1.0

    aoii_slack_nbits = max(1, int(np.ceil(np.log2(max(num_levels, 2)))))
    for i, m in association.items():
        slack_val = max(0, worst_level - int(aoii_cost_d[i, m]))
        for b in range(aoii_slack_nbits):
            if (i, b) in index_map._aoii_slack and slack_val & (1 << b):
                x[index_map.aoii_slack(i, b)] = 1.0

    for m in range(num_nodes):
        load_count = sum(1 for ii, mm in association.items() if mm == m)
        slack_val = max(0, int(capacity[m]) - load_count)
        for (mk, b), flat in index_map._cap_slack.items():
            if mk == m and slack_val & (1 << b):
                x[flat] = 1.0

    return x


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics(association, rate_matrix, psi_table, packet_size_bits,
                     capacity, scenario_rate_matrices=None,
                     freshness_threshold=5.0):
    num_ues = rate_matrix.shape[0]
    num_nodes = rate_matrix.shape[1]
    Y_max = len(psi_table) - 1

    aoii_cost = compute_aoii_cost_matrix(
        num_ues, num_nodes, rate_matrix, psi_table,
        packet_size_bits, scenario_rate_matrices,
    )

    all_rates_mats = [rate_matrix]
    if scenario_rate_matrices:
        all_rates_mats.extend(scenario_rate_matrices)

    per_ue_aoii = []
    per_ue_Y = []
    per_ue_rate = []

    for i in range(num_ues):
        if i not in association:
            per_ue_aoii.append(float(psi_table[Y_max]))
            per_ue_Y.append(Y_max)
            per_ue_rate.append(0.0)
            continue
        m = association[i]
        per_ue_aoii.append(float(aoii_cost[i, m]))
        worst_Y = 0
        for rates in all_rates_mats:
            r = rates[i, m]
            y = min(int(np.ceil(packet_size_bits / r)), Y_max) if r > 0 else Y_max
            worst_Y = max(worst_Y, y)
        per_ue_Y.append(worst_Y)
        per_ue_rate.append(float(rate_matrix[i, m]))

    worst_aoii = max(per_ue_aoii)
    avg_aoii = float(np.mean(per_ue_aoii))
    sum_rate = sum(per_ue_rate)
    avg_latency = float(np.mean(per_ue_Y))
    delivery_ratio = sum(1 for a in per_ue_aoii if a < freshness_threshold) / num_ues
    fresh_ratio = sum(1 for y in per_ue_Y if y <= 2) / num_ues

    node_load = {}
    for i_ue, m_node in association.items():
        node_load[m_node] = node_load.get(m_node, 0) + 1
    capacity_ok = all(
        node_load.get(m, 0) <= int(capacity[m]) for m in range(num_nodes)
    )

    return {
        "worst_aoii": worst_aoii,
        "avg_aoii": avg_aoii,
        "sum_rate": sum_rate,
        "avg_latency": avg_latency,
        "delivery_ratio": delivery_ratio,
        "fresh_ratio": fresh_ratio,
        "capacity_ok": capacity_ok,
        "per_ue_aoii": per_ue_aoii,
    }


# ---------------------------------------------------------------------------
# QUBO build + calibration
# ---------------------------------------------------------------------------

def _build_instance(num_ues, num_nodes, psi_table, rate_matrix,
                    capacity, packet_size_bits, num_levels,
                    scenario_rate_matrices):
    aoii_cost = compute_aoii_cost_matrix(
        num_ues, num_nodes, rate_matrix, psi_table,
        packet_size_bits, scenario_rate_matrices,
    )
    max_aoii = float(np.max(aoii_cost))
    level_step = max_aoii / max(num_levels - 1, 1) if max_aoii > 0 else 1.0
    max_eta = (num_levels - 1) * level_step
    pen_base = max(10.0, 2.0 * max_eta)

    Q, idx_map = build_vanguard_qubo(
        num_ues=num_ues, num_nodes=num_nodes,
        psi_table=psi_table, rate_matrix=rate_matrix,
        capacity=capacity, packet_size_bits=packet_size_bits,
        num_levels=num_levels, level_step=level_step,
        penalty_single_assoc=pen_base,
        penalty_capacity=pen_base * 0.5,
        penalty_aoii=max(1.0, pen_base / max(num_levels - 1, 1)),
        penalty_epigraph=pen_base,
        scenario_rate_matrices=scenario_rate_matrices,
    )
    return Q, idx_map, aoii_cost, level_step


# ---------------------------------------------------------------------------
# Single-instance runner (all solvers)
# ---------------------------------------------------------------------------

def run_instance(
    num_ues: int,
    seed: int,
    num_scenarios: int = 1,
    severity: str = "moderate",
    packet_size_bits: float = 1000.0,
    num_levels: int = 10,
    num_bs: Optional[int] = None,
    num_sat: Optional[int] = None,
    capacity_per_node: Optional[int] = None,
    run_sa: bool = True,
    run_admm: bool = True,
    run_exact: bool = False,
    run_qaoa: bool = False,
    qaoa_depth: int = 1,
    qaoa_shots: int = 2048,
    sa_reads: int = 200,
    sa_sweeps: int = 1000,
    admm_max_iter: int = 50,
    admm_restarts: int = 5,
    admm_sub_solver: str = "auto",
    freshness_threshold: float = 5.0,
) -> Dict:
    """Run solvers on one problem instance.

    This is the main entry point for all experiments. Configure which
    solvers to run, disruption parameters, and topology.
    """
    if num_bs is None or num_sat is None:
        topo = auto_topology(num_ues)
        num_bs = num_bs or topo["num_bs"]
        num_sat = num_sat or topo["num_sat"]

    num_nodes = num_bs + num_sat
    rng = np.random.default_rng(seed)
    psi_table = precompute_psi(P4, Y_max=30)

    # Generate rate matrices
    rate_matrix, scenario_matrices = generate_rate_matrices(
        num_ues, num_nodes, num_bs, packet_size_bits, rng,
        num_scenarios=num_scenarios, severity=severity,
    )

    # Capacity: tight — total capacity ~ 1.15x UEs so solvers must trade off
    if capacity_per_node is not None:
        capacity = np.full(num_nodes, capacity_per_node)
    else:
        # BS gets slightly more capacity than SAT
        bs_cap = max(2, int(np.ceil(0.65 * num_ues / num_bs))) if num_bs > 0 else 2
        sat_cap = max(2, int(np.ceil(0.55 * num_ues / num_sat))) if num_sat > 0 else 2
        capacity = np.array(
            [bs_cap] * num_bs + [sat_cap] * num_sat
        )
        # Ensure total capacity >= num_ues (feasibility)
        while int(np.sum(capacity)) < num_ues:
            capacity[np.argmin(capacity)] += 1

    # Pre-compute AoII cost matrix
    aoii_cost = compute_aoii_cost_matrix(
        num_ues, num_nodes, rate_matrix, psi_table,
        packet_size_bits, scenario_matrices,
    )

    results = {
        "num_ues": num_ues,
        "num_nodes": num_nodes,
        "num_bs": num_bs,
        "num_sat": num_sat,
        "seed": seed,
        "num_scenarios": num_scenarios,
        "severity": severity,
        "capacity": capacity.tolist(),
        "solvers": {},
    }

    def _add(name, assoc, extra=None):
        metrics = _compute_metrics(
            assoc, rate_matrix, psi_table, packet_size_bits,
            capacity, scenario_matrices, freshness_threshold,
        )
        if extra:
            metrics.update(extra)
        metrics["association"] = {str(k): v for k, v in assoc.items()}
        results["solvers"][name] = metrics

    # --- Baselines ---
    rng_b = np.random.default_rng(seed)
    t0 = time.perf_counter()
    _add("random", _random_assoc(num_ues, num_nodes, capacity, rng_b),
         {"timing_s": time.perf_counter() - t0})

    t0 = time.perf_counter()
    _add("greedy_rate", _greedy_rate_assoc(num_ues, num_nodes, rate_matrix, capacity),
         {"timing_s": time.perf_counter() - t0})

    t0 = time.perf_counter()
    greedy_aoii = _greedy_aoii_assoc(num_ues, num_nodes, aoii_cost, capacity)
    _add("greedy_aoii", greedy_aoii,
         {"timing_s": time.perf_counter() - t0})

    t0 = time.perf_counter()
    greedy_ls = _local_search(dict(greedy_aoii), aoii_cost, capacity)
    _add("greedy_aoii_ls", greedy_ls,
         {"timing_s": time.perf_counter() - t0})

    # --- Build QUBO (needed for SA/ADMM/exact/QAOA) ---
    need_qubo = run_sa or run_admm or run_exact or run_qaoa
    Q = idx_map = level_step = None
    if need_qubo:
        Q, idx_map, _, level_step = _build_instance(
            num_ues, num_nodes, psi_table, rate_matrix,
            capacity, packet_size_bits, num_levels,
            scenario_matrices,
        )
        results["num_vars"] = idx_map.num_vars

        # Warm-start from greedy-AoII
        greedy_init = _encode_assoc_for_qubo(
            greedy_aoii, aoii_cost, idx_map, num_levels, level_step, capacity,
        )

    # --- SA on full QUBO ---
    if run_sa and Q is not None:
        t0 = time.perf_counter()
        res_sa = solve_sa(
            Q, index_map=idx_map, level_step=level_step, capacity=capacity,
            num_reads=sa_reads, num_sweeps=sa_sweeps,
            seed=seed, initial_states=greedy_init,
        )
        sa_time = time.perf_counter() - t0
        sa_assoc = res_sa.get("association", {})
        if sa_assoc:
            sa_assoc = _local_search(sa_assoc, aoii_cost, capacity)
        _add("sa", sa_assoc, {"timing_s": sa_time, "energy": res_sa["energy"]})

    # --- ADMM ---
    if run_admm and Q is not None:
        t0 = time.perf_counter()
        res_admm = solve_admm(
            Q, idx_map, level_step=level_step, capacity=capacity,
            max_iter=admm_max_iter, rho=2.0, seed=seed,
            num_restarts=admm_restarts, warm_start=greedy_init,
            sub_solver=admm_sub_solver,
            qaoa_depth=qaoa_depth, qaoa_shots=qaoa_shots,
        )
        admm_time = time.perf_counter() - t0
        admm_assoc = res_admm["association"]

        # Report ADMM raw (before LS) separately
        _add("admm_raw", admm_assoc, {
            "timing_s": admm_time,
            "energy": res_admm["energy"],
            "iterations": res_admm["iterations"],
            "converged": res_admm["converged"],
            "domain_sizes": res_admm["domain_sizes"],
        })

        # ADMM + LS post-processing
        admm_assoc_ls = _local_search(admm_assoc, aoii_cost, capacity)
        _add("admm", admm_assoc_ls, {
            "timing_s": admm_time,
            "energy": res_admm["energy"],
            "iterations": res_admm["iterations"],
            "converged": res_admm["converged"],
            "domain_sizes": res_admm["domain_sizes"],
        })

    # --- Exact (small only) ---
    if run_exact and Q is not None and idx_map.num_vars <= 25:
        from sagin_research_sim.solver.solve_exact import solve_exact
        t0 = time.perf_counter()
        res_exact = solve_exact(
            Q, index_map=idx_map, level_step=level_step, capacity=capacity,
        )
        _add("exact", res_exact["association"], {
            "timing_s": time.perf_counter() - t0,
            "energy": res_exact["energy"],
        })

    # --- QAOA (small only) ---
    if run_qaoa and Q is not None and idx_map.num_vars <= 20:
        from sagin_research_sim.solver.solve_qaoa import solve_qaoa
        t0 = time.perf_counter()
        res_qaoa = solve_qaoa(
            Q, index_map=idx_map, level_step=level_step, capacity=capacity,
            p=qaoa_depth, shots=qaoa_shots, maxiter=100, seed=seed,
        )
        _add("qaoa", res_qaoa["association"], {
            "timing_s": time.perf_counter() - t0,
            "energy": res_qaoa["energy"],
            "num_qubits": res_qaoa["num_qubits"],
            "opt_nfev": res_qaoa["opt_nfev"],
            "depth": res_qaoa["depth"],
        })

    return results


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def experiment_1_small_validation(output_dir: str, seeds: List[int] = None):
    """Small-instance validation: exact + QAOA + SA + ADMM."""
    if seeds is None:
        seeds = list(range(5))
    ue_counts = [3, 5, 8]
    all_results = []

    print("=" * 60)
    print("EXPERIMENT 1: Small-Instance Validation")
    print("=" * 60)

    for n_ue in ue_counts:
        for seed in seeds:
            print(f"  {n_ue} UEs, seed {seed} ...")
            res = run_instance(
                num_ues=n_ue, seed=seed,
                num_scenarios=2, severity="moderate",
                run_sa=True, run_admm=True,
                run_exact=True, run_qaoa=True,
                qaoa_depth=2, qaoa_shots=2048,
                sa_reads=100, sa_sweeps=500,
                admm_restarts=3,
            )
            all_results.append(res)

    _save_results(all_results, output_dir, "exp1_small_validation.json")
    return all_results


def experiment_2_solver_comparison(output_dir: str, seeds: List[int] = None):
    """Medium-scale solver comparison."""
    if seeds is None:
        seeds = list(range(5))
    ue_counts = [10, 25, 50, 75, 100]
    all_results = []

    print("=" * 60)
    print("EXPERIMENT 2: Solver Comparison (10-100 UEs)")
    print("=" * 60)

    for n_ue in ue_counts:
        run_sa_flag = n_ue <= 50  # SA on full QUBO only feasible up to ~50
        for seed in seeds:
            print(f"  {n_ue} UEs, seed {seed} (SA={run_sa_flag}) ...")
            res = run_instance(
                num_ues=n_ue, seed=seed,
                num_scenarios=3, severity="moderate",
                run_sa=run_sa_flag, run_admm=True,
                sa_reads=200, sa_sweeps=1000,
                admm_restarts=5,
            )
            all_results.append(res)

    _save_results(all_results, output_dir, "exp2_solver_comparison.json")
    return all_results


def experiment_3_large_scale(output_dir: str, seeds: List[int] = None):
    """Large-scale scalability: 100-550 UEs with ADMM + baselines."""
    if seeds is None:
        seeds = list(range(3))
    ue_counts = [100, 200, 350, 550]
    all_results = []

    print("=" * 60)
    print("EXPERIMENT 3: Large-Scale Scalability (100-550 UEs)")
    print("=" * 60)

    for n_ue in ue_counts:
        for seed in seeds:
            print(f"  {n_ue} UEs, seed {seed} ...")
            res = run_instance(
                num_ues=n_ue, seed=seed,
                num_scenarios=3, severity="moderate",
                run_sa=False, run_admm=True,
                admm_max_iter=40, admm_restarts=3,
            )
            all_results.append(res)

    _save_results(all_results, output_dir, "exp3_large_scale.json")
    return all_results


def experiment_4_disruption_severity(output_dir: str, seeds: List[int] = None):
    """Disruption severity sweep: shows robustness benefit."""
    if seeds is None:
        seeds = list(range(5))
    severities = ["mild", "moderate", "severe", "extreme"]
    num_ues = 50
    all_results = []

    print("=" * 60)
    print("EXPERIMENT 4: Disruption Severity Sweep (50 UEs)")
    print("=" * 60)

    for sev in severities:
        for seed in seeds:
            print(f"  severity={sev}, seed {seed} ...")
            res = run_instance(
                num_ues=num_ues, seed=seed,
                num_scenarios=3, severity=sev,
                run_sa=True, run_admm=True,
                sa_reads=200, sa_sweeps=1000,
                admm_restarts=5,
            )
            all_results.append(res)

    _save_results(all_results, output_dir, "exp4_disruption_severity.json")
    return all_results


def experiment_5_multi_scenario(output_dir: str, seeds: List[int] = None):
    """Multi-scenario robustness: vary S=1,3,5 disruption scenarios."""
    if seeds is None:
        seeds = list(range(5))
    scenario_counts = [1, 3, 5]
    num_ues = 50
    all_results = []

    print("=" * 60)
    print("EXPERIMENT 5: Multi-Scenario Robustness (50 UEs)")
    print("=" * 60)

    for n_sc in scenario_counts:
        for seed in seeds:
            print(f"  S={n_sc} scenarios, seed {seed} ...")
            res = run_instance(
                num_ues=num_ues, seed=seed,
                num_scenarios=n_sc, severity="severe",
                run_sa=True, run_admm=True,
                sa_reads=200, sa_sweeps=1000,
                admm_restarts=5,
            )
            all_results.append(res)

    _save_results(all_results, output_dir, "exp5_multi_scenario.json")
    return all_results


def experiment_6_qaoa_in_admm(output_dir: str, seeds: List[int] = None):
    """QAOA-in-ADMM: QAOA as sub-QUBO solver within ADMM pipeline."""
    if seeds is None:
        seeds = list(range(5))
    ue_counts = [5, 8, 10]
    all_results = []

    print("=" * 60)
    print("EXPERIMENT 6: QAOA-in-ADMM Pipeline")
    print("=" * 60)

    for n_ue in ue_counts:
        for seed in seeds:
            print(f"  {n_ue} UEs, seed {seed} ...")

            # ADMM with SA sub-solver (baseline)
            res_sa = run_instance(
                num_ues=n_ue, seed=seed,
                num_scenarios=2, severity="moderate",
                run_sa=False, run_admm=True,
                admm_sub_solver="auto",
                admm_restarts=3, admm_max_iter=30,
            )

            # ADMM with QAOA sub-solver
            res_qaoa = run_instance(
                num_ues=n_ue, seed=seed,
                num_scenarios=2, severity="moderate",
                run_sa=False, run_admm=True,
                admm_sub_solver="qaoa",
                qaoa_depth=2, qaoa_shots=2048,
                admm_restarts=2, admm_max_iter=20,
            )

            # Merge into one result
            combined = dict(res_sa)
            combined["solvers"]["admm_sa_sub"] = res_sa["solvers"].get("admm", {})
            combined["solvers"]["admm_qaoa_sub"] = res_qaoa["solvers"].get("admm", {})
            all_results.append(combined)

    _save_results(all_results, output_dir, "exp6_qaoa_in_admm.json")
    return all_results


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def _save_results(results: List[Dict], output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    # Make JSON-serializable (convert numpy types)
    def _clean(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"  -> Saved: {path}")


def load_results(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)
