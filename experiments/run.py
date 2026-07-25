"""Experiment runner for VANGUARD SAGIN optimisation.

Sweeps UE counts x seeds, runs all solvers (SA, ADMM, baselines),
logs metrics: worst-case AoII, avg AoII, delivery ratio, fresh-update
ratio, latency, sum rate, computation time.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from sagin_research_sim.aoii.renewal import precompute_psi
from sagin_research_sim.channels.rate import compute_all_rates
from sagin_research_sim.config.radio_config import (
    BandwidthMode,
    ChannelModelConfig,
    DisruptionScenario,
)
from sagin_research_sim.config.simulation_config import SimulationConfig
from sagin_research_sim.config.topology_config import TopologyConfig
from sagin_research_sim.nodes.topology import generate_topology
from sagin_research_sim.solver.admm import solve_admm
from sagin_research_sim.solver.qubo import (
    build_vanguard_qubo,
    compute_aoii_cost_matrix,
)
from sagin_research_sim.solver.solve_classical import solve_sa

# Default 4-state Markov source
P4 = np.array([
    [0.7, 0.2, 0.05, 0.05],
    [0.1, 0.6, 0.2, 0.1],
    [0.05, 0.15, 0.6, 0.2],
    [0.1, 0.1, 0.2, 0.6],
])


# -----------------------------------------------------------------------
# Rate-matrix extraction
# -----------------------------------------------------------------------

def build_rate_matrix(topology, channel_cfg, topo_cfg, seed=0, scenario=None):
    """Extract an (I x M) rate matrix from the full simulator."""
    results = compute_all_rates(
        topology, channel_cfg, topo_cfg, seed=seed, scenario=scenario,
    )
    num_ues = len(topology.ues)
    num_nodes = len(topology.serving_nodes)

    node_idx = {node.uuid: k for k, node in enumerate(topology.serving_nodes)}
    ue_idx = {ue.uuid: i for i, ue in enumerate(topology.ues)}

    rate = np.zeros((num_ues, num_nodes))
    for r in results:
        i = ue_idx[r.ue_uuid]
        m = node_idx[r.node_uuid]
        rate[i, m] = max(r.rate_bps, 0.0)
    return rate


# -----------------------------------------------------------------------
# Baseline associations
# -----------------------------------------------------------------------

def random_baseline(num_ues, num_nodes, capacity, rng):
    """Random association respecting capacity constraints."""
    assoc: Dict[int, int] = {}
    load = np.zeros(num_nodes, dtype=int)
    order = list(range(num_ues))
    rng.shuffle(order)

    for i in order:
        candidates = [m for m in range(num_nodes) if load[m] < capacity[m]]
        if not candidates:
            candidates = list(range(num_nodes))
        m = int(rng.choice(candidates))
        assoc[i] = m
        load[m] += 1
    return assoc


def greedy_rate_baseline(num_ues, num_nodes, rate_matrix, capacity):
    """Greedy: each UE picks the best-rate node with remaining capacity."""
    assoc: Dict[int, int] = {}
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


# -----------------------------------------------------------------------
# Metric computation
# -----------------------------------------------------------------------

def compute_metrics(
    association: Dict[int, int],
    rate_matrix: NDArray,
    psi_table: NDArray,
    packet_size_bits: float,
    capacity: NDArray,
    scenario_rate_matrices: Optional[List[NDArray]] = None,
    freshness_threshold: float = 5.0,
) -> Dict:
    """Compute performance metrics for a given association."""
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

    per_ue_aoii: List[float] = []
    per_ue_Y: List[int] = []
    per_ue_rate: List[float] = []

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
            if r <= 0:
                y = Y_max
            else:
                y = min(int(np.ceil(packet_size_bits / r)), Y_max)
            worst_Y = max(worst_Y, y)
        per_ue_Y.append(worst_Y)
        per_ue_rate.append(float(rate_matrix[i, m]))

    worst_aoii = max(per_ue_aoii)
    avg_aoii = float(np.mean(per_ue_aoii))
    sum_rate = sum(per_ue_rate)
    avg_latency = float(np.mean(per_ue_Y))
    delivery_ratio = sum(1 for a in per_ue_aoii if a < freshness_threshold) / num_ues
    fresh_ratio = sum(1 for y in per_ue_Y if y <= 2) / num_ues

    node_load: Dict[int, int] = {}
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
        "per_ue_Y": per_ue_Y,
        "per_ue_rate": per_ue_rate,
    }


# -----------------------------------------------------------------------
# Single-instance runner
# -----------------------------------------------------------------------

def run_single(
    num_ues: int,
    seed: int,
    P: Optional[NDArray] = None,
    packet_size_bits: float = 1000.0,
    num_levels: int = 8,
    level_step: float = 0.3,
    topo_cfg: Optional[TopologyConfig] = None,
    channel_cfg: Optional[ChannelModelConfig] = None,
    freshness_threshold: float = 5.0,
    sa_reads: int = 200,
    sa_sweeps: int = 1000,
    admm_max_iter: int = 40,
    admm_rho: float = 2.0,
    run_exact: bool = False,
    run_qaoa: bool = False,
    qaoa_depth: int = 1,
    qaoa_shots: int = 2048,
) -> Dict:
    """Run all solvers on one problem instance."""
    if P is None:
        P = P4
    if topo_cfg is None:
        topo_cfg = TopologyConfig(num_bs=2, num_satellites=2)
    if channel_cfg is None:
        channel_cfg = ChannelModelConfig(
            bandwidth_mode=BandwidthMode.CONTINUOUS,
            include_fading=False,
            include_interference=False,
        )

    sim_cfg = SimulationConfig(seed=seed, num_ues=num_ues)
    topology = generate_topology(sim_cfg, topo_cfg)
    num_nodes = len(topology.serving_nodes)

    # Nominal rate matrix
    rate_matrix = build_rate_matrix(topology, channel_cfg, topo_cfg, seed=seed)

    # Disrupted scenario: satellites get 10 dB extra loss
    disruption = DisruptionScenario(
        name="sat_degraded", total_extra_loss_db=10.0, outage_probability=0.0,
    )
    rate_disrupted = build_rate_matrix(
        topology, channel_cfg, topo_cfg, seed=seed, scenario=disruption,
    )
    scenario_rate_matrices = [rate_disrupted]

    # Capacity: roughly ceil(I/M) + 1 per node, minimum 2
    cap_per_node = max(2, (num_ues + num_nodes - 1) // num_nodes + 1)
    capacity = np.full(num_nodes, cap_per_node)

    psi_table = precompute_psi(P, Y_max=20)

    # Build QUBO
    Q, idx_map = build_vanguard_qubo(
        num_ues=num_ues, num_nodes=num_nodes,
        psi_table=psi_table, rate_matrix=rate_matrix,
        capacity=capacity, packet_size_bits=packet_size_bits,
        num_levels=num_levels, level_step=level_step,
        penalty_single_assoc=10.0, penalty_capacity=8.0,
        penalty_aoii=6.0, penalty_epigraph=10.0,
        scenario_rate_matrices=scenario_rate_matrices,
    )

    results: Dict = {
        "num_ues": num_ues,
        "num_nodes": num_nodes,
        "seed": seed,
        "num_vars": idx_map.num_vars,
        "capacity": capacity.tolist(),
        "solvers": {},
    }

    def _add_solver(name, association, extra=None):
        metrics = compute_metrics(
            association, rate_matrix, psi_table, packet_size_bits,
            capacity, scenario_rate_matrices, freshness_threshold,
        )
        if extra:
            metrics.update(extra)
        metrics["association"] = association
        results["solvers"][name] = metrics

    # --- Random baseline ---
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    rand_assoc = random_baseline(num_ues, num_nodes, capacity, rng)
    _add_solver("random", rand_assoc, {"timing_s": time.perf_counter() - t0})

    # --- Greedy baseline ---
    t0 = time.perf_counter()
    greedy_assoc = greedy_rate_baseline(num_ues, num_nodes, rate_matrix, capacity)
    _add_solver("greedy", greedy_assoc, {"timing_s": time.perf_counter() - t0})

    # --- SA ---
    res_sa = solve_sa(
        Q, index_map=idx_map, level_step=level_step, capacity=capacity,
        num_reads=sa_reads, num_sweeps=sa_sweeps, seed=seed,
    )
    _add_solver("sa", res_sa["association"], {
        "timing_s": res_sa["timing_s"], "energy": res_sa["energy"],
    })

    # --- ADMM ---
    res_admm = solve_admm(
        Q, idx_map, level_step=level_step, capacity=capacity,
        max_iter=admm_max_iter, rho=admm_rho, seed=seed,
    )
    _add_solver("admm", res_admm["association"], {
        "timing_s": res_admm["timing_s"],
        "energy": res_admm["energy"],
        "iterations": res_admm["iterations"],
        "converged": res_admm["converged"],
        "primal_residuals": res_admm["primal_residuals"],
    })

    # --- Exact (small instances only) ---
    if run_exact and idx_map.num_vars <= 25:
        from sagin_research_sim.solver.solve_exact import solve_exact
        res_exact = solve_exact(
            Q, index_map=idx_map, level_step=level_step, capacity=capacity,
        )
        _add_solver("exact", res_exact["association"], {
            "timing_s": res_exact["timing_s"], "energy": res_exact["energy"],
        })

    # --- QAOA (small instances only) ---
    if run_qaoa and idx_map.num_vars <= 20:
        from sagin_research_sim.solver.solve_qaoa import solve_qaoa
        res_qaoa = solve_qaoa(
            Q, index_map=idx_map, level_step=level_step, capacity=capacity,
            p=qaoa_depth, shots=qaoa_shots, maxiter=100, seed=seed,
        )
        _add_solver("qaoa", res_qaoa["association"], {
            "timing_s": res_qaoa["timing_s"], "energy": res_qaoa["energy"],
        })

    return results


# -----------------------------------------------------------------------
# Full sweep
# -----------------------------------------------------------------------

def run_sweep(
    ue_counts: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    **kwargs,
) -> List[Dict]:
    """Sweep UE counts x seeds, return list of result dicts."""
    if ue_counts is None:
        ue_counts = [5, 8, 10, 15, 20]
    if seeds is None:
        seeds = list(range(5))

    all_results: List[Dict] = []
    for n_ue in ue_counts:
        for seed in seeds:
            print(f"  Running: {n_ue} UEs, seed {seed} ...")
            res = run_single(n_ue, seed, **kwargs)
            all_results.append(res)
    return all_results
