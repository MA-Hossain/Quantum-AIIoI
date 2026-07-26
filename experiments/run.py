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
    """Greedy: each UE picks the best-rate node with remaining capacity.

    Ignores AoII structure and disruption robustness — just maximises
    instantaneous nominal rate.
    """
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


def generate_heterogeneous_rates(
    num_ues: int,
    num_nodes: int,
    packet_size_bits: float,
    psi_table: NDArray,
    rng: np.random.Generator,
    num_bs: int = 2,
):
    """Generate synthetic rate matrices with heterogeneous AoII structure.

    Design goals:
    - BS links span Y in [1, 8] — wide variation per UE-BS pair
    - SAT links span Y in [2, 10] — moderate, more uniform
    - Under disruption, BS links with low nominal Y degrade heavily
      (3-7x), creating "disruption traps" for greedy-rate solvers
    - SAT links are resilient to disruption (1-1.5x degradation)
    - The worst-case AoII (max over scenarios) creates a non-trivial
      assignment problem where some UEs are better off on SAT

    Returns (rate_matrix, rate_disrupted).
    """
    rate_matrix = np.zeros((num_ues, num_nodes))
    for i in range(num_ues):
        for m in range(num_bs):
            # Wide range: some excellent (Y=1-2), some mediocre (Y=5-8)
            y_target = rng.uniform(1.0, 8.0)
            rate_matrix[i, m] = packet_size_bits / y_target

        for m in range(num_bs, num_nodes):
            y_target = rng.uniform(2.0, 10.0)
            rate_matrix[i, m] = packet_size_bits / y_target

    # Disruption: BS links with best nominal rates are MOST vulnerable
    rate_disrupted = rate_matrix.copy()
    for i in range(num_ues):
        for m in range(num_bs):
            nominal_y = packet_size_bits / rate_matrix[i, m]
            if nominal_y < 2.5:
                degradation = rng.uniform(3.0, 7.0)
            elif nominal_y < 5.0:
                degradation = rng.uniform(2.0, 4.0)
            else:
                degradation = rng.uniform(1.2, 2.0)
            rate_disrupted[i, m] = rate_matrix[i, m] / degradation

        # SAT links are resilient
        for m in range(num_bs, num_nodes):
            degradation = rng.uniform(1.0, 1.5)
            rate_disrupted[i, m] = rate_matrix[i, m] / degradation

    return rate_matrix, rate_disrupted


def nearest_aoii_baseline(
    num_ues, num_nodes, aoii_cost, capacity,
):
    """Greedy AoII-aware: each UE picks the node with lowest worst-case
    AoII cost, respecting capacity. Bottleneck UEs (highest best-option
    AoII) are assigned first to ensure they get their preferred node."""
    assoc: Dict[int, int] = {}
    load = np.zeros(num_nodes, dtype=int)

    # Bottleneck-first: UEs whose best option is worst go first
    ue_order = sorted(
        range(num_ues), key=lambda i: -float(np.min(aoii_cost[i])),
    )

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


# -----------------------------------------------------------------------
# Local search refinement (minimax hill climbing)
# -----------------------------------------------------------------------

def local_search(
    association: Dict[int, int],
    aoii_cost: NDArray,
    capacity: NDArray,
    max_iters: int = 200,
) -> Dict[int, int]:
    """Swap-based hill climbing minimising worst-case AoII.

    Repeatedly finds the worst-off UE and tries to improve it by:
    1. Direct move to a node with lower AoII (if capacity available)
    2. Pairwise swap with another UE (capacity-neutral)
    """
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

        # Try direct move to a better node
        best_m = old_m
        best_new_worst = worst_val
        for new_m in range(num_nodes):
            if new_m == old_m:
                continue
            if load[new_m] >= capacity[new_m]:
                continue
            candidate_per_ue = per_ue.copy()
            candidate_per_ue[worst_i] = aoii_cost[worst_i, new_m]
            cand_worst = float(np.max(candidate_per_ue))
            if cand_worst < best_new_worst - 1e-10:
                best_new_worst = cand_worst
                best_m = new_m

        if best_m != old_m:
            assoc[worst_i] = best_m
            load[old_m] -= 1
            load[best_m] += 1
            continue

        # Try pairwise swap
        swapped = False
        for j in range(num_ues):
            if j == worst_i or assoc[j] == old_m:
                continue
            partner_m = assoc[j]
            candidate_per_ue = per_ue.copy()
            candidate_per_ue[worst_i] = aoii_cost[worst_i, partner_m]
            candidate_per_ue[j] = aoii_cost[j, old_m]
            cand_worst = float(np.max(candidate_per_ue))
            if cand_worst < worst_val - 1e-10:
                assoc[worst_i] = partner_m
                assoc[j] = old_m
                swapped = True
                break

        if not swapped:
            break

    return assoc


# -----------------------------------------------------------------------
# QUBO warm-start encoding
# -----------------------------------------------------------------------

def encode_association_for_qubo(
    association: Dict[int, int],
    aoii_cost: NDArray,
    index_map,
    num_levels: int,
    level_step: float,
    capacity: NDArray,
) -> NDArray:
    """Encode an association as a feasible QUBO binary vector.

    Sets association bits, computes the correct epigraph level,
    and fills in slack variables so all constraints are satisfied.
    """
    N = index_map.num_vars
    x = np.zeros(N)
    num_ues, num_nodes = aoii_cost.shape

    # Association bits
    for i, m in association.items():
        x[index_map.assoc(i, m)] = 1.0

    # Compute discretised AoII for assigned nodes
    aoii_cost_d = np.clip(
        np.round(aoii_cost / level_step).astype(int), 0, num_levels - 1,
    )
    worst_level = 0
    for i, m in association.items():
        worst_level = max(worst_level, int(aoii_cost_d[i, m]))
    worst_level = min(worst_level, num_levels - 1)

    # Epigraph one-hot
    x[index_map.epigraph(worst_level)] = 1.0

    # AoII slack: slack_i = worst_level - aoii_d[i, assigned_m]
    aoii_slack_nbits = max(1, int(np.ceil(np.log2(max(num_levels, 2)))))
    for i, m in association.items():
        slack_val = max(0, worst_level - int(aoii_cost_d[i, m]))
        for b in range(aoii_slack_nbits):
            if (i, b) in index_map._aoii_slack and slack_val & (1 << b):
                x[index_map.aoii_slack(i, b)] = 1.0

    # Capacity slack: slack_m = capacity[m] - load[m]
    load_count: Dict[int, int] = {}
    for i, m in association.items():
        load_count[m] = load_count.get(m, 0) + 1
    for m in range(num_nodes):
        slack_val = max(0, int(capacity[m]) - load_count.get(m, 0))
        for (mk, b), flat in index_map._cap_slack.items():
            if mk == m and slack_val & (1 << b):
                x[flat] = 1.0

    return x


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
# Constrained SA on association space
# -----------------------------------------------------------------------

def solve_minimax_sa(
    aoii_cost: NDArray,
    capacity: NDArray,
    initial_assoc: Optional[Dict[int, int]] = None,
    num_reads: int = 50,
    num_sweeps: int = 3000,
    seed: Optional[int] = None,
) -> Dict[int, int]:
    """Constrained SA minimising worst-case AoII directly.

    Operates on the association space {UE -> node}, not the QUBO.
    Moves: single-UE reassignment (50%) or pairwise swap (50%).
    All moves respect capacity constraints.
    """
    num_ues, num_nodes = aoii_cost.shape
    rng = np.random.default_rng(seed)
    cap = capacity.astype(int)

    best_assoc: Optional[Dict[int, int]] = None
    best_worst = float("inf")

    for read in range(num_reads):
        # Initialise
        if read == 0 and initial_assoc is not None:
            assoc = np.array(
                [initial_assoc[i] for i in range(num_ues)], dtype=int,
            )
        else:
            assoc = np.zeros(num_ues, dtype=int)
            load_init = np.zeros(num_nodes, dtype=int)
            for i in rng.permutation(num_ues):
                cands = np.where(load_init < cap)[0]
                if len(cands) == 0:
                    cands = np.arange(num_nodes)
                assoc[i] = int(rng.choice(cands))
                load_init[assoc[i]] += 1

        load = np.bincount(assoc, minlength=num_nodes).astype(int)
        per_ue = np.array([aoii_cost[i, assoc[i]] for i in range(num_ues)])
        current_worst = float(np.max(per_ue))

        beta_min, beta_max = 0.5, 10.0
        for step in range(num_sweeps):
            beta = beta_min + (beta_max - beta_min) * step / max(num_sweeps - 1, 1)

            if rng.random() < 0.5:
                # MOVE: reassign one UE
                i = int(rng.integers(0, num_ues))
                old_m = int(assoc[i])
                cands = [
                    m for m in range(num_nodes)
                    if m != old_m and load[m] < cap[m]
                ]
                if not cands:
                    continue
                new_m = int(rng.choice(cands))
                new_aoii_i = aoii_cost[i, new_m]

                if new_aoii_i >= current_worst:
                    new_worst = new_aoii_i
                elif per_ue[i] >= current_worst - 1e-10:
                    new_worst = float(max(
                        new_aoii_i,
                        max(per_ue[j] for j in range(num_ues) if j != i),
                    ))
                else:
                    new_worst = max(new_aoii_i, current_worst)

                delta = new_worst - current_worst
                if delta < 0 or rng.random() < np.exp(-beta * max(delta, 0)):
                    assoc[i] = new_m
                    load[old_m] -= 1
                    load[new_m] += 1
                    per_ue[i] = new_aoii_i
                    current_worst = new_worst
            else:
                # SWAP: exchange two UEs' nodes
                i = int(rng.integers(0, num_ues))
                j = int(rng.integers(0, num_ues))
                if i == j or assoc[i] == assoc[j]:
                    continue
                new_i = aoii_cost[i, assoc[j]]
                new_j = aoii_cost[j, assoc[i]]

                old_max_ij = max(per_ue[i], per_ue[j])
                new_max_ij = max(new_i, new_j)

                if old_max_ij >= current_worst - 1e-10:
                    others_max = max(
                        (per_ue[k] for k in range(num_ues) if k != i and k != j),
                        default=0.0,
                    )
                    new_worst = max(new_max_ij, others_max)
                else:
                    new_worst = max(new_max_ij, current_worst)

                delta = new_worst - current_worst
                if delta < 0 or rng.random() < np.exp(-beta * max(delta, 0)):
                    assoc[i], assoc[j] = assoc[j], assoc[i]
                    per_ue[i] = new_i
                    per_ue[j] = new_j
                    current_worst = new_worst

        if current_worst < best_worst:
            best_worst = current_worst
            best_assoc = {i: int(assoc[i]) for i in range(num_ues)}

    return best_assoc  # type: ignore[return-value]


# -----------------------------------------------------------------------
# Auto-calibrate QUBO discretisation
# -----------------------------------------------------------------------

def auto_level_step(
    aoii_cost: NDArray,
    num_levels: int,
) -> float:
    """Set level_step so the full AoII range is representable.

    level_step = max(aoii_cost) / (L - 1), ensuring the worst-case
    AoII maps to level L-1 instead of being clipped.
    """
    max_aoii = float(np.max(aoii_cost))
    if max_aoii <= 0 or num_levels <= 1:
        return 1.0
    return max_aoii / (num_levels - 1)


# -----------------------------------------------------------------------
# Single-instance runner
# -----------------------------------------------------------------------

def run_single(
    num_ues: int,
    seed: int,
    P: Optional[NDArray] = None,
    packet_size_bits: float = 1000.0,
    num_levels: int = 10,
    topo_cfg: Optional[TopologyConfig] = None,
    channel_cfg: Optional[ChannelModelConfig] = None,
    capacity_per_node: Optional[int] = None,
    freshness_threshold: float = 5.0,
    sa_reads: int = 200,
    sa_sweeps: int = 1000,
    admm_max_iter: int = 50,
    admm_rho: float = 2.0,
    admm_restarts: int = 3,
    use_synthetic: bool = True,
    run_exact: bool = False,
    run_qaoa: bool = False,
    qaoa_depth: int = 1,
    qaoa_shots: int = 2048,
) -> Dict:
    """Run all solvers on one problem instance.

    The QUBO level_step is auto-calibrated from the actual AoII cost
    matrix so the discretisation covers the full range.

    When use_synthetic=True (default), generates heterogeneous rate
    matrices where the optimal assignment is non-trivial.
    """
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

    num_nodes = topo_cfg.num_bs + topo_cfg.num_satellites + topo_cfg.num_uavs
    rng_rates = np.random.default_rng(seed)

    if use_synthetic:
        psi_table = precompute_psi(P, Y_max=20)
        rate_matrix, rate_disrupted = generate_heterogeneous_rates(
            num_ues, num_nodes, packet_size_bits, psi_table, rng_rates,
            num_bs=topo_cfg.num_bs,
        )
        scenario_rate_matrices = [rate_disrupted]
    else:
        sim_cfg = SimulationConfig(seed=seed, num_ues=num_ues)
        topology = generate_topology(sim_cfg, topo_cfg)
        num_nodes = len(topology.serving_nodes)
        rate_matrix = build_rate_matrix(topology, channel_cfg, topo_cfg, seed=seed)
        disruption = DisruptionScenario(
            name="sat_degraded", total_extra_loss_db=10.0, outage_probability=0.0,
        )
        rate_disrupted = build_rate_matrix(
            topology, channel_cfg, topo_cfg, seed=seed, scenario=disruption,
        )
        scenario_rate_matrices = [rate_disrupted]
        psi_table = precompute_psi(P, Y_max=20)

    # Capacity: deliberately scarce so some UEs must use SAT
    if capacity_per_node is not None:
        capacity = np.full(num_nodes, capacity_per_node)
    else:
        # BS nodes get ceil(I/M)+1, SAT nodes get ceil(I/M)
        base_cap = max(2, (num_ues + num_nodes - 1) // num_nodes)
        capacity = np.array([
            base_cap + 1 if m < topo_cfg.num_bs else base_cap
            for m in range(num_nodes)
        ])

    # Pre-compute AoII cost matrix for auto-calibration and baselines
    aoii_cost = compute_aoii_cost_matrix(
        num_ues, num_nodes, rate_matrix, psi_table,
        packet_size_bits, scenario_rate_matrices,
    )

    # Auto-calibrate level_step so discretisation covers the full AoII range
    level_step = auto_level_step(aoii_cost, num_levels)

    # Scale penalties relative to the objective range (max_eta).
    # C1 and C4 have unit coefficients — standard scaling.
    # C3 has coefficients up to (L-1), so its _add_equality_penalty expands
    # with terms ~ pen_aoii * (L-1)^2.  Scale pen_aoii down by (L-1) so
    # the effective penalty per variable stays ~ pen_base.
    max_eta = (num_levels - 1) * level_step
    pen_base = max(10.0, 2.0 * max_eta)
    pen_assoc = pen_base
    pen_epi = pen_base
    pen_cap = pen_base * 0.5
    pen_aoii = max(1.0, pen_base / max(num_levels - 1, 1))

    # Build QUBO with calibrated discretisation and scaled penalties
    Q, idx_map = build_vanguard_qubo(
        num_ues=num_ues, num_nodes=num_nodes,
        psi_table=psi_table, rate_matrix=rate_matrix,
        capacity=capacity, packet_size_bits=packet_size_bits,
        num_levels=num_levels, level_step=level_step,
        penalty_single_assoc=pen_assoc, penalty_capacity=pen_cap,
        penalty_aoii=pen_aoii, penalty_epigraph=pen_epi,
        scenario_rate_matrices=scenario_rate_matrices,
    )

    results: Dict = {
        "num_ues": num_ues,
        "num_nodes": num_nodes,
        "seed": seed,
        "num_vars": idx_map.num_vars,
        "capacity": capacity.tolist(),
        "level_step": level_step,
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

    # --- Greedy rate baseline ---
    t0 = time.perf_counter()
    greedy_assoc = greedy_rate_baseline(num_ues, num_nodes, rate_matrix, capacity)
    _add_solver("greedy", greedy_assoc, {"timing_s": time.perf_counter() - t0})

    # --- Greedy AoII baseline ---
    t0 = time.perf_counter()
    aoii_greedy_assoc = nearest_aoii_baseline(num_ues, num_nodes, aoii_cost, capacity)
    _add_solver("greedy_aoii", aoii_greedy_assoc, {"timing_s": time.perf_counter() - t0})

    # --- Constrained SA (directly on association space) ---
    t0 = time.perf_counter()
    sa_assoc = solve_minimax_sa(
        aoii_cost, capacity, initial_assoc=aoii_greedy_assoc,
        num_reads=sa_reads, num_sweeps=sa_sweeps * 3, seed=seed,
    )
    sa_assoc = local_search(sa_assoc, aoii_cost, capacity)
    sa_time = time.perf_counter() - t0
    _add_solver("sa", sa_assoc, {"timing_s": sa_time})

    # --- ADMM (warm-started with greedy-AoII, + local search) ---
    greedy_init = encode_association_for_qubo(
        aoii_greedy_assoc, aoii_cost, idx_map, num_levels, level_step, capacity,
    )
    res_admm = solve_admm(
        Q, idx_map, level_step=level_step, capacity=capacity,
        max_iter=admm_max_iter, rho=admm_rho, seed=seed,
        num_restarts=admm_restarts,
        warm_start=greedy_init,
    )
    admm_assoc = res_admm["association"]
    admm_assoc_ls = local_search(admm_assoc, aoii_cost, capacity)
    _add_solver("admm", admm_assoc_ls, {
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
