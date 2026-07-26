"""ADMM decomposition for large VANGUARD QUBOs.

Partitions the full QUBO into K per-node domains plus one global domain.
Each domain solves a smaller sub-QUBO independently, and ADMM consensus
updates enforce cross-domain constraints (single-association, epigraph).

Domains
-------
    Domain k  (k = 0 .. M-1):  x_{i,k} for all UEs i  +  cap_slack_{k,*}
    Global domain:              eta_l  +  aoii_slack_{i,*}

ADMM iterations
---------------
    1.  x-update   — solve each domain's augmented sub-QUBO
    2.  z-update   — consensus projection (one-hot association, one-hot eta)
    3.  u-update   — dual variable:  u ← u + x − z
    4.  Convergence check:  ‖x − z‖ < ε
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from sagin_research_sim.solver.qubo import IndexMap, evaluate_qubo, extract_solution


# ---------------------------------------------------------------------------
# Domain partitioning
# ---------------------------------------------------------------------------

@dataclass
class DomainPartition:
    """Holds per-node domains and the global domain."""
    node_domains: List[List[int]]    # node_domains[k] = flat indices
    global_domain: List[int]         # flat indices of eta + aoii_slack
    num_ues: int
    num_nodes: int
    num_levels: int


def partition_by_node(index_map: IndexMap) -> DomainPartition:
    """Split QUBO variables into per-node and global domains."""
    ue_set = {i for (i, _) in index_map._assoc}
    node_set = {m for (_, m) in index_map._assoc}
    num_ues = max(ue_set) + 1 if ue_set else 0
    num_nodes = max(node_set) + 1 if node_set else 0
    num_levels = len(index_map._epigraph)

    node_domains: List[List[int]] = []
    for k in range(num_nodes):
        indices: List[int] = []
        for i in range(num_ues):
            if (i, k) in index_map._assoc:
                indices.append(index_map.assoc(i, k))
        for (m, b), flat in index_map._cap_slack.items():
            if m == k:
                indices.append(flat)
        node_domains.append(sorted(indices))

    global_domain = sorted(
        list(index_map._epigraph.values())
        + list(index_map._aoii_slack.values())
    )

    return DomainPartition(
        node_domains=node_domains,
        global_domain=global_domain,
        num_ues=num_ues,
        num_nodes=num_nodes,
        num_levels=num_levels,
    )


# ---------------------------------------------------------------------------
# Sub-QUBO extraction
# ---------------------------------------------------------------------------

def _build_sub_qubo(
    Q: NDArray,
    domain_indices: List[int],
    x_full: NDArray,
    z: NDArray,
    u: NDArray,
    rho: float,
) -> NDArray:
    """Build augmented sub-QUBO for one domain.

    1. Extract Q submatrix for the domain's variables.
    2. Add coupling from fixed variables (other domains) as linear terms.
    3. Add ADMM penalty  (rho/2)(1 − 2(z_j − u_j))  on diagonal.
    """
    domain_set = set(domain_indices)
    n_d = len(domain_indices)
    N = Q.shape[0]

    Q_sub = np.zeros((n_d, n_d))

    for s1, g1 in enumerate(domain_indices):
        # Intra-domain quadratic terms (upper-triangular)
        for s2 in range(s1, n_d):
            g2 = domain_indices[s2]
            r, c = (g1, g2) if g1 <= g2 else (g2, g1)
            Q_sub[s1, s2] = Q[r, c]

        # Coupling with fixed (out-of-domain) variables → linear on x_{g1}
        coupling = 0.0
        for g2 in range(N):
            if g2 in domain_set:
                continue
            r, c = (g1, g2) if g1 < g2 else (g2, g1)
            coupling += Q[r, c] * x_full[g2]
        Q_sub[s1, s1] += coupling

        # ADMM augmented Lagrangian
        # (rho/2)||x_j - z_j + u_j||^2 for binary x_j
        #   = (rho/2)[(1 - 2(z_j - u_j)) x_j + (z_j - u_j)^2]
        Q_sub[s1, s1] += (rho / 2.0) * (1.0 - 2.0 * (z[g1] - u[g1]))

    return Q_sub


# ---------------------------------------------------------------------------
# Sub-QUBO solvers (auto-select by size)
# ---------------------------------------------------------------------------

def _solve_sub_exact(Q_sub: NDArray) -> NDArray:
    """Brute-force solve a small sub-QUBO (N <= 20)."""
    n = Q_sub.shape[0]
    best_e = float("inf")
    best_x = np.zeros(n)
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        e = float(x @ Q_sub @ x)
        if e < best_e:
            best_e = e
            best_x = x.copy()
    return best_x


def _solve_sub_sa(Q_sub: NDArray, num_reads: int, num_sweeps: int,
                  seed: Optional[int]) -> NDArray:
    """Solve a sub-QUBO with SA (uses neal if available, else numpy)."""
    from sagin_research_sim.solver.solve_classical import solve_sa
    res = solve_sa(Q_sub, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    return res["x"]


def _solve_sub(Q_sub: NDArray, max_exact: int = 18,
               sa_reads: int = 50, sa_sweeps: int = 500,
               seed: Optional[int] = None) -> NDArray:
    """Auto-select solver based on sub-QUBO size."""
    n = Q_sub.shape[0]
    if n <= max_exact:
        return _solve_sub_exact(Q_sub)
    return _solve_sub_sa(Q_sub, sa_reads, sa_sweeps, seed)


# ---------------------------------------------------------------------------
# Consensus projection
# ---------------------------------------------------------------------------

def _consensus_update(
    x: NDArray,
    u: NDArray,
    index_map: IndexMap,
    partition: DomainPartition,
) -> NDArray:
    """Project x + u onto feasibility constraints.

    - Association variables: one-hot per UE  (argmax projection)
    - Epigraph variables:   one-hot          (argmax projection)
    - Slack variables:      round to {0, 1}
    """
    z = np.zeros_like(x)
    v = x + u

    # Association: for each UE pick the node with highest v
    for i in range(partition.num_ues):
        best_m = 0
        best_v = -float("inf")
        for m in range(partition.num_nodes):
            key = (i, m)
            if key not in index_map._assoc:
                continue
            flat = index_map.assoc(i, m)
            if v[flat] > best_v:
                best_v = v[flat]
                best_m = m
        z[index_map.assoc(i, best_m)] = 1.0

    # Epigraph: one-hot
    best_l = 0
    best_v = -float("inf")
    for l_val, flat in index_map._epigraph.items():
        if v[flat] > best_v:
            best_v = v[flat]
            best_l = l_val
    z[index_map.epigraph(best_l)] = 1.0

    # Slack variables: round to nearest binary
    for flat in index_map._cap_slack.values():
        z[flat] = 1.0 if v[flat] > 0.5 else 0.0
    for flat in index_map._aoii_slack.values():
        z[flat] = 1.0 if v[flat] > 0.5 else 0.0

    return z


# ---------------------------------------------------------------------------
# Broadcast global eta
# ---------------------------------------------------------------------------

def _broadcast_eta(
    z: NDArray,
    index_map: IndexMap,
    level_step: float,
) -> float:
    """Read the current worst-case AoII level from the consensus vector."""
    for l_val, flat in index_map._epigraph.items():
        if z[flat] > 0.5:
            return l_val * level_step
    return float("inf")


# ---------------------------------------------------------------------------
# Warm-start heuristics
# ---------------------------------------------------------------------------

def _greedy_warm_start(
    Q: NDArray,
    index_map: IndexMap,
    partition: DomainPartition,
    capacity: Optional[NDArray] = None,
) -> NDArray:
    """Greedy warm-start: each UE picks the node with the most negative
    QUBO diagonal entry, respecting capacity constraints."""
    N = index_map.num_vars
    x = np.zeros(N)
    load = np.zeros(partition.num_nodes, dtype=int)
    cap = capacity if capacity is not None else np.full(partition.num_nodes, partition.num_ues)

    # Sort UEs by how much they prefer their best node (most decisive first)
    ue_prefs: List[Tuple[float, int]] = []
    for i in range(partition.num_ues):
        diags = []
        for m in range(partition.num_nodes):
            if (i, m) in index_map._assoc:
                diags.append(Q[index_map.assoc(i, m), index_map.assoc(i, m)])
        diags.sort()
        gap = diags[1] - diags[0] if len(diags) > 1 else 0.0
        ue_prefs.append((-gap, i))  # largest gap first
    ue_prefs.sort()

    for _, i in ue_prefs:
        best_m = 0
        best_diag = float("inf")
        for m in range(partition.num_nodes):
            if (i, m) not in index_map._assoc:
                continue
            if load[m] >= cap[m]:
                continue
            flat = index_map.assoc(i, m)
            if Q[flat, flat] < best_diag:
                best_diag = Q[flat, flat]
                best_m = m
        # Fallback: if all nodes full, pick best regardless
        if best_diag == float("inf"):
            for m in range(partition.num_nodes):
                if (i, m) not in index_map._assoc:
                    continue
                flat = index_map.assoc(i, m)
                if Q[flat, flat] < best_diag:
                    best_diag = Q[flat, flat]
                    best_m = m
        x[index_map.assoc(i, best_m)] = 1.0
        load[best_m] += 1

    # Set eta to the level that minimises the diagonal
    best_l = 0
    best_diag = float("inf")
    for l_val, flat in index_map._epigraph.items():
        if Q[flat, flat] < best_diag:
            best_diag = Q[flat, flat]
            best_l = l_val
    x[index_map.epigraph(best_l)] = 1.0

    return x


def _random_warm_start(
    index_map: IndexMap,
    partition: DomainPartition,
    rng: np.random.Generator,
) -> NDArray:
    """Random feasible warm-start."""
    N = index_map.num_vars
    x = np.zeros(N)
    for i in range(partition.num_ues):
        m = int(rng.integers(0, partition.num_nodes))
        x[index_map.assoc(i, m)] = 1.0
    mid_l = partition.num_levels // 2
    x[index_map.epigraph(mid_l)] = 1.0
    return x


# ---------------------------------------------------------------------------
# Single ADMM run (internal)
# ---------------------------------------------------------------------------

def _admm_run(
    Q: NDArray,
    index_map: IndexMap,
    partition: DomainPartition,
    all_domains: List[List[int]],
    x_init: NDArray,
    level_step: float,
    capacity: Optional[NDArray],
    max_iter: int,
    epsilon: float,
    rho: float,
    rho_update: float,
    sub_max_exact: int,
    sub_sa_reads: int,
    sub_sa_sweeps: int,
    seed: Optional[int],
) -> Dict:
    """Execute one ADMM run from a given starting point."""
    N = index_map.num_vars
    x = x_init.copy()

    z = _consensus_update(x, np.zeros(N), index_map, partition)
    u = np.zeros(N)

    primal_residuals: List[float] = []
    dual_residuals: List[float] = []
    energies: List[float] = []
    etas: List[float] = []
    converged = False
    final_iter = 0
    current_rho = rho

    for iteration in range(max_iter):
        z_old = z.copy()

        # Step 1: x-update
        for domain_idx in all_domains:
            if not domain_idx:
                continue
            Q_sub = _build_sub_qubo(Q, domain_idx, x, z, u, current_rho)
            x_sub = _solve_sub(
                Q_sub, max_exact=sub_max_exact,
                sa_reads=sub_sa_reads, sa_sweeps=sub_sa_sweeps, seed=seed,
            )
            for s, g in enumerate(domain_idx):
                x[g] = x_sub[s]

        # Step 2: z-update
        z = _consensus_update(x, u, index_map, partition)

        # Step 3: u-update
        u = u + x - z

        # Step 4: residuals & convergence
        primal_res = float(np.linalg.norm(x - z))
        dual_res = float(current_rho * np.linalg.norm(z - z_old))
        primal_residuals.append(primal_res)
        dual_residuals.append(dual_res)

        e = evaluate_qubo(Q, z)
        energies.append(e)

        eta_val = _broadcast_eta(z, index_map, level_step)
        etas.append(eta_val)

        final_iter = iteration + 1
        if primal_res < epsilon:
            converged = True
            break

        current_rho *= rho_update

    return {
        "x": z.copy(),
        "energy": evaluate_qubo(Q, z),
        "iterations": final_iter,
        "converged": converged,
        "primal_residuals": primal_residuals,
        "dual_residuals": dual_residuals,
        "energy_history": energies,
        "eta_history": etas,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_admm(
    Q: NDArray,
    index_map: IndexMap,
    level_step: float = 1.0,
    capacity: Optional[NDArray] = None,
    max_iter: int = 50,
    epsilon: float = 1e-3,
    rho: float = 2.0,
    rho_update: float = 1.0,
    sub_max_exact: int = 15,
    sub_sa_reads: int = 50,
    sub_sa_sweeps: int = 500,
    seed: Optional[int] = None,
    warm_start: Optional[NDArray] = None,
    num_restarts: int = 3,
) -> Dict:
    """Solve a VANGUARD QUBO via ADMM decomposition.

    Parameters
    ----------
    Q : (N, N) upper-triangular QUBO matrix.
    index_map : IndexMap from build_vanguard_qubo.
    level_step : AoII quantisation step.
    capacity : (M,) node capacities.
    max_iter : maximum ADMM iterations.
    epsilon : primal residual convergence threshold.
    rho : initial ADMM penalty parameter.
    rho_update : multiplicative factor for rho each iteration (1.0 = fixed).
    sub_max_exact : sub-QUBOs with N <= this use brute-force.
    sub_sa_reads : SA restarts for larger sub-QUBOs.
    sub_sa_sweeps : SA sweeps for larger sub-QUBOs.
    seed : RNG seed.
    warm_start : optional initial binary vector (overrides internal heuristic).
    num_restarts : number of restarts (greedy + random starts, best kept).

    Returns
    -------
    dict with keys: x, energy, solver, iterations, primal_residuals,
    converged, eta, timing_s, plus decoded solution fields.
    """
    t0 = time.perf_counter()

    N = index_map.num_vars
    partition = partition_by_node(index_map)
    all_domains = partition.node_domains + [partition.global_domain]

    rng = np.random.default_rng(seed)

    # -- Build starting points -----------------------------------------------
    starts: List[NDArray] = []
    if warm_start is not None:
        starts.append(warm_start)
    else:
        starts.append(_greedy_warm_start(Q, index_map, partition, capacity))
        for _ in range(max(0, num_restarts - 1)):
            starts.append(_random_warm_start(index_map, partition, rng))

    # -- Run ADMM from each start, keep best ---------------------------------
    best_result: Optional[Dict] = None
    best_energy = float("inf")

    for x_init in starts:
        run_result = _admm_run(
            Q, index_map, partition, all_domains, x_init,
            level_step, capacity, max_iter, epsilon, rho, rho_update,
            sub_max_exact, sub_sa_reads, sub_sa_sweeps, seed,
        )
        if run_result["energy"] < best_energy:
            best_energy = run_result["energy"]
            best_result = run_result

    elapsed = time.perf_counter() - t0

    result: Dict = {
        "x": best_result["x"],
        "energy": best_result["energy"],
        "solver": "admm",
        "iterations": best_result["iterations"],
        "converged": best_result["converged"],
        "primal_residuals": best_result["primal_residuals"],
        "dual_residuals": best_result["dual_residuals"],
        "energy_history": best_result["energy_history"],
        "eta_history": best_result["eta_history"],
        "num_domains": len(all_domains),
        "domain_sizes": [len(d) for d in all_domains],
        "timing_s": elapsed,
        "num_restarts": len(starts),
    }

    result.update(extract_solution(best_result["x"], index_map, level_step, capacity))
    return result
