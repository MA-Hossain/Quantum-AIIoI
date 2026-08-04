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
    3.  u-update   — dual variable:  u <- u + x - z
    4.  Convergence check:  ||x - z|| < epsilon
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
    Q_sym: Optional[NDArray] = None,
) -> NDArray:
    """Build augmented sub-QUBO for one domain (vectorized).

    1. Extract Q submatrix for the domain's variables.
    2. Add coupling from fixed variables (other domains) as linear terms.
    3. Add ADMM penalty  (rho/2)(1 - 2(z_j - u_j))  on diagonal.
    """
    darr = np.array(domain_indices)
    n_d = len(darr)

    # Extract sub-matrix (domain indices are sorted, so Q[darr][:,darr]
    # preserves upper-triangular structure)
    Q_sub = np.triu(Q[np.ix_(darr, darr)]).copy()

    # Coupling with out-of-domain fixed variables (vectorized)
    if Q_sym is None:
        Q_sym = Q + Q.T - np.diag(np.diag(Q))
    full_coupling = Q_sym[darr, :] @ x_full          # (n_d,)
    domain_coupling = Q_sym[np.ix_(darr, darr)] @ x_full[darr]  # (n_d,)
    external_coupling = full_coupling - domain_coupling  # (n_d,)

    # Add external coupling + ADMM penalty to diagonal
    admm_diag = (rho / 2.0) * (1.0 - 2.0 * (z[darr] - u[darr]))
    diag_idx = np.arange(n_d)
    Q_sub[diag_idx, diag_idx] += external_coupling + admm_diag

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


def _solve_sub_qaoa(Q_sub: NDArray, seed: Optional[int],
                    qaoa_depth: int = 1, qaoa_shots: int = 1024) -> NDArray:
    """Solve a sub-QUBO with QAOA (Qiskit Aer)."""
    from sagin_research_sim.solver.solve_qaoa import solve_qaoa
    res = solve_qaoa(Q_sub, p=qaoa_depth, shots=qaoa_shots,
                     maxiter=80, seed=seed)
    return res["x"]


def _solve_sub(Q_sub: NDArray, max_exact: int = 15,
               sa_reads: int = 50, sa_sweeps: int = 500,
               seed: Optional[int] = None,
               sub_solver: str = "auto",
               qaoa_depth: int = 1,
               qaoa_shots: int = 1024) -> NDArray:
    """Select solver for sub-QUBO.

    sub_solver: "auto" (exact if small, else SA), "sa", "qaoa", "exact".
    """
    n = Q_sub.shape[0]
    if sub_solver == "qaoa" and n <= 20:
        return _solve_sub_qaoa(Q_sub, seed, qaoa_depth, qaoa_shots)
    if sub_solver == "exact" or (sub_solver == "auto" and n <= max_exact):
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
# Random warm-start
# ---------------------------------------------------------------------------

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
    sub_solver: str = "auto",
    qaoa_depth: int = 1,
    qaoa_shots: int = 1024,
) -> Dict:
    """Execute one ADMM run from a given starting point.

    Tracks TWO solution streams:
    - x-repaired: the sub-QUBO solutions rounded to feasibility
      (what the distributed solvers actually produce)
    - z: the consensus projection (argmax over x+u)
    Both are evaluated on the full QUBO; the best of each is kept.
    """
    N = index_map.num_vars
    x = x_init.copy()
    u_zero = np.zeros(N)

    # Precompute symmetric Q for vectorized coupling (done once)
    Q_sym = Q + Q.T - np.diag(np.diag(Q))

    z = _consensus_update(x, u_zero, index_map, partition)
    u = np.zeros(N)

    # Track best from EACH source independently
    best_x_repaired = _consensus_update(x, u_zero, index_map, partition)
    best_x_energy = evaluate_qubo(Q, best_x_repaired)

    best_z = z.copy()
    best_z_energy = evaluate_qubo(Q, z)

    primal_residuals: List[float] = []
    dual_residuals: List[float] = []
    x_energies: List[float] = []
    z_energies: List[float] = []
    converged = False
    final_iter = 0
    current_rho = rho

    for iteration in range(max_iter):
        z_old = z.copy()

        # Step 1: x-update (sub-QUBO solves — the real distributed work)
        for domain_idx in all_domains:
            if not domain_idx:
                continue
            Q_sub = _build_sub_qubo(Q, domain_idx, x, z, u, current_rho,
                                    Q_sym=Q_sym)
            x_sub = _solve_sub(
                Q_sub, max_exact=sub_max_exact,
                sa_reads=sub_sa_reads, sa_sweeps=sub_sa_sweeps, seed=seed,
                sub_solver=sub_solver, qaoa_depth=qaoa_depth,
                qaoa_shots=qaoa_shots,
            )
            for s, g in enumerate(domain_idx):
                x[g] = x_sub[s]

        # What did the sub-QUBO solves actually produce?
        # Round x to feasibility WITHOUT the dual bias u.
        x_repaired = _consensus_update(x, u_zero, index_map, partition)
        e_x = evaluate_qubo(Q, x_repaired)
        x_energies.append(e_x)
        if e_x < best_x_energy:
            best_x_energy = e_x
            best_x_repaired = x_repaired.copy()

        # Step 2: z-update (consensus projection with dual bias)
        z = _consensus_update(x, u, index_map, partition)
        e_z = evaluate_qubo(Q, z)
        z_energies.append(e_z)
        if e_z < best_z_energy:
            best_z_energy = e_z
            best_z = z.copy()

        # Step 3: u-update
        u = u + x - z

        # Step 4: residuals & convergence
        primal_res = float(np.linalg.norm(x - z))
        dual_res = float(current_rho * np.linalg.norm(z - z_old))
        primal_residuals.append(primal_res)
        dual_residuals.append(dual_res)

        final_iter = iteration + 1
        # Require minimum iterations so dual variables u build up
        # consensus pressure before checking convergence
        if iteration >= 9 and primal_res < epsilon:
            converged = True
            break

        current_rho *= rho_update

    return {
        "x_best": best_x_repaired,
        "x_energy": best_x_energy,
        "z_best": best_z,
        "z_energy": best_z_energy,
        "iterations": final_iter,
        "converged": converged,
        "primal_residuals": primal_residuals,
        "dual_residuals": dual_residuals,
        "x_energy_history": x_energies,
        "z_energy_history": z_energies,
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
    rho_update: float = 1.05,
    sub_max_exact: int = 15,
    sub_sa_reads: int = 50,
    sub_sa_sweeps: int = 500,
    seed: Optional[int] = None,
    warm_start: Optional[NDArray] = None,
    num_restarts: int = 3,
    sub_solver: str = "auto",
    qaoa_depth: int = 1,
    qaoa_shots: int = 1024,
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
    rho_update : multiplicative factor for rho each iteration.
    sub_max_exact : sub-QUBOs with N <= this use brute-force.
    sub_sa_reads : SA restarts for larger sub-QUBOs.
    sub_sa_sweeps : SA sweeps for larger sub-QUBOs.
    seed : RNG seed.
    warm_start : optional initial binary vector.  When None, all
        restarts use random initialisation (cold start).
    num_restarts : number of independent ADMM runs (best kept).

    Returns
    -------
    dict with keys: x, energy, solver, iterations, primal_residuals,
    converged, timing_s, solution_source, plus decoded solution fields.
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
    for _ in range(max(0, num_restarts - len(starts))):
        starts.append(_random_warm_start(index_map, partition, rng))

    # -- Run ADMM from each start, keep best ---------------------------------
    best_result: Optional[Dict] = None
    best_energy = float("inf")

    for x_init in starts:
        run_result = _admm_run(
            Q, index_map, partition, all_domains, x_init,
            level_step, capacity, max_iter, epsilon, rho, rho_update,
            sub_max_exact, sub_sa_reads, sub_sa_sweeps, seed,
            sub_solver=sub_solver, qaoa_depth=qaoa_depth,
            qaoa_shots=qaoa_shots,
        )
        # Take the better of x-update and z-projection for this run
        run_best_energy = min(run_result["x_energy"], run_result["z_energy"])
        if run_best_energy < best_energy:
            best_energy = run_best_energy
            best_result = run_result

    elapsed = time.perf_counter() - t0

    # Choose the solution source with lower QUBO energy
    if best_result["x_energy"] <= best_result["z_energy"]:
        best_x = best_result["x_best"]
        reported_energy = best_result["x_energy"]
        source = "x_update"
    else:
        best_x = best_result["z_best"]
        reported_energy = best_result["z_energy"]
        source = "z_projection"

    result: Dict = {
        "x": best_x,
        "energy": reported_energy,
        "x_energy": best_result["x_energy"],
        "z_energy": best_result["z_energy"],
        "solution_source": source,
        "solver": "admm",
        "iterations": best_result["iterations"],
        "converged": best_result["converged"],
        "primal_residuals": best_result["primal_residuals"],
        "dual_residuals": best_result["dual_residuals"],
        "x_energy_history": best_result["x_energy_history"],
        "z_energy_history": best_result["z_energy_history"],
        "num_domains": len(all_domains),
        "domain_sizes": [len(d) for d in all_domains],
        "timing_s": elapsed,
        "num_restarts": len(starts),
    }

    result.update(extract_solution(best_x, index_map, level_step, capacity))
    return result
