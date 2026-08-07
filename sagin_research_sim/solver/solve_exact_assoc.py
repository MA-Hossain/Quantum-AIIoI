"""Exact solvers for the *original* minimax AoII association problem.

The QUBO-space brute force in ``solve_exact`` enumerates 2^N binary vectors,
where N includes epigraph and slack variables.  Even the smallest experiment
instance (3 UEs, 4 nodes, L=10) needs N=42, which is far beyond enumeration.

The original problem is much smaller.  It asks only:

    minimise    max_i  aoii_cost[i, a(i)]
    over        a : {0..I-1} -> {0..M-1}
    subject to  |{ i : a(i) = m }|  <=  capacity[m]   for every node m

``aoii_cost`` already carries the worst case over disruption scenarios, so this
is exactly the robust objective reported by the experiment metrics.

Two solvers are provided:

``solve_exact_assoc_bruteforce``
    Enumerates all M^I assignments.  Obviously correct, used as the reference
    implementation to validate the fast solver.  Feasible to ~10 UEs.

``solve_exact_assoc``
    Bottleneck assignment via threshold search plus bipartite max-flow
    feasibility.  Returns the same optimum as brute force but runs in
    polynomial time, so exact ground truth is available at experiment scale
    (hundreds of UEs) rather than only on toy instances.
"""

from __future__ import annotations

import itertools
import time
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray


def _feasible_at_threshold(
    aoii_cost: NDArray,
    capacity: NDArray,
    threshold: float,
    tol: float = 1e-9,
) -> Optional[Dict[int, int]]:
    """Assign every UE to a node with cost <= threshold, respecting capacity.

    Modelled as a max-flow problem:

        source -> UE i            capacity 1
        UE i   -> node m          capacity 1, only if aoii_cost[i,m] <= threshold
        node m -> sink            capacity[m]

    A saturating flow (value == I) exists exactly when a feasible assignment
    within the threshold exists.

    Returns the assignment dict, or None when no feasible assignment exists.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    num_ues, num_nodes = aoii_cost.shape

    # Node layout: 0 = source, 1..I = UEs, I+1..I+M = nodes, I+M+1 = sink
    source = 0
    ue_base = 1
    node_base = 1 + num_ues
    sink = 1 + num_ues + num_nodes
    size = sink + 1

    rows, cols, vals = [], [], []

    for i in range(num_ues):
        rows.append(source)
        cols.append(ue_base + i)
        vals.append(1)

    allowed = aoii_cost <= threshold + tol
    ue_idx, node_idx = np.nonzero(allowed)
    for i, m in zip(ue_idx, node_idx):
        rows.append(ue_base + int(i))
        cols.append(node_base + int(m))
        vals.append(1)

    for m in range(num_nodes):
        cap_m = int(capacity[m])
        if cap_m <= 0:
            continue
        rows.append(node_base + m)
        cols.append(sink)
        vals.append(cap_m)

    graph = csr_matrix(
        (np.array(vals, dtype=np.int32),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(size, size),
    )

    res = maximum_flow(graph, source, sink)
    if int(res.flow_value) < num_ues:
        return None

    # Decode: a UE->node edge carrying flow is an assignment.
    flow = res.flow.tocoo()
    association: Dict[int, int] = {}
    for r, c, f in zip(flow.row, flow.col, flow.data):
        if f <= 0:
            continue
        if ue_base <= r < ue_base + num_ues and node_base <= c < node_base + num_nodes:
            association[int(r) - ue_base] = int(c) - node_base

    if len(association) != num_ues:
        return None
    return association


def solve_exact_assoc(
    aoii_cost: NDArray,
    capacity: NDArray,
) -> Dict:
    """Exact minimax (bottleneck) association via threshold search + max-flow.

    The optimal worst-case AoII is always one of the values appearing in
    ``aoii_cost``, so binary searching the sorted distinct values yields the
    exact optimum in O(log(IM)) feasibility checks.

    Returns
    -------
    dict with keys: association, worst_aoii, solver, timing_s, num_thresholds_tested.
    """
    aoii_cost = np.asarray(aoii_cost, dtype=float)
    capacity = np.asarray(capacity)
    num_ues, _ = aoii_cost.shape

    if int(np.sum(capacity)) < num_ues:
        raise ValueError(
            f"Infeasible instance: total capacity {int(np.sum(capacity))} "
            f"< {num_ues} UEs."
        )

    t0 = time.perf_counter()

    candidates = np.unique(aoii_cost)
    lo, hi = 0, len(candidates) - 1
    best_assoc = _feasible_at_threshold(aoii_cost, capacity, candidates[hi])
    if best_assoc is None:
        raise ValueError(
            "Infeasible instance: no assignment satisfies the capacity limits."
        )
    best_idx = hi
    tested = 1

    while lo <= hi:
        mid = (lo + hi) // 2
        assoc = _feasible_at_threshold(aoii_cost, capacity, candidates[mid])
        tested += 1
        if assoc is not None:
            best_assoc = assoc
            best_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1

    worst = max(float(aoii_cost[i, m]) for i, m in best_assoc.items())

    return {
        "association": best_assoc,
        "worst_aoii": worst,
        "optimal_threshold": float(candidates[best_idx]),
        "solver": "exact_assoc",
        "timing_s": time.perf_counter() - t0,
        "num_thresholds_tested": tested,
    }


def solve_exact_assoc_bruteforce(
    aoii_cost: NDArray,
    capacity: NDArray,
    max_assignments: int = 20_000_000,
) -> Dict:
    """Reference implementation: enumerate all M^I assignments.

    Used to validate ``solve_exact_assoc``.  Raises when the enumeration would
    exceed ``max_assignments``.
    """
    aoii_cost = np.asarray(aoii_cost, dtype=float)
    capacity = np.asarray(capacity)
    num_ues, num_nodes = aoii_cost.shape

    total = num_nodes ** num_ues
    if total > max_assignments:
        raise ValueError(
            f"Brute force would enumerate {total} assignments "
            f"(limit {max_assignments}). Use solve_exact_assoc() instead."
        )

    t0 = time.perf_counter()

    best_worst = float("inf")
    best_assoc: Optional[Dict[int, int]] = None
    cap = capacity.astype(int)

    for combo in itertools.product(range(num_nodes), repeat=num_ues):
        load = np.bincount(np.asarray(combo), minlength=num_nodes)
        if np.any(load > cap):
            continue
        worst = max(aoii_cost[i, m] for i, m in enumerate(combo))
        if worst < best_worst - 1e-12:
            best_worst = float(worst)
            best_assoc = {i: int(m) for i, m in enumerate(combo)}

    if best_assoc is None:
        raise ValueError(
            "Infeasible instance: no assignment satisfies the capacity limits."
        )

    return {
        "association": best_assoc,
        "worst_aoii": best_worst,
        "solver": "exact_assoc_bruteforce",
        "timing_s": time.perf_counter() - t0,
        "num_evaluated": total,
    }
