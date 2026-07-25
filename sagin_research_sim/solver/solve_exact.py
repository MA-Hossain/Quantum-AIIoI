"""Brute-force exact QUBO solver for small instances.

Enumerates all 2^N binary vectors and returns the global minimum of
x^T Q x.  Only feasible for N <= 25 or so.
"""

from __future__ import annotations

import itertools
import time
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from sagin_research_sim.solver.qubo import IndexMap, evaluate_qubo, extract_solution


def solve_exact(
    Q: NDArray,
    index_map: Optional[IndexMap] = None,
    level_step: float = 1.0,
    capacity: Optional[NDArray] = None,
    max_vars: int = 25,
) -> Dict:
    """Find the global minimum of x^T Q x by exhaustive enumeration.

    Parameters
    ----------
    Q : (N, N) upper-triangular QUBO matrix.
    index_map : optional IndexMap for decoding solutions.
    level_step : AoII quantisation step (for extract_solution).
    capacity : (M,) node capacities (for feasibility check).
    max_vars : refuse to run if N exceeds this (default 25 → 33 M evals).

    Returns
    -------
    dict with keys: x, energy, solver, num_evaluated, timing_s,
    plus decoded fields (association, eta, feasible …) when index_map given.
    """
    N = Q.shape[0]
    if N > max_vars:
        raise ValueError(
            f"Exact solver limited to N<={max_vars} variables, got N={N}. "
            f"Use solve_sa() for larger instances."
        )

    t0 = time.perf_counter()

    best_energy = float("inf")
    best_x = np.zeros(N)

    for bits in itertools.product([0, 1], repeat=N):
        x = np.array(bits, dtype=float)
        e = float(x @ Q @ x)
        if e < best_energy:
            best_energy = e
            best_x = x.copy()

    elapsed = time.perf_counter() - t0

    result: Dict = {
        "x": best_x,
        "energy": best_energy,
        "solver": "exact",
        "num_evaluated": 2 ** N,
        "timing_s": elapsed,
    }

    if index_map is not None:
        result.update(extract_solution(best_x, index_map, level_step, capacity))

    return result
