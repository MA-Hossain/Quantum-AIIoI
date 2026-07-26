"""Simulated-annealing QUBO solver.

Primary backend: D-Wave neal (``dwave-neal`` package).
Fallback: pure-NumPy SA when neal is not installed.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from sagin_research_sim.solver.qubo import IndexMap, evaluate_qubo, extract_solution

# ---------------------------------------------------------------------------
# Neal availability
# ---------------------------------------------------------------------------

try:
    import neal as _neal          # type: ignore[import-untyped]

    _HAS_NEAL = True
except ImportError:
    _HAS_NEAL = False


def has_neal() -> bool:
    """Return True if D-Wave neal is importable."""
    return _HAS_NEAL


# ---------------------------------------------------------------------------
# QUBO → dict conversion
# ---------------------------------------------------------------------------

def _qubo_to_dict(Q: NDArray) -> Dict[Tuple[int, int], float]:
    """Convert an upper-triangular Q matrix to a sparse QUBO dict."""
    N = Q.shape[0]
    qubo: Dict[Tuple[int, int], float] = {}
    for i in range(N):
        for j in range(i, N):
            val = float(Q[i, j])
            if abs(val) > 1e-15:
                qubo[(i, j)] = val
    return qubo


# ---------------------------------------------------------------------------
# D-Wave neal backend
# ---------------------------------------------------------------------------

def _solve_neal(
    Q: NDArray,
    num_reads: int,
    num_sweeps: int,
    seed: Optional[int],
    initial_states: Optional[NDArray] = None,
) -> Tuple[NDArray, float]:
    """Run SimulatedAnnealingSampler from dwave-neal."""
    import dimod

    qubo = _qubo_to_dict(Q)
    sampler = _neal.SimulatedAnnealingSampler()

    kwargs = dict(num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)

    if initial_states is not None:
        # Convert binary vector(s) to dimod SampleSet for warm-starting
        if initial_states.ndim == 1:
            initial_states = initial_states.reshape(1, -1)
        samples = [
            {j: int(initial_states[k, j]) for j in range(initial_states.shape[1])}
            for k in range(initial_states.shape[0])
        ]
        init_ss = dimod.SampleSet.from_samples(
            samples, vartype="BINARY", energy=[0.0] * len(samples),
        )
        kwargs["initial_states"] = init_ss

    response = sampler.sample_qubo(qubo, **kwargs)
    best = response.first
    N = Q.shape[0]
    x = np.array([best.sample.get(i, 0) for i in range(N)], dtype=float)
    return x, float(best.energy)


# ---------------------------------------------------------------------------
# Pure-NumPy fallback SA
# ---------------------------------------------------------------------------

def _sa_sweep(
    Q: NDArray,
    x: NDArray,
    energy: float,
    rng: np.random.Generator,
    beta: float,
    N: int,
) -> Tuple[NDArray, float]:
    """One full sweep: attempt to flip each bit once."""
    for k in range(N):
        # Energy change from flipping bit k
        local = float(Q[k, k])
        if k + 1 < N:
            local += float(Q[k, k + 1 :] @ x[k + 1 :])
        if k > 0:
            local += float(Q[:k, k] @ x[:k])
        delta = (1.0 - 2.0 * x[k]) * local

        if delta < 0 or rng.random() < np.exp(-beta * delta):
            x[k] = 1.0 - x[k]
            energy += delta

    return x, energy


def _solve_numpy(
    Q: NDArray,
    num_reads: int,
    num_sweeps: int,
    seed: Optional[int],
    beta_range: Tuple[float, float],
) -> Tuple[NDArray, float]:
    """Pure-NumPy simulated annealing."""
    rng = np.random.default_rng(seed)
    N = Q.shape[0]
    beta_min, beta_max = beta_range

    best_energy = float("inf")
    best_x = np.zeros(N)

    for _ in range(num_reads):
        x = rng.integers(0, 2, size=N).astype(float)
        energy = float(x @ Q @ x)

        for step in range(num_sweeps):
            beta = beta_min + (beta_max - beta_min) * step / max(num_sweeps - 1, 1)
            x, energy = _sa_sweep(Q, x, energy, rng, beta, N)

        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()

    return best_x, best_energy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_sa(
    Q: NDArray,
    index_map: Optional[IndexMap] = None,
    level_step: float = 1.0,
    capacity: Optional[NDArray] = None,
    num_reads: int = 100,
    num_sweeps: int = 1000,
    seed: Optional[int] = None,
    beta_range: Tuple[float, float] = (0.1, 3.0),
    backend: Optional[str] = None,
    initial_states: Optional[NDArray] = None,
) -> Dict:
    """Solve a QUBO via simulated annealing.

    Parameters
    ----------
    Q : (N, N) upper-triangular QUBO matrix.
    index_map : optional IndexMap for decoding.
    level_step : AoII quantisation step.
    capacity : (M,) node capacities.
    num_reads : independent SA restarts.
    num_sweeps : sweeps (temperature steps) per read.
    seed : RNG seed for reproducibility.
    beta_range : (beta_min, beta_max) inverse-temperature schedule.
    backend : ``"neal"`` | ``"numpy"`` | ``None`` (auto-detect).

    Returns
    -------
    dict with keys: x, energy, solver, backend, timing_s,
    plus decoded fields when index_map is given.
    """
    if backend is None:
        backend = "neal" if _HAS_NEAL else "numpy"

    t0 = time.perf_counter()

    if backend == "neal":
        if not _HAS_NEAL:
            raise ImportError(
                "D-Wave neal is not installed. "
                "Install with: pip install dwave-neal"
            )
        x, energy = _solve_neal(Q, num_reads, num_sweeps, seed, initial_states)
    elif backend == "numpy":
        x, energy = _solve_numpy(Q, num_reads, num_sweeps, seed, beta_range)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    elapsed = time.perf_counter() - t0

    result: Dict = {
        "x": x,
        "energy": energy,
        "solver": "sa",
        "backend": backend,
        "num_reads": num_reads,
        "num_sweeps": num_sweeps,
        "timing_s": elapsed,
    }

    if index_map is not None:
        result.update(extract_solution(x, index_map, level_step, capacity))

    return result
