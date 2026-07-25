"""QAOA solver for QUBO problems using Qiskit Aer.

Pipeline:
    1. Convert upper-triangular QUBO matrix Q to Ising Hamiltonian (h, J)
    2. Build a p-layer QAOA circuit with parameters (gamma, beta)
    3. Optimise parameters with COBYLA (gradient-free)
    4. Return the best sampled bitstring as the QUBO solution
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from sagin_research_sim.solver.qubo import IndexMap, extract_solution


# ---------------------------------------------------------------------------
# QUBO  →  Ising
# ---------------------------------------------------------------------------

def qubo_to_ising(Q: NDArray) -> Tuple[NDArray, Dict[Tuple[int, int], float], float]:
    r"""Convert an upper-triangular QUBO to Ising form.

    Substitution  x_i = (1 - Z_i) / 2  yields:

        x^T Q x  =  offset  +  \sum_i h_i Z_i  +  \sum_{i<j} J_{ij} Z_i Z_j

    Returns
    -------
    h      : (N,) linear coefficients.
    J      : dict  {(i, j): J_ij}  for i < j, only non-zero entries.
    offset : constant energy shift.
    """
    N = Q.shape[0]
    h = np.zeros(N)
    J: Dict[Tuple[int, int], float] = {}
    offset = 0.0

    for i in range(N):
        offset += Q[i, i] / 2.0
        h[i] -= Q[i, i] / 2.0

    for i in range(N):
        for j in range(i + 1, N):
            qij = Q[i, j]
            if abs(qij) < 1e-15:
                continue
            offset += qij / 4.0
            h[i] -= qij / 4.0
            h[j] -= qij / 4.0
            J[(i, j)] = qij / 4.0

    return h, J, offset


# ---------------------------------------------------------------------------
# QAOA circuit
# ---------------------------------------------------------------------------

def build_qaoa_circuit(
    h: NDArray,
    J: Dict[Tuple[int, int], float],
    num_qubits: int,
    p: int,
    gamma: NDArray,
    beta: NDArray,
):
    """Construct a p-layer QAOA circuit.

    Cost unitary   U_C(gamma)  =  exp(-i gamma H_C)
        - Rz(2 gamma h_i)      for every qubit i
        - RZZ(2 gamma J_ij)    for every interacting pair (i, j)

    Mixer unitary  U_M(beta)   =  exp(-i beta sum_i X_i)
        - Rx(2 beta)           for every qubit
    """
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(num_qubits)

    # Initial superposition |+>^n
    qc.h(range(num_qubits))

    for k in range(p):
        # --- cost layer ---
        for i in range(num_qubits):
            if abs(h[i]) > 1e-15:
                qc.rz(2.0 * gamma[k] * h[i], i)
        for (i, j), jij in J.items():
            if abs(jij) > 1e-15:
                qc.rzz(2.0 * gamma[k] * jij, i, j)

        # --- mixer layer ---
        for i in range(num_qubits):
            qc.rx(2.0 * beta[k], i)

    qc.measure_all()
    return qc


# ---------------------------------------------------------------------------
# Helper: bitstring → x
# ---------------------------------------------------------------------------

def _bitstring_to_x(bitstring: str, N: int) -> NDArray:
    """Qiskit bitstrings are big-endian (qubit 0 = rightmost).  Reverse."""
    return np.array([int(b) for b in reversed(bitstring)], dtype=float)[:N]


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def solve_qaoa(
    Q: NDArray,
    index_map: Optional[IndexMap] = None,
    level_step: float = 1.0,
    capacity: Optional[NDArray] = None,
    p: int = 1,
    shots: int = 1024,
    maxiter: int = 100,
    seed: Optional[int] = None,
    initial_point: Optional[NDArray] = None,
) -> Dict:
    """Solve a QUBO via QAOA on the Qiskit Aer simulator.

    Parameters
    ----------
    Q : (N, N) upper-triangular QUBO matrix.
    p : QAOA depth (number of cost+mixer layers).
    shots : measurement shots per circuit evaluation.
    maxiter : maximum COBYLA iterations.
    seed : RNG seed (simulator + initial point).
    initial_point : shape (2p,) starting parameters [gamma; beta].

    Returns
    -------
    dict with keys: x, energy, solver, depth, num_qubits, opt_energy,
    opt_params, history, timing_s, and decoded fields when index_map given.
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    from scipy.optimize import minimize

    N = Q.shape[0]
    h, J, offset = qubo_to_ising(Q)

    backend = AerSimulator(seed_simulator=seed)

    history: Dict[str, List] = {"energies": [], "params": []}

    def objective(params: NDArray) -> float:
        gamma = params[:p]
        beta_p = params[p:]

        qc = build_qaoa_circuit(h, J, N, p, gamma, beta_p)
        t_qc = transpile(qc, backend)
        job = backend.run(t_qc, shots=shots, seed_simulator=seed)
        counts = job.result().get_counts()

        total = 0.0
        for bitstring, count in counts.items():
            x = _bitstring_to_x(bitstring, N)
            total += float(x @ Q @ x) * count
        avg = total / shots

        history["energies"].append(avg)
        history["params"].append(params.copy())
        return avg

    # Starting point
    if initial_point is None:
        rng = np.random.default_rng(seed)
        initial_point = rng.uniform(0, np.pi, size=2 * p)

    t0 = time.perf_counter()
    opt = minimize(objective, initial_point, method="COBYLA",
                   options={"maxiter": maxiter, "rhobeg": 0.5})
    elapsed = time.perf_counter() - t0

    # Final sampling with optimised parameters (more shots)
    best_gamma = opt.x[:p]
    best_beta = opt.x[p:]
    qc_final = build_qaoa_circuit(h, J, N, p, best_gamma, best_beta)
    t_qc = transpile(qc_final, backend)
    job = backend.run(t_qc, shots=shots * 4, seed_simulator=seed)
    counts = job.result().get_counts()

    best_energy = float("inf")
    best_x = np.zeros(N)
    for bitstring, count in counts.items():
        x = _bitstring_to_x(bitstring, N)
        e = float(x @ Q @ x)
        if e < best_energy:
            best_energy = e
            best_x = x.copy()

    result: Dict = {
        "x": best_x,
        "energy": best_energy,
        "solver": "qaoa",
        "depth": p,
        "num_qubits": N,
        "shots": shots,
        "opt_energy": float(opt.fun),
        "opt_params": opt.x.tolist(),
        "opt_nfev": opt.nfev,
        "history": history,
        "timing_s": elapsed,
    }

    if index_map is not None:
        result.update(extract_solution(best_x, index_map, level_step, capacity))

    return result


# ---------------------------------------------------------------------------
# Depth sweep helper
# ---------------------------------------------------------------------------

def sweep_depths(
    Q: NDArray,
    depths: List[int] = [1, 2, 3],
    index_map: Optional[IndexMap] = None,
    level_step: float = 1.0,
    capacity: Optional[NDArray] = None,
    shots: int = 1024,
    maxiter: int = 100,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Run QAOA at multiple depths and return results list."""
    results = []
    for p in depths:
        res = solve_qaoa(
            Q, index_map=index_map, level_step=level_step,
            capacity=capacity, p=p, shots=shots, maxiter=maxiter,
            seed=seed,
        )
        results.append(res)
    return results
