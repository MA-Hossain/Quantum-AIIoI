"""Accumulated Age of Incorrect Information (AoII) via Markov renewal theory.

Given a Markov source with transition matrix P_i and distortion f(x, x_hat),
compute the expected accumulated AoII over an inter-update interval of length Y.

After a successful update at t=0, the monitor's estimate equals the true state.
Over the next Y slots the source evolves via P while the estimate stays frozen.

    Psi(Y) = sum_{t=1}^{Y} sum_{i,j} pi(i) * [P^t]_{i,j} * f(i, j)

where pi is the stationary distribution of P and f(x, x_hat) = |x - x_hat|.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def stationary_distribution(P: NDArray) -> NDArray:
    """Compute the stationary distribution of an irreducible transition matrix."""
    S = P.shape[0]
    # Solve pi @ P = pi  =>  pi @ (P - I) = 0, with sum(pi) = 1
    A = (P.T - np.eye(S))
    # Replace last row with normalization constraint sum(pi) = 1
    A[-1, :] = 1.0
    b = np.zeros(S)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    return pi


def distortion_matrix(S: int) -> NDArray:
    """Build S x S distortion matrix F[i,j] = |i - j|."""
    idx = np.arange(S)
    return np.abs(idx[:, None] - idx[None, :]).astype(float)


def psi(P: NDArray, f: NDArray, Y: int) -> float:
    """Expected accumulated AoII over interval of length Y.

    Parameters
    ----------
    P : (S, S) transition matrix
    f : (S, S) distortion matrix, f[i,j] = distortion when true state was i
        and current state is j (estimate frozen at i)
    Y : inter-update interval length (number of slots)

    Returns
    -------
    Accumulated expected AoII (scalar).
    """
    pi = stationary_distribution(P)
    S = P.shape[0]
    total = 0.0
    P_power = np.eye(S)
    for t in range(1, Y + 1):
        P_power = P_power @ P
        # q(t) = sum_{i,j} pi(i) * P^t[i,j] * f[i,j]
        weighted = (pi[:, None] * P_power) * f
        total += weighted.sum()
    return float(total)


def precompute_psi(P: NDArray, Y_max: int, f: NDArray | None = None) -> NDArray:
    """Precompute Psi(Y) for Y = 0, 1, ..., Y_max.

    Parameters
    ----------
    P : (S, S) transition matrix
    Y_max : maximum interval length
    f : (S, S) distortion matrix. If None, uses |i - j|.

    Returns
    -------
    psi_table : (Y_max + 1,) array where psi_table[Y] = Psi(Y).
                psi_table[0] = 0 by convention.
    """
    S = P.shape[0]
    if f is None:
        f = distortion_matrix(S)
    pi = stationary_distribution(P)

    psi_table = np.zeros(Y_max + 1)
    P_power = np.eye(S)
    cumulative = 0.0

    for t in range(1, Y_max + 1):
        P_power = P_power @ P
        q_t = (pi[:, None] * P_power * f).sum()
        cumulative += q_t
        psi_table[t] = cumulative

    return psi_table
