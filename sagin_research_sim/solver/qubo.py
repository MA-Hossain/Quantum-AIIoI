"""VANGUARD QUBO builder for joint association-scheduling in SAGIN.

Encodes the minimax AoII user-association problem as a Quadratic
Unconstrained Binary Optimization (QUBO):

    min  x^T Q x

Binary variables (multi-index x_{i,m,q,r,n,l}):
    x_{i,m}        association — UE i served by node m
    eta_l          epigraph   — one-hot worst-case AoII level
    s_{m,b}        capacity slack bits (inequality -> equality)
    sigma_{i,b}    AoII slack bits (epigraph inequality -> equality)

Quadratic penalties enforce:
    (C1) Single-association   sum_m x_{i,m} = 1             for all i
    (C2) Capacity             sum_i x_{i,m} + slack = C_m   for all m
    (C3) Robust AoII          AoII_i + slack_i = eta        for all i
    (C4) Epigraph one-hot     sum_l eta_l = 1

The objective (linear in eta) plus the penalties form a single QUBO
matrix Q whose ground state encodes the optimal association.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Variable bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class VariableIndex:
    """Describes one binary variable in the QUBO.

    The six-index notation x_{i,m,q,r,n,l} is stored sparsely: unused
    indices are set to -1.
    """

    flat: int
    var_type: str          # "assoc" | "epigraph" | "cap_slack" | "aoii_slack"
    i: int = -1            # UE index
    m: int = -1            # serving-node index
    q: int = -1            # update-generation flag
    r: int = -1            # resource-block index
    n: int = -1            # disruption-scenario index
    l: int = -1            # AoII level / slack bit position

    @property
    def label(self) -> str:
        if self.var_type == "assoc":
            return f"x[{self.i},{self.m}]"
        if self.var_type == "epigraph":
            return f"eta[{self.l}]"
        if self.var_type == "cap_slack":
            return f"cs[{self.m},{self.l}]"
        if self.var_type == "aoii_slack":
            return f"as[{self.i},{self.l}]"
        return f"v[{self.flat}]"


class IndexMap:
    """Bidirectional mapping between multi-index tuples and flat indices."""

    def __init__(self) -> None:
        self.variables: List[VariableIndex] = []
        self.num_vars: int = 0
        self._assoc: Dict[Tuple[int, int], int] = {}
        self._epigraph: Dict[int, int] = {}
        self._cap_slack: Dict[Tuple[int, int], int] = {}
        self._aoii_slack: Dict[Tuple[int, int], int] = {}

    # -- registration helpers ------------------------------------------------

    def _add(self, vi: VariableIndex) -> int:
        idx = self.num_vars
        vi.flat = idx
        self.variables.append(vi)
        self.num_vars += 1
        return idx

    def add_assoc(self, i: int, m: int) -> int:
        idx = self._add(VariableIndex(flat=0, var_type="assoc", i=i, m=m, q=1))
        self._assoc[(i, m)] = idx
        return idx

    def add_epigraph(self, l: int) -> int:
        idx = self._add(VariableIndex(flat=0, var_type="epigraph", l=l))
        self._epigraph[l] = idx
        return idx

    def add_cap_slack(self, m: int, bit: int) -> int:
        idx = self._add(VariableIndex(flat=0, var_type="cap_slack", m=m, l=bit))
        self._cap_slack[(m, bit)] = idx
        return idx

    def add_aoii_slack(self, i: int, bit: int) -> int:
        idx = self._add(VariableIndex(flat=0, var_type="aoii_slack", i=i, l=bit))
        self._aoii_slack[(i, bit)] = idx
        return idx

    # -- fast lookups --------------------------------------------------------

    def assoc(self, i: int, m: int) -> int:
        return self._assoc[(i, m)]

    def epigraph(self, l: int) -> int:
        return self._epigraph[l]

    def cap_slack(self, m: int, bit: int) -> int:
        return self._cap_slack[(m, bit)]

    def aoii_slack(self, i: int, bit: int) -> int:
        return self._aoii_slack[(i, bit)]

    # -- solution decoding ---------------------------------------------------

    def decode_solution(self, x: NDArray) -> Dict:
        """Decode a binary vector into structured fields."""
        association: Dict[int, int] = {}
        for (i, m), idx in self._assoc.items():
            if x[idx] > 0.5:
                association[i] = m

        eta_level = -1
        for l_val, idx in self._epigraph.items():
            if x[idx] > 0.5:
                eta_level = l_val

        return {"association": association, "eta_level": eta_level}


# ---------------------------------------------------------------------------
# Penalty helper
# ---------------------------------------------------------------------------

def _add_equality_penalty(
    Q: NDArray,
    var_indices: List[int],
    coeffs: List[float],
    rhs: float,
    weight: float,
) -> None:
    r"""Add  weight * (sum_j c_j z_j  -  rhs)^2  to upper-triangular *Q*.

    For binary z_j, expansion is:

        sum_j (c_j^2 - 2 rhs c_j) z_j
      + 2 sum_{j<k} c_j c_k z_j z_k
      + rhs^2  (constant, dropped)
    """
    for j, (vj, cj) in enumerate(zip(var_indices, coeffs)):
        Q[vj, vj] += weight * (cj * cj - 2.0 * rhs * cj)
        for k in range(j + 1, len(var_indices)):
            vk, ck = var_indices[k], coeffs[k]
            r, c = (vj, vk) if vj < vk else (vk, vj)
            Q[r, c] += weight * 2.0 * cj * ck


# ---------------------------------------------------------------------------
# AoII cost pre-computation
# ---------------------------------------------------------------------------

def compute_aoii_cost_matrix(
    num_ues: int,
    num_nodes: int,
    rate_matrix: NDArray,
    psi_table: NDArray,
    packet_size_bits: float,
    scenario_rate_matrices: Optional[List[NDArray]] = None,
) -> NDArray:
    """Worst-case AoII cost c[i, m] for each UE-node pair.

    For every scenario the inter-update interval is

        Y_{i,m,n} = ceil(packet_size / R_{i,m,n})

    and the AoII is Psi(Y).  The returned matrix takes the worst case
    (maximum) over all scenarios.
    """
    Y_max = len(psi_table) - 1
    all_rates = [rate_matrix]
    if scenario_rate_matrices is not None:
        all_rates.extend(scenario_rate_matrices)

    cost = np.zeros((num_ues, num_nodes))
    for i in range(num_ues):
        for m in range(num_nodes):
            worst_psi = 0.0
            for rates in all_rates:
                r = rates[i, m]
                if r <= 0:
                    y = Y_max
                else:
                    y = min(int(np.ceil(packet_size_bits / r)), Y_max)
                worst_psi = max(worst_psi, float(psi_table[y]))
            cost[i, m] = worst_psi
    return cost


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_vanguard_qubo(
    num_ues: int,
    num_nodes: int,
    psi_table: NDArray,
    rate_matrix: NDArray,
    capacity: NDArray,
    packet_size_bits: float = 1000.0,
    num_levels: int = 8,
    level_step: float = 1.0,
    penalty_single_assoc: float = 10.0,
    penalty_capacity: float = 8.0,
    penalty_aoii: float = 6.0,
    penalty_epigraph: float = 10.0,
    scenario_rate_matrices: Optional[List[NDArray]] = None,
) -> Tuple[NDArray, IndexMap]:
    """Build the VANGUARD QUBO matrix.

    Parameters
    ----------
    num_ues : int
        Number of UEs (I).
    num_nodes : int
        Number of serving nodes, M = |BS| + |SAT|.
    psi_table : (Y_max+1,) array
        Pre-computed Psi(Y) from AoII renewal theory.
    rate_matrix : (I, M) array
        Achievable uplink rate R[i, m] in bps (nominal scenario).
    capacity : (M,) array
        Maximum number of UEs each node can serve.
    packet_size_bits : float
        Status-update packet size in bits.
    num_levels : int
        Number of discrete AoII levels L for the epigraph encoding.
    level_step : float
        Quantisation step between consecutive AoII levels.
    penalty_* : float
        Penalty weights for each constraint class.
    scenario_rate_matrices : list of (I, M) arrays, optional
        Rate matrices for disruption scenarios (robust AoII).

    Returns
    -------
    Q : (N, N) upper-triangular QUBO matrix.
    index_map : IndexMap mapping flat indices to multi-index variables.
    """
    I, M, L = num_ues, num_nodes, num_levels

    # -- Pre-compute worst-case AoII costs -----------------------------------
    aoii_cost = compute_aoii_cost_matrix(
        I, M, rate_matrix, psi_table, packet_size_bits, scenario_rate_matrices,
    )
    # Discretise to integer multiples of level_step
    aoii_cost_d = np.clip(np.round(aoii_cost / level_step).astype(int), 0, L - 1)

    # -- Build index map -----------------------------------------------------
    idx = IndexMap()

    for i in range(I):
        for m in range(M):
            idx.add_assoc(i, m)

    for l in range(L):
        idx.add_epigraph(l)

    cap_slack_bits: Dict[int, int] = {}
    for m in range(M):
        C_m = int(capacity[m])
        n_bits = max(1, int(np.ceil(np.log2(max(C_m + 1, 2)))))
        cap_slack_bits[m] = n_bits
        for b in range(n_bits):
            idx.add_cap_slack(m, b)

    aoii_slack_nbits = max(1, int(np.ceil(np.log2(max(L, 2)))))
    for i in range(I):
        for b in range(aoii_slack_nbits):
            idx.add_aoii_slack(i, b)

    N = idx.num_vars
    Q = np.zeros((N, N))

    # === Objective: minimise eta = sum_l  l * level_step * eta_l =============
    for l in range(L):
        Q[idx.epigraph(l), idx.epigraph(l)] += l * level_step

    # === (C1) Single-association =============================================
    for i in range(I):
        vis = [idx.assoc(i, m) for m in range(M)]
        cfs = [1.0] * M
        _add_equality_penalty(Q, vis, cfs, 1.0, penalty_single_assoc)

    # === (C2) Capacity =======================================================
    for m in range(M):
        C_m = int(capacity[m])
        n_bits = cap_slack_bits[m]
        vis = [idx.assoc(i, m) for i in range(I)]
        cfs: List[float] = [1.0] * I
        for b in range(n_bits):
            vis.append(idx.cap_slack(m, b))
            cfs.append(float(2 ** b))
        _add_equality_penalty(Q, vis, cfs, float(C_m), penalty_capacity)

    # === (C3) Robust AoII epigraph ===========================================
    #   For each UE i:
    #     sum_m aoii_d[i,m] x_{i,m}  +  slack_i  =  sum_l l eta_l
    for i in range(I):
        vis: List[int] = []
        cfs_c3: List[float] = []
        for m in range(M):
            vis.append(idx.assoc(i, m))
            cfs_c3.append(float(aoii_cost_d[i, m]))
        for b in range(aoii_slack_nbits):
            vis.append(idx.aoii_slack(i, b))
            cfs_c3.append(float(2 ** b))
        for l in range(L):
            vis.append(idx.epigraph(l))
            cfs_c3.append(float(-l))
        _add_equality_penalty(Q, vis, cfs_c3, 0.0, penalty_aoii)

    # === (C4) Epigraph one-hot ===============================================
    eta_vis = [idx.epigraph(l) for l in range(L)]
    eta_cfs = [1.0] * L
    _add_equality_penalty(Q, eta_vis, eta_cfs, 1.0, penalty_epigraph)

    return Q, idx


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_qubo(Q: NDArray, x: NDArray) -> float:
    """Evaluate x^T Q x for a binary vector x."""
    return float(x @ Q @ x)


def extract_solution(
    x: NDArray,
    idx_map: IndexMap,
    level_step: float = 1.0,
    capacity: Optional[NDArray] = None,
) -> Dict:
    """Decode a binary vector into a structured solution dict.

    Returns
    -------
    dict with keys:
        association  : {ue_idx: node_idx}
        eta_level    : selected discrete level (-1 if none)
        eta          : worst-case AoII value (eta_level * level_step)
        single_assoc : True if every UE is assigned to exactly one node
        capacity_ok  : True if no node exceeds its capacity (needs *capacity*)
    """
    decoded = idx_map.decode_solution(x)
    eta_l = decoded["eta_level"]
    decoded["eta"] = eta_l * level_step if eta_l >= 0 else float("inf")

    # Check single-association
    ue_counts: Dict[int, int] = {}
    for (i, _m), flat in idx_map._assoc.items():
        if x[flat] > 0.5:
            ue_counts[i] = ue_counts.get(i, 0) + 1
    all_ues = {i for (i, _m) in idx_map._assoc}
    decoded["single_assoc"] = all(ue_counts.get(i, 0) == 1 for i in all_ues)

    # Check capacity
    if capacity is not None:
        node_load: Dict[int, int] = {}
        for (i, m), flat in idx_map._assoc.items():
            if x[flat] > 0.5:
                node_load[m] = node_load.get(m, 0) + 1
        decoded["capacity_ok"] = all(
            node_load.get(m, 0) <= int(capacity[m])
            for m in range(len(capacity))
        )
    else:
        decoded["capacity_ok"] = None

    return decoded
