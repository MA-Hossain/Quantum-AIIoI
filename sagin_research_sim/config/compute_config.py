from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EdgeComputeConfig:
    bs_compute_capacity_cycles_per_s: float = 10e9
    satellite_compute_capacity_cycles_per_s: float = 2e9
    uav_compute_capacity_cycles_per_s: float = 5e9
