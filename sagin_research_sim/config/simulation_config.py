from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimulationConfig:
    seed: int = 42
    num_ues: int = 5
