from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from sagin_research_sim.config.radio_config import RadioConfig


class NodeType(str, Enum):
    UE = "ue"
    BS = "bs"
    SATELLITE = "satellite"
    UAV = "uav"


@dataclass
class Position:
    x_m: float
    y_m: float
    z_m: float = 0.0
    lat_deg: float = 0.0
    lon_deg: float = 0.0
    alt_m: float = 0.0


@dataclass
class NetworkNode:
    """Base class for every node in the simulator.

    uuid:
        Globally unique identifier. Use this for dictionaries, topology graphs,
        optimization variables, and scheduler allocations. It avoids collisions
        such as BS0 and SAT0 both having local_id=0.
    local_id:
        Small integer ID only unique within one node type.
    name:
        Human-readable label such as UE0, BS1, SAT2, UAV0.
    """

    uuid: str
    local_id: int
    name: str
    node_type: NodeType
    position: Position
    radio: Optional[RadioConfig] = None
    compute_capacity_cycles_per_s: float = 0.0

    @property
    def node_id(self) -> int:
        """Backward-compatible alias for old code."""
        return self.local_id

    @property
    def display_name(self) -> str:
        return self.name
