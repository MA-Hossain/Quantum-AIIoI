from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NetworkMode(str, Enum):
    SAGIN = "sagin"
    GROUND_ONLY = "ground_only"
    SATELLITE_ONLY = "satellite_only"


@dataclass
class TopologyConfig:
    mode: NetworkMode = NetworkMode.SAGIN
    num_bs: int = 2
    num_satellites: int = 2
    satellite_altitude_m: float = 550_000.0
    num_uavs: int = 0
    area_width_m: float = 1000.0
    area_height_m: float = 1000.0
    min_sat_elevation_deg: float = 10.0
