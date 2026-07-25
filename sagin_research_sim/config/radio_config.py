from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class BandwidthMode(str, Enum):
    RESOURCE_BLOCK = "resource_block"
    CONTINUOUS = "continuous"


@dataclass
class ResourceGridConfig:
    num_freq_blocks: int = 100
    num_time_slots: int = 10
    subcarrier_spacing_hz: float = 15e3
    subcarriers_per_rb: int = 12

    @property
    def rb_bandwidth_hz(self) -> float:
        return self.subcarrier_spacing_hz * self.subcarriers_per_rb


@dataclass
class RadioConfig:
    name: str = "default"
    carrier_freq_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    tx_power_dbm: float = 23.0
    noise_figure_db: float = 7.0
    max_spectral_efficiency_bps_hz: float = 6.0
    resource_grid: Optional[ResourceGridConfig] = None


@dataclass
class DisruptionScenario:
    name: str = "nominal"
    total_extra_loss_db: float = 0.0
    outage_probability: float = 0.0
    outage_duration_slots: int = 0


@dataclass
class ChannelModelConfig:
    bandwidth_mode: BandwidthMode = BandwidthMode.RESOURCE_BLOCK
    include_fading: bool = True
    include_interference: bool = False
    default_interference_dbm: float = -100.0
    thermal_noise_density_dbm_hz: float = -174.0
    disruption_scenarios: List[DisruptionScenario] = field(default_factory=lambda: [DisruptionScenario()])

    def effective_bandwidth_mode(self) -> BandwidthMode:
        return self.bandwidth_mode
