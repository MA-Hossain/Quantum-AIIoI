from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sagin_research_sim.config.radio_config import RadioConfig
from sagin_research_sim.nodes.base import NetworkNode, NodeType, Position
from sagin_research_sim.radio.wifi_radio import make_wifi_radio
from sagin_research_sim.config.compute_config import EdgeComputeConfig


@dataclass
class UAV(NetworkNode):
    def __init__(self, uuid: str, local_id: int, position: Position, radio: Optional[RadioConfig] = None):
        super().__init__(
            uuid=uuid,
            local_id=local_id,
            name=f"UAV{local_id}",
            node_type=NodeType.UAV,
            position=position,
            radio=radio or make_wifi_radio(),
            compute_capacity_cycles_per_s=EdgeComputeConfig().uav_compute_capacity_cycles_per_s,
        )
