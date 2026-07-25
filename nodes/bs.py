from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sagin_research_sim.config.radio_config import RadioConfig
from sagin_research_sim.nodes.base import NetworkNode, NodeType, Position
from sagin_research_sim.radio.bs_radio import make_bs_radio
from sagin_research_sim.config.compute_config import EdgeComputeConfig


@dataclass
class BaseStation(NetworkNode):
    def __init__(self, uuid: str, local_id: int, position: Position, radio: Optional[RadioConfig] = None):
        super().__init__(
            uuid=uuid,
            local_id=local_id,
            name=f"BS{local_id}",
            node_type=NodeType.BS,
            position=position,
            radio=radio or make_bs_radio(),
            compute_capacity_cycles_per_s=EdgeComputeConfig().bs_compute_capacity_cycles_per_s,
        )
