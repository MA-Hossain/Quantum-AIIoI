from __future__ import annotations

from dataclasses import dataclass

from sagin_research_sim.nodes.base import NetworkNode, NodeType, Position


@dataclass
class UserEquipment(NetworkNode):
    tx_power_dbm: float = 23.0

    def __init__(self, uuid: str, local_id: int, position: Position, tx_power_dbm: float = 23.0):
        super().__init__(
            uuid=uuid,
            local_id=local_id,
            name=f"UE{local_id}",
            node_type=NodeType.UE,
            position=position,
            radio=None,
        )
        self.tx_power_dbm = tx_power_dbm
