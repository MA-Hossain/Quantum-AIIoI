from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import List

from sagin_research_sim.config.simulation_config import SimulationConfig
from sagin_research_sim.config.topology_config import NetworkMode, TopologyConfig
from sagin_research_sim.nodes.base import NetworkNode, NodeType, Position
from sagin_research_sim.nodes.bs import BaseStation
from sagin_research_sim.nodes.satellite import Satellite
from sagin_research_sim.nodes.uav import UAV
from sagin_research_sim.nodes.ue import UserEquipment


@dataclass
class Topology:
    ues: List[UserEquipment]
    serving_nodes: List[NetworkNode]
    base_stations: List[BaseStation]
    satellites: List[Satellite]
    uavs: List[UAV]


def make_node_uuid(seed: int, node_type: NodeType, local_id: int) -> str:
    """Return a deterministic globally unique UUID string for a node.

    It is deterministic for reproducible simulations, but it is globally unique
    across node types, so BS0 and SAT0 will not collide.
    """
    key = f"sagin-sim:{seed}:{node_type.value}:{local_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def _random_position(rng: random.Random, topo_cfg: TopologyConfig, altitude_m: float = 0.0) -> Position:
    x = rng.uniform(0, topo_cfg.area_width_m)
    y = rng.uniform(0, topo_cfg.area_height_m)
    return Position(x_m=x, y_m=y, z_m=altitude_m, alt_m=altitude_m)


def generate_topology(sim_cfg: SimulationConfig, topo_cfg: TopologyConfig) -> Topology:
    rng = random.Random(sim_cfg.seed)

    ues = [
        UserEquipment(
            uuid=make_node_uuid(sim_cfg.seed, NodeType.UE, i),
            local_id=i,
            position=_random_position(rng, topo_cfg),
            tx_power_dbm=23.0,
        )
        for i in range(sim_cfg.num_ues)
    ]

    base_stations: List[BaseStation] = []
    satellites: List[Satellite] = []
    uavs: List[UAV] = []

    if topo_cfg.mode in (NetworkMode.SAGIN, NetworkMode.GROUND_ONLY):
        for k in range(topo_cfg.num_bs):
            pos = _random_position(rng, topo_cfg, altitude_m=25.0)
            base_stations.append(
                BaseStation(
                    uuid=make_node_uuid(sim_cfg.seed, NodeType.BS, k),
                    local_id=k,
                    position=pos,
                )
            )

    if topo_cfg.mode in (NetworkMode.SAGIN, NetworkMode.SATELLITE_ONLY):
        for j in range(topo_cfg.num_satellites):
            pos = _random_position(rng, topo_cfg, altitude_m=topo_cfg.satellite_altitude_m)
            satellites.append(
                Satellite(
                    uuid=make_node_uuid(sim_cfg.seed, NodeType.SATELLITE, j),
                    local_id=j,
                    position=pos,
                )
            )

    for a in range(topo_cfg.num_uavs):
        pos = _random_position(rng, topo_cfg, altitude_m=100.0)
        uavs.append(
            UAV(
                uuid=make_node_uuid(sim_cfg.seed, NodeType.UAV, a),
                local_id=a,
                position=pos,
            )
        )

    serving_nodes: List[NetworkNode] = []
    serving_nodes.extend(base_stations)
    serving_nodes.extend(satellites)
    serving_nodes.extend(uavs)

    return Topology(
        ues=ues,
        serving_nodes=serving_nodes,
        base_stations=base_stations,
        satellites=satellites,
        uavs=uavs,
    )
