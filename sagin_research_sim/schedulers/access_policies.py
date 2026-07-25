from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from sagin_research_sim.channels.pathloss import elevation_angle_deg
from sagin_research_sim.config.radio_config import BandwidthMode, ChannelModelConfig
from sagin_research_sim.config.topology_config import TopologyConfig
from sagin_research_sim.nodes.base import NetworkNode, NodeType
from sagin_research_sim.nodes.ue import UserEquipment
from sagin_research_sim.schedulers.resource_blocks import ResourceBlock, ResourceGrid

# Common allocation key: (ue_uuid, serving_node_uuid).
AllocationKey = Tuple[str, str]
ContinuousBandwidthAllocation = Dict[AllocationKey, float]
ComputeAllocation = Dict[AllocationKey, float]  # cycles/s allocated to UE at serving node
RBAllocation = Dict[AllocationKey, List[ResourceBlock]]
AssociationMap = Dict[str, str]  # ue_uuid -> serving_node_uuid
UpdateGenerationMap = Dict[str, bool]  # ue_uuid -> whether UE samples/transmits this slot


@dataclass
class SchedulingDecision:
    """Output of a scheduling/access policy.

    association maps each UE UUID to one selected serving-node UUID.

    update_generation controls whether a UE actually sends a fresh status update
    in this slot. A UE can be associated but inactive. Inactive UEs consume no
    bandwidth/RBs in the helper policies below and cannot deliver an update.

    Depending on the bandwidth model, either continuous_bandwidth_allocation or
    rb_allocation is populated. The same object can be fed directly into the rate
    calculator and the AoII environment.
    """

    association: AssociationMap = field(default_factory=dict)
    update_generation: UpdateGenerationMap = field(default_factory=dict)
    continuous_bandwidth_allocation: ContinuousBandwidthAllocation = field(default_factory=dict)
    rb_allocation: RBAllocation = field(default_factory=dict)
    compute_allocation_cycles_per_s: ComputeAllocation = field(default_factory=dict)
    policy_name: str = ""

    def selected_node_uuid(self, ue_uuid: str) -> Optional[str]:
        return self.association.get(ue_uuid)

    def is_active(self, ue_uuid: str) -> bool:
        # Backward compatible: if a decision does not explicitly specify update
        # generation, all associated UEs are treated as active.
        if not self.update_generation:
            return ue_uuid in self.association
        return bool(self.update_generation.get(ue_uuid, False))


def is_visible(ue: UserEquipment, node: NetworkNode, topo_cfg: TopologyConfig) -> bool:
    """Return whether a serving node is usable by a UE before stochastic outage."""
    if node.node_type == NodeType.SATELLITE:
        return elevation_angle_deg(ue, node) >= topo_cfg.min_sat_elevation_deg
    return True


def equal_share_resources(
    topology,
    channel_cfg: ChannelModelConfig,
    decision: SchedulingDecision,
) -> SchedulingDecision:
    """Populate equal continuous-bandwidth or 2D-RB allocation for active UEs.

    This is the key scheduling-aware helper. It makes the actual rate depend on
    how many UEs are active on the same AP/SAT/BS.
    """
    node_by_uuid = {node.uuid: node for node in topology.serving_nodes}
    ue_by_uuid = {ue.uuid: ue for ue in topology.ues}
    users_by_node: Dict[str, List[UserEquipment]] = {node.uuid: [] for node in topology.serving_nodes}

    for ue_uuid, node_uuid in decision.association.items():
        if decision.is_active(ue_uuid) and node_uuid in users_by_node and ue_uuid in ue_by_uuid:
            users_by_node[node_uuid].append(ue_by_uuid[ue_uuid])

    decision.continuous_bandwidth_allocation.clear()
    decision.rb_allocation.clear()
    decision.compute_allocation_cycles_per_s.clear()

    if channel_cfg.effective_bandwidth_mode() == BandwidthMode.RESOURCE_BLOCK:
        for node_uuid, users in users_by_node.items():
            if not users:
                continue
            node = node_by_uuid[node_uuid]
            grid = ResourceGrid(node.radio.resource_grid)
            per_ue = grid.equal_allocate([ue.uuid for ue in users])
            for ue in users:
                decision.rb_allocation[(ue.uuid, node.uuid)] = per_ue[ue.uuid]
                decision.compute_allocation_cycles_per_s[(ue.uuid, node.uuid)] = node.compute_capacity_cycles_per_s / len(users)
    else:
        for node_uuid, users in users_by_node.items():
            if not users:
                continue
            node = node_by_uuid[node_uuid]
            bw_per_ue = node.radio.bandwidth_hz / len(users)
            for ue in users:
                decision.continuous_bandwidth_allocation[(ue.uuid, node.uuid)] = bw_per_ue
                decision.compute_allocation_cycles_per_s[(ue.uuid, node.uuid)] = node.compute_capacity_cycles_per_s / len(users)
    return decision


def build_equal_share_decision(
    topology,
    channel_cfg: ChannelModelConfig,
    association: AssociationMap,
    active_ue_uuids: Optional[Iterable[str]] = None,
    policy_name: str = "equal_share",
) -> SchedulingDecision:
    active_set = set(association.keys()) if active_ue_uuids is None else set(active_ue_uuids)
    decision = SchedulingDecision(policy_name=policy_name)
    decision.association.update(association)
    for ue_uuid in association:
        decision.update_generation[ue_uuid] = ue_uuid in active_set
    return equal_share_resources(topology, channel_cfg, decision)


def random_access_equal_share(
    topology,
    channel_cfg: ChannelModelConfig,
    topo_cfg: TopologyConfig,
    seed: int = 0,
    active_probability: float = 1.0,
) -> SchedulingDecision:
    """Random access + equal resource sharing baseline.

    Step 1: each UE randomly decides whether to generate an update.
    Step 2: active UEs randomly select one currently visible serving node.
    Step 3: each serving node equally shares resources among its active UEs.
    """
    rng = random.Random(seed)
    association: AssociationMap = {}
    active: List[str] = []

    for ue in topology.ues:
        candidates = [node for node in topology.serving_nodes if is_visible(ue, node, topo_cfg)]
        if not candidates:
            continue
        selected = rng.choice(candidates)
        association[ue.uuid] = selected.uuid
        if rng.random() < active_probability:
            active.append(ue.uuid)

    return build_equal_share_decision(
        topology=topology,
        channel_cfg=channel_cfg,
        association=association,
        active_ue_uuids=active,
        policy_name="random_access_equal_share",
    )
