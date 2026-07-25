from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sagin_research_sim.channels.fading import fading_gain_db
from sagin_research_sim.channels.interference import aggregate_interference_watt, dbm_to_watt
from sagin_research_sim.channels.pathloss import distance_3d_m, elevation_angle_deg, path_loss_db
from sagin_research_sim.config.radio_config import BandwidthMode, ChannelModelConfig, DisruptionScenario
from sagin_research_sim.config.topology_config import TopologyConfig
from sagin_research_sim.nodes.base import NetworkNode, NodeType
from sagin_research_sim.nodes.ue import UserEquipment
from sagin_research_sim.schedulers.resource_blocks import ResourceBlock, allocation_bandwidth_hz

# Preferred allocation key: (ue_uuid, serving_node_uuid).
RBAllocation = Dict[Tuple[str, str], List[ResourceBlock]]
ContinuousBandwidthAllocation = Dict[Tuple[str, str], float]
AssociationMap = Dict[str, str]  # ue_uuid -> serving_node_uuid


@dataclass
class LinkRateResult:
    ue_uuid: str
    ue_name: str
    node_uuid: str
    node_name: str
    node_type: str
    ue_local_id: int
    node_local_id: int
    bandwidth_hz: float
    distance_m: float
    path_loss_db: float
    snr_db: float
    rate_bps: float
    visible: bool
    num_resource_blocks: int = 0
    scenario_name: str = "nominal"
    disruption_loss_db: float = 0.0
    outage: bool = False
    outage_duration_slots: int = 0

    @property
    def ue_id(self) -> int:
        """Backward-compatible alias."""
        return self.ue_local_id

    @property
    def node_id(self) -> int:
        """Backward-compatible alias."""
        return self.node_local_id


def noise_power_watt(bandwidth_hz: float, noise_density_dbm_hz: float, noise_figure_db: float) -> float:
    noise_dbm = noise_density_dbm_hz + 10.0 * math.log10(max(bandwidth_hz, 1.0)) + noise_figure_db
    return dbm_to_watt(noise_dbm)


def shannon_rate_bps(bandwidth_hz: float, snr_linear: float, se_cap: float) -> float:
    spectral_efficiency = min(math.log2(1.0 + max(snr_linear, 0.0)), se_cap)
    return bandwidth_hz * spectral_efficiency


def _selected_bandwidth_hz(
    ue: UserEquipment,
    serving_node: NetworkNode,
    channel_cfg: ChannelModelConfig,
    rb_allocation: Optional[RBAllocation],
    bandwidth_allocation: Optional[ContinuousBandwidthAllocation] = None,
) -> Tuple[float, int]:
    mode = channel_cfg.effective_bandwidth_mode()
    if mode == BandwidthMode.RESOURCE_BLOCK:
        key = (ue.uuid, serving_node.uuid)
        rbs = [] if rb_allocation is None else rb_allocation.get(key, [])
        return allocation_bandwidth_hz(serving_node.radio.resource_grid, rbs), len(rbs)
    key = (ue.uuid, serving_node.uuid)
    if bandwidth_allocation is not None and key in bandwidth_allocation:
        return max(bandwidth_allocation[key], 0.0), 0
    return serving_node.radio.bandwidth_hz, 0


def _satellite_disruption_state(
    serving_node: NetworkNode,
    scenario: Optional[DisruptionScenario],
    rng: random.Random,
) -> Tuple[str, float, bool, int]:
    """Return disruption metadata for this link.

    Scenario losses/outages are applied only to satellite links. For terrestrial
    nodes, this function returns the nominal state even if a scenario is supplied.
    """
    if scenario is None or serving_node.node_type != NodeType.SATELLITE:
        return "nominal", 0.0, False, 0

    outage = rng.random() < scenario.outage_probability
    return scenario.name, scenario.total_extra_loss_db, outage, scenario.outage_duration_slots if outage else 0


def compute_link_rate(
    ue: UserEquipment,
    serving_node: NetworkNode,
    channel_cfg: ChannelModelConfig,
    topo_cfg: TopologyConfig,
    rb_allocation: Optional[RBAllocation] = None,
    bandwidth_allocation: Optional[ContinuousBandwidthAllocation] = None,
    scenario: Optional[DisruptionScenario] = None,
    rng: Optional[random.Random] = None,
) -> LinkRateResult:
    if rng is None:
        rng = random.Random(0)

    visible = True
    if serving_node.node_type == NodeType.SATELLITE:
        visible = elevation_angle_deg(ue, serving_node) >= topo_cfg.min_sat_elevation_deg

    scenario_name, disruption_loss_db, outage, outage_duration_slots = _satellite_disruption_state(
        serving_node, scenario, rng
    )
    if outage:
        visible = False

    bandwidth_hz, num_rbs = _selected_bandwidth_hz(
        ue,
        serving_node,
        channel_cfg,
        rb_allocation,
        bandwidth_allocation=bandwidth_allocation,
    )
    distance_m = distance_3d_m(ue, serving_node)

    if bandwidth_hz <= 0 or not visible:
        return LinkRateResult(
            ue_uuid=ue.uuid,
            ue_name=ue.name,
            node_uuid=serving_node.uuid,
            node_name=serving_node.name,
            node_type=serving_node.node_type.value,
            ue_local_id=ue.local_id,
            node_local_id=serving_node.local_id,
            bandwidth_hz=bandwidth_hz,
            distance_m=distance_m,
            path_loss_db=float("inf"),
            snr_db=-float("inf"),
            rate_bps=0.0,
            visible=visible,
            num_resource_blocks=num_rbs,
            scenario_name=scenario_name,
            disruption_loss_db=disruption_loss_db,
            outage=outage,
            outage_duration_slots=outage_duration_slots,
        )

    pl_db = path_loss_db(ue, serving_node, serving_node.radio.carrier_freq_hz)
    pl_db += disruption_loss_db
    pl_db += fading_gain_db(rng, channel_cfg.include_fading)

    rx_power_watt = dbm_to_watt(ue.tx_power_dbm - pl_db)
    noise_watt = noise_power_watt(
        bandwidth_hz,
        channel_cfg.thermal_noise_density_dbm_hz,
        serving_node.radio.noise_figure_db,
    )
    interf_watt = aggregate_interference_watt(
        channel_cfg.default_interference_dbm,
        channel_cfg.include_interference,
    )
    snr_linear = rx_power_watt / max(noise_watt + interf_watt, 1e-30)
    snr_db = 10.0 * math.log10(max(snr_linear, 1e-30))
    rate_bps = shannon_rate_bps(
        bandwidth_hz,
        snr_linear,
        serving_node.radio.max_spectral_efficiency_bps_hz,
    )

    return LinkRateResult(
        ue_uuid=ue.uuid,
        ue_name=ue.name,
        node_uuid=serving_node.uuid,
        node_name=serving_node.name,
        node_type=serving_node.node_type.value,
        ue_local_id=ue.local_id,
        node_local_id=serving_node.local_id,
        bandwidth_hz=bandwidth_hz,
        distance_m=distance_m,
        path_loss_db=pl_db,
        snr_db=snr_db,
        rate_bps=rate_bps,
        visible=visible,
        num_resource_blocks=num_rbs,
        scenario_name=scenario_name,
        disruption_loss_db=disruption_loss_db,
        outage=outage,
        outage_duration_slots=outage_duration_slots,
    )


def compute_all_rates(
    topology,
    channel_cfg: ChannelModelConfig,
    topo_cfg: TopologyConfig,
    rb_allocation: Optional[RBAllocation] = None,
    bandwidth_allocation: Optional[ContinuousBandwidthAllocation] = None,
    association: Optional[AssociationMap] = None,
    scheduled_only: bool = False,
    seed: int = 0,
    scenario: Optional[DisruptionScenario] = None,
) -> List[LinkRateResult]:
    rng = random.Random(seed)
    results: List[LinkRateResult] = []
    for ue in topology.ues:
        for node in topology.serving_nodes:
            if association is not None and association.get(ue.uuid) != node.uuid:
                if scheduled_only:
                    continue
                # Non-selected links receive zero scheduled bandwidth. This makes
                # candidate links explicit while preventing impossible rates.
                local_bw_allocation = dict(bandwidth_allocation or {})
                local_rb_allocation = dict(rb_allocation or {})
                local_bw_allocation.setdefault((ue.uuid, node.uuid), 0.0)
                local_rb_allocation.setdefault((ue.uuid, node.uuid), [])
            else:
                local_bw_allocation = bandwidth_allocation
                local_rb_allocation = rb_allocation

            results.append(
                compute_link_rate(
                    ue,
                    node,
                    channel_cfg,
                    topo_cfg,
                    rb_allocation=local_rb_allocation,
                    bandwidth_allocation=local_bw_allocation,
                    scenario=scenario,
                    rng=rng,
                )
            )
    return results


def compute_all_rates_for_scenarios(
    topology,
    channel_cfg: ChannelModelConfig,
    topo_cfg: TopologyConfig,
    rb_allocation: Optional[RBAllocation] = None,
    bandwidth_allocation: Optional[ContinuousBandwidthAllocation] = None,
    association: Optional[AssociationMap] = None,
    scheduled_only: bool = False,
    seed: int = 0,
) -> Dict[str, List[LinkRateResult]]:
    """Compute link rates for every configured disruption scenario."""
    output: Dict[str, List[LinkRateResult]] = {}
    for idx, scenario in enumerate(channel_cfg.disruption_scenarios):
        output[scenario.name] = compute_all_rates(
            topology,
            channel_cfg,
            topo_cfg,
            rb_allocation=rb_allocation,
            bandwidth_allocation=bandwidth_allocation,
            association=association,
            scheduled_only=scheduled_only,
            seed=seed + idx,
            scenario=scenario,
        )
    return output


def compute_scheduling_aware_rates(
    topology,
    channel_cfg: ChannelModelConfig,
    topo_cfg: TopologyConfig,
    scheduling_decision,
    seed: int = 0,
    scenario: Optional[DisruptionScenario] = None,
    scheduled_only: bool = True,
) -> List[LinkRateResult]:
    """Compute actual scheduled rates after association and resource allocation.

    scheduling_decision is intentionally duck-typed. It should provide:
        - association: Dict[ue_uuid, node_uuid]
        - continuous_bandwidth_allocation: Dict[(ue_uuid, node_uuid), bandwidth_hz]
        - rb_allocation: Dict[(ue_uuid, node_uuid), List[ResourceBlock]]
    
    Use this function for ML state/reward generation when the AP resource is
    shared among multiple associated UEs.
    """
    return compute_all_rates(
        topology,
        channel_cfg,
        topo_cfg,
        rb_allocation=getattr(scheduling_decision, "rb_allocation", None),
        bandwidth_allocation=getattr(scheduling_decision, "continuous_bandwidth_allocation", None),
        association=getattr(scheduling_decision, "association", None),
        scheduled_only=scheduled_only,
        seed=seed,
        scenario=scenario,
    )


def compute_scheduling_aware_rates_for_scenarios(
    topology,
    channel_cfg: ChannelModelConfig,
    topo_cfg: TopologyConfig,
    scheduling_decision,
    seed: int = 0,
    scheduled_only: bool = True,
) -> Dict[str, List[LinkRateResult]]:
    """Compute actual scheduled rates for every configured disruption scenario."""
    output: Dict[str, List[LinkRateResult]] = {}
    for idx, scenario in enumerate(channel_cfg.disruption_scenarios):
        output[scenario.name] = compute_scheduling_aware_rates(
            topology,
            channel_cfg,
            topo_cfg,
            scheduling_decision=scheduling_decision,
            seed=seed + idx,
            scenario=scenario,
            scheduled_only=scheduled_only,
        )
    return output
