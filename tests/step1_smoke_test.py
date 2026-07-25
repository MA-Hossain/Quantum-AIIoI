"""Step 1 Smoke Test: Verify all modules import and run end-to-end."""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

lines = []

def log(msg=""):
    print(msg)
    lines.append(msg)


log("=" * 60)
log("STEP 1: CONFIG PACKAGE + IMPORT SMOKE TEST")
log("=" * 60)

# --- Part A: Import every module ---
log("\n--- Part A: Module Imports ---")
import_results = []

modules = [
    ("sagin_research_sim.config.radio_config",
     ["RadioConfig", "ResourceGridConfig", "ChannelModelConfig", "BandwidthMode", "DisruptionScenario"]),
    ("sagin_research_sim.config.topology_config",
     ["TopologyConfig", "NetworkMode"]),
    ("sagin_research_sim.config.simulation_config",
     ["SimulationConfig"]),
    ("sagin_research_sim.config.compute_config",
     ["EdgeComputeConfig"]),
    ("sagin_research_sim.channels.fading",
     ["fading_gain_db"]),
    ("sagin_research_sim.channels.interference",
     ["dbm_to_watt", "watt_to_dbm", "aggregate_interference_watt"]),
    ("sagin_research_sim.channels.pathloss",
     ["distance_3d_m", "elevation_angle_deg", "fspl_db", "path_loss_db"]),
    ("sagin_research_sim.channels.rate",
     ["compute_link_rate", "compute_all_rates", "LinkRateResult"]),
    ("sagin_research_sim.nodes.base",
     ["NetworkNode", "NodeType", "Position"]),
    ("sagin_research_sim.nodes.ue",
     ["UserEquipment"]),
    ("sagin_research_sim.nodes.bs",
     ["BaseStation"]),
    ("sagin_research_sim.nodes.satellite",
     ["Satellite"]),
    ("sagin_research_sim.nodes.uav",
     ["UAV"]),
    ("sagin_research_sim.nodes.topology",
     ["Topology", "generate_topology"]),
    ("sagin_research_sim.radio.bs_radio",
     ["make_bs_radio"]),
    ("sagin_research_sim.radio.satellite_radio",
     ["make_satellite_radio"]),
    ("sagin_research_sim.radio.wifi_radio",
     ["make_wifi_radio"]),
    ("sagin_research_sim.radio.mmwave_radio",
     ["make_mmwave_radio"]),
    ("sagin_research_sim.schedulers.resource_blocks",
     ["ResourceBlock", "ResourceGrid", "rb_bandwidth_hz", "allocation_bandwidth_hz"]),
    ("sagin_research_sim.schedulers.access_policies",
     ["SchedulingDecision", "random_access_equal_share"]),
]

all_passed = True
for mod_path, names in modules:
    try:
        mod = __import__(mod_path, fromlist=names)
        for name in names:
            getattr(mod, name)
        log(f"  [PASS] {mod_path} -> {', '.join(names)}")
    except Exception as e:
        log(f"  [FAIL] {mod_path} -> {e}")
        all_passed = False

log(f"\nImport result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

# --- Part B: Config class instantiation ---
log("\n--- Part B: Config Class Defaults ---")
from sagin_research_sim.config.radio_config import (
    RadioConfig, ResourceGridConfig, ChannelModelConfig, BandwidthMode, DisruptionScenario,
)
from sagin_research_sim.config.topology_config import TopologyConfig, NetworkMode
from sagin_research_sim.config.simulation_config import SimulationConfig
from sagin_research_sim.config.compute_config import EdgeComputeConfig

grid = ResourceGridConfig()
log(f"  ResourceGridConfig: {grid.num_freq_blocks} freq x {grid.num_time_slots} time, "
    f"SCS={grid.subcarrier_spacing_hz/1e3:.1f}kHz, rb_bw={grid.rb_bandwidth_hz/1e3:.1f}kHz")

radio = RadioConfig()
log(f"  RadioConfig: {radio.name}, {radio.carrier_freq_hz/1e9:.1f}GHz, "
    f"{radio.bandwidth_hz/1e6:.0f}MHz, {radio.tx_power_dbm}dBm")

ch_cfg = ChannelModelConfig()
log(f"  ChannelModelConfig: mode={ch_cfg.bandwidth_mode.value}, fading={ch_cfg.include_fading}, "
    f"scenarios={len(ch_cfg.disruption_scenarios)}")

topo_cfg = TopologyConfig()
log(f"  TopologyConfig: mode={topo_cfg.mode.value}, BS={topo_cfg.num_bs}, SAT={topo_cfg.num_satellites}, "
    f"UAV={topo_cfg.num_uavs}, area={topo_cfg.area_width_m}x{topo_cfg.area_height_m}m")

sim_cfg = SimulationConfig()
log(f"  SimulationConfig: seed={sim_cfg.seed}, num_ues={sim_cfg.num_ues}")

edge = EdgeComputeConfig()
log(f"  EdgeComputeConfig: BS={edge.bs_compute_capacity_cycles_per_s/1e9:.0f}G, "
    f"SAT={edge.satellite_compute_capacity_cycles_per_s/1e9:.0f}G, "
    f"UAV={edge.uav_compute_capacity_cycles_per_s/1e9:.0f}G cycles/s")

# --- Part C: End-to-end topology + scheduling + rate ---
log("\n--- Part C: End-to-End Test ---")
from sagin_research_sim.nodes.topology import generate_topology
from sagin_research_sim.schedulers.access_policies import random_access_equal_share
from sagin_research_sim.channels.rate import compute_scheduling_aware_rates

sim_cfg = SimulationConfig(seed=42, num_ues=5)
topo_cfg = TopologyConfig(num_bs=2, num_satellites=1)
channel_cfg = ChannelModelConfig(include_fading=False)

topo = generate_topology(sim_cfg, topo_cfg)
log(f"  Topology: {len(topo.ues)} UEs, {len(topo.base_stations)} BS, "
    f"{len(topo.satellites)} SAT, {len(topo.uavs)} UAV")

for node in topo.serving_nodes:
    log(f"    {node.name}: pos=({node.position.x_m:.0f}, {node.position.y_m:.0f}, {node.position.z_m:.0f})m, "
        f"radio={node.radio.name}, freq={node.radio.carrier_freq_hz/1e9:.1f}GHz")

decision = random_access_equal_share(topo, channel_cfg, topo_cfg, seed=0)
log(f"  Scheduling: {len(decision.association)} associations, policy={decision.policy_name}")

rates = compute_scheduling_aware_rates(topo, channel_cfg, topo_cfg, decision, seed=0)
log(f"  Link rates:")
for r in rates:
    log(f"    {r.ue_name} -> {r.node_name}: {r.rate_bps/1e6:.2f} Mbps "
        f"(dist={r.distance_m:.0f}m, SNR={r.snr_db:.1f}dB, visible={r.visible})")

log("\n" + "=" * 60)
log("STEP 1 COMPLETE")
log("=" * 60)

# Save output
output_path = os.path.join(OUTPUT_DIR, "step1_output.txt")
with open(output_path, "w") as f:
    f.write("\n".join(lines) + "\n")
log(f"\nOutput saved to: {output_path}")
