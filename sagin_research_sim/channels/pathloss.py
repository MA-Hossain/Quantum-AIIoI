import math
from sagin_research_sim.nodes.base import NetworkNode, NodeType

C = 299_792_458.0


def distance_3d_m(a: NetworkNode, b: NetworkNode) -> float:
    dx = a.position.x_m - b.position.x_m
    dy = a.position.y_m - b.position.y_m
    dz = a.position.alt_m - b.position.alt_m
    return max(math.sqrt(dx * dx + dy * dy + dz * dz), 1.0)


def elevation_angle_deg(ue: NetworkNode, sat: NetworkNode) -> float:
    horizontal = math.sqrt((ue.position.x_m - sat.position.x_m) ** 2 + (ue.position.y_m - sat.position.y_m) ** 2)
    vertical = sat.position.alt_m - ue.position.alt_m
    return math.degrees(math.atan2(vertical, max(horizontal, 1.0)))


def fspl_db(distance_m: float, carrier_freq_hz: float) -> float:
    return 20.0 * math.log10(4.0 * math.pi * distance_m * carrier_freq_hz / C)


def path_loss_db(tx: NetworkNode, rx: NetworkNode, carrier_freq_hz: float) -> float:
    d = distance_3d_m(tx, rx)
    pl = fspl_db(d, carrier_freq_hz)
    if rx.node_type == NodeType.BS:
        pl += 5.0   # simple terrestrial margin
    elif rx.node_type == NodeType.SATELLITE:
        pl += 2.0   # simple satellite implementation margin
    return pl
