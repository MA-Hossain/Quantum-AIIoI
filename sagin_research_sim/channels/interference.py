import math


def dbm_to_watt(x_dbm: float) -> float:
    return 10 ** ((x_dbm - 30.0) / 10.0)


def watt_to_dbm(x_watt: float) -> float:
    return 10.0 * math.log10(max(x_watt, 1e-30)) + 30.0


def aggregate_interference_watt(default_interference_dbm: float, enabled: bool = False) -> float:
    if not enabled:
        return 0.0
    return dbm_to_watt(default_interference_dbm)
