import random


def fading_gain_db(rng: random.Random, enabled: bool = True) -> float:
    if not enabled:
        return 0.0
    # Lightweight placeholder: log-normal shadow/fading component in dB.
    return rng.gauss(0.0, 2.0)
