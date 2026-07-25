from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List

from sagin_research_sim.config.radio_config import ResourceGridConfig


@dataclass(frozen=True)
class ResourceBlock:
    """One 2D radio resource: one frequency block in one time slot."""

    freq_idx: int
    time_idx: int


class ResourceGrid:
    def __init__(self, config: ResourceGridConfig):
        self.config = config
        self.blocks = [
            ResourceBlock(f, t)
            for t in range(config.num_time_slots)
            for f in range(config.num_freq_blocks)
        ]

    def all_blocks(self) -> List[ResourceBlock]:
        return list(self.blocks)

    def equal_allocate(self, user_keys: Iterable[Hashable]) -> Dict[Hashable, List[ResourceBlock]]:
        """Allocate all RBs round-robin to arbitrary keys, e.g., UE UUIDs."""
        user_keys = list(user_keys)
        allocation = {u: [] for u in user_keys}
        if not user_keys:
            return allocation
        for idx, rb in enumerate(self.blocks):
            allocation[user_keys[idx % len(user_keys)]].append(rb)
        return allocation


def rb_bandwidth_hz(grid_cfg: ResourceGridConfig) -> float:
    return grid_cfg.rb_bandwidth_hz


def allocation_bandwidth_hz(grid_cfg: ResourceGridConfig, allocated_rbs: List[ResourceBlock]) -> float:
    # Scheduling often averages over a frame. Here we convert total 2D RBs into
    # equivalent per-slot frequency bandwidth: (#RBs / #time_slots) * RB bandwidth.
    equivalent_freq_rbs = len(allocated_rbs) / max(grid_cfg.num_time_slots, 1)
    return equivalent_freq_rbs * grid_cfg.rb_bandwidth_hz
