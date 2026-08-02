"""Distributed master/slave computing for the discovery pipeline."""

from star_pattern.distributed.config import DistributedConfig
from star_pattern.distributed.protocol import WorkResult, WorkUnit

__all__ = [
    "DistributedConfig",
    "WorkUnit",
    "WorkResult",
]
