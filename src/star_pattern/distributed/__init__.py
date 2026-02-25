"""Distributed master/slave computing for the discovery pipeline."""

from star_pattern.distributed.config import DistributedConfig
from star_pattern.distributed.protocol import WorkUnit, WorkResult

__all__ = [
    "DistributedConfig",
    "WorkUnit",
    "WorkResult",
]
