"""Configuration for distributed master/slave computing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DistributedConfig:
    """Configuration for distributed pipeline execution.

    mode:
        "standalone" -- single-machine (default, no behavior change)
        "master"     -- dispatches work to slaves, processes locally every Nth cycle
        "slave"      -- listens for work from a master node
    """

    mode: str = "standalone"
    slave_addresses: list[str] = field(default_factory=list)
    listen_host: str = "0.0.0.0"
    listen_port: int = 7827
    auth_token: str = ""
    heartbeat_interval: float = 30.0
    work_timeout: float = 600.0
    max_retries: int = 2
    max_concurrent_per_slave: int = 2
    connect_timeout: float = 10.0
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10
    local_cycle_interval: int = 10

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DistributedConfig:
        """Create config from a dictionary."""
        return cls(
            mode=d.get("mode", "standalone"),
            slave_addresses=d.get("slave_addresses", []),
            listen_host=d.get("listen_host", "0.0.0.0"),
            listen_port=d.get("listen_port", 7827),
            auth_token=d.get("auth_token", ""),
            heartbeat_interval=d.get("heartbeat_interval", 30.0),
            work_timeout=d.get("work_timeout", 600.0),
            max_retries=d.get("max_retries", 2),
            max_concurrent_per_slave=d.get("max_concurrent_per_slave", 2),
            connect_timeout=d.get("connect_timeout", 10.0),
            reconnect_delay=d.get("reconnect_delay", 5.0),
            max_reconnect_attempts=d.get("max_reconnect_attempts", 10),
            local_cycle_interval=d.get("local_cycle_interval", 10),
        )
