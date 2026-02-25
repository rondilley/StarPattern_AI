"""Wire protocol for distributed master/slave communication.

Framing: [4-byte big-endian length][gzip-compressed JSON]
Auth: HMAC-SHA256 of (auth_token, timestamp) with 300s tolerance.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import hmac
import json
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from star_pattern.utils.logging import get_logger

logger = get_logger("distributed.protocol")

PROTOCOL_VERSION = 1
MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100 MB
AUTH_TIMESTAMP_TOLERANCE = 300.0  # seconds


# --- Framing ---

async def send_message(
    writer: asyncio.StreamWriter, msg: dict[str, Any],
) -> None:
    """Send a length-prefixed gzip-compressed JSON message."""
    payload = gzip.compress(json.dumps(msg, default=str).encode("utf-8"))
    header = struct.pack(">I", len(payload))
    writer.write(header + payload)
    await writer.drain()


async def recv_message(
    reader: asyncio.StreamReader,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Receive a length-prefixed gzip-compressed JSON message.

    Raises:
        asyncio.TimeoutError: If timeout expires.
        ConnectionError: If the connection is closed or message is invalid.
    """
    async def _recv() -> dict[str, Any]:
        header = await reader.readexactly(4)
        (length,) = struct.unpack(">I", header)
        if length > MAX_MESSAGE_SIZE:
            raise ConnectionError(
                f"Message size {length} exceeds limit {MAX_MESSAGE_SIZE}"
            )
        payload = await reader.readexactly(length)
        data = gzip.decompress(payload)
        return json.loads(data)

    if timeout is not None:
        return await asyncio.wait_for(_recv(), timeout=timeout)
    return await _recv()


# --- Authentication ---

def make_auth(token: str, timestamp: float) -> str:
    """Create HMAC-SHA256 digest for authentication."""
    msg = f"{token}:{timestamp}".encode("utf-8")
    return hmac.new(token.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def verify_auth(token: str, timestamp: float, digest: str) -> bool:
    """Verify HMAC-SHA256 digest and timestamp freshness."""
    if not token and not digest:
        # No auth configured -- allow
        return True
    if abs(time.time() - timestamp) > AUTH_TIMESTAMP_TOLERANCE:
        logger.warning("Auth rejected: timestamp outside tolerance")
        return False
    expected = make_auth(token, timestamp)
    return hmac.compare_digest(expected, digest)


# --- Data classes ---

@dataclass
class WorkUnit:
    """A unit of work dispatched from master to slave."""

    work_id: str = ""
    region: dict[str, Any] = field(default_factory=dict)
    detection_config: dict[str, Any] = field(default_factory=dict)
    genome_dict: dict[str, Any] = field(default_factory=dict)
    data_config: dict[str, Any] = field(default_factory=dict)
    temporal_config: dict[str, Any] = field(default_factory=dict)
    include_temporal: bool = False
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.work_id:
            self.work_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "region": self.region,
            "detection_config": self.detection_config,
            "genome_dict": self.genome_dict,
            "data_config": self.data_config,
            "temporal_config": self.temporal_config,
            "include_temporal": self.include_temporal,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkUnit:
        return cls(
            work_id=d.get("work_id", ""),
            region=d.get("region", {}),
            detection_config=d.get("detection_config", {}),
            genome_dict=d.get("genome_dict", {}),
            data_config=d.get("data_config", {}),
            temporal_config=d.get("temporal_config", {}),
            include_temporal=d.get("include_temporal", False),
            priority=d.get("priority", 0),
        )


@dataclass
class WorkResult:
    """Result of a completed work unit from a slave."""

    work_id: str = ""
    slave_id: str = ""
    region: dict[str, Any] = field(default_factory=dict)
    pattern_results: list[dict[str, Any]] = field(default_factory=list)
    detection_summaries: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "slave_id": self.slave_id,
            "region": self.region,
            "pattern_results": self.pattern_results,
            "detection_summaries": self.detection_summaries,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkResult:
        return cls(
            work_id=d.get("work_id", ""),
            slave_id=d.get("slave_id", ""),
            region=d.get("region", {}),
            pattern_results=d.get("pattern_results", []),
            detection_summaries=d.get("detection_summaries", []),
            elapsed_seconds=d.get("elapsed_seconds", 0.0),
            error=d.get("error"),
        )
