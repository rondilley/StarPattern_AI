"""Master dispatcher: connects to slave nodes, dispatches work, collects results."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from star_pattern.distributed.config import DistributedConfig
from star_pattern.distributed.protocol import (
    PROTOCOL_VERSION,
    WorkUnit,
    WorkResult,
    make_auth,
    recv_message,
    send_message,
    verify_auth,
)
from star_pattern.utils.logging import get_logger

logger = get_logger("distributed.master")


class SlaveConnection:
    """Manages a TCP connection to a single slave node."""

    def __init__(
        self,
        address: str,
        auth_token: str,
        connect_timeout: float = 10.0,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 10,
    ) -> None:
        self.address = address
        self.auth_token = auth_token
        self.connect_timeout = connect_timeout
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        host_port = address.split(":")
        self.host = host_port[0]
        self.port = int(host_port[1]) if len(host_port) > 1 else 7827

        self.slave_id: str = ""
        self.n_workers: int = 1
        self.pending_count: int = 0
        self.connected: bool = False

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._recv_task: asyncio.Task | None = None
        self._result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._reconnect_count: int = 0

    async def connect(self) -> bool:
        """Establish connection and perform handshake."""
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.connect_timeout,
            )
        except (OSError, asyncio.TimeoutError) as e:
            logger.error(f"Failed to connect to {self.address}: {e}")
            return False

        # Send handshake
        ts = time.time()
        await send_message(self._writer, {
            "type": "handshake",
            "version": PROTOCOL_VERSION,
            "auth_digest": make_auth(self.auth_token, ts),
            "timestamp": ts,
        })

        # Wait for ack
        try:
            ack = await recv_message(self._reader, timeout=10.0)
        except (asyncio.TimeoutError, ConnectionError) as e:
            logger.error(f"Handshake failed with {self.address}: {e}")
            self._close()
            return False

        if ack.get("type") != "handshake_ack":
            logger.error(f"Unexpected handshake response from {self.address}")
            self._close()
            return False

        self.slave_id = ack.get("slave_id", self.address)
        self.n_workers = ack.get("n_workers", 1)
        self.connected = True
        self._reconnect_count = 0

        # Start receive loop
        self._recv_task = asyncio.create_task(self._recv_loop())

        logger.info(
            f"Connected to slave {self.slave_id} "
            f"({self.address}, {self.n_workers} workers)"
        )
        return True

    async def _recv_loop(self) -> None:
        """Continuously receive messages from the slave."""
        while self.connected and self._reader is not None:
            try:
                msg = await recv_message(self._reader)
                msg_type = msg.get("type")

                if msg_type in ("work_result", "work_error"):
                    self.pending_count = max(0, self.pending_count - 1)
                    await self._result_queue.put(msg)

                elif msg_type == "heartbeat":
                    pass  # Heartbeat response, no action needed

                else:
                    logger.debug(f"Slave {self.slave_id}: unknown msg {msg_type}")

            except (asyncio.IncompleteReadError, ConnectionError):
                logger.warning(f"Connection lost to {self.slave_id}")
                self.connected = False
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Recv error from {self.slave_id}: {e}")
                self.connected = False
                break

    async def send_work(self, unit: WorkUnit) -> None:
        """Send a work unit to this slave."""
        if not self.connected or self._writer is None:
            raise ConnectionError(f"Not connected to {self.address}")
        await send_message(self._writer, {
            "type": "work_dispatch",
            "payload": unit.to_dict(),
        })
        self.pending_count += 1

    async def send_config_update(
        self, detection_config: dict[str, Any], genome_dict: dict[str, Any],
    ) -> None:
        """Push updated detection config and genome to slave."""
        if not self.connected or self._writer is None:
            return
        await send_message(self._writer, {
            "type": "config_update",
            "payload": {
                "detection_config": detection_config,
                "genome_dict": genome_dict,
            },
        })

    async def send_heartbeat(self) -> None:
        """Send a heartbeat ping."""
        if not self.connected or self._writer is None:
            return
        await send_message(self._writer, {
            "type": "heartbeat",
            "timestamp": time.time(),
        })

    async def send_shutdown(self) -> None:
        """Tell the slave to shut down."""
        if self.connected and self._writer is not None:
            try:
                await send_message(self._writer, {"type": "shutdown"})
            except Exception:
                pass

    async def reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff."""
        self._close()
        while self._reconnect_count < self.max_reconnect_attempts:
            self._reconnect_count += 1
            delay = self.reconnect_delay * (2 ** (self._reconnect_count - 1))
            delay = min(delay, 60.0)
            logger.info(
                f"Reconnecting to {self.address} in {delay:.1f}s "
                f"(attempt {self._reconnect_count}/{self.max_reconnect_attempts})"
            )
            await asyncio.sleep(delay)
            if await self.connect():
                return True
        logger.error(
            f"Giving up on {self.address} after "
            f"{self.max_reconnect_attempts} attempts"
        )
        return False

    def _close(self) -> None:
        """Close the connection."""
        self.connected = False
        if self._recv_task is not None:
            self._recv_task.cancel()
            self._recv_task = None
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        self._reader = None

    async def close(self) -> None:
        """Gracefully close the connection."""
        self._close()


class MasterDispatcher:
    """Manages connections to multiple slave nodes and dispatches work."""

    def __init__(self, config: DistributedConfig) -> None:
        self.config = config
        self._slaves: list[SlaveConnection] = []
        self._result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending_work: dict[str, WorkUnit] = {}
        self._collector_tasks: list[asyncio.Task] = []

    async def connect_all(self) -> int:
        """Connect to all configured slave nodes. Returns number of successes."""
        tasks = []
        for addr in self.config.slave_addresses:
            conn = SlaveConnection(
                address=addr,
                auth_token=self.config.auth_token,
                connect_timeout=self.config.connect_timeout,
                reconnect_delay=self.config.reconnect_delay,
                max_reconnect_attempts=self.config.max_reconnect_attempts,
            )
            self._slaves.append(conn)
            tasks.append(conn.connect())

        results = await asyncio.gather(*tasks, return_exceptions=True)
        n_connected = sum(
            1 for r in results if r is True
        )

        # Start result collector for each connected slave
        for slave in self._slaves:
            if slave.connected:
                task = asyncio.create_task(self._collect_from(slave))
                self._collector_tasks.append(task)

        logger.info(
            f"Connected to {n_connected}/{len(self._slaves)} slaves"
        )
        return n_connected

    async def _collect_from(self, slave: SlaveConnection) -> None:
        """Forward results from a slave's queue to the central queue."""
        while True:
            try:
                msg = await slave._result_queue.get()
                await self._result_queue.put(msg)
            except asyncio.CancelledError:
                break

    def _least_loaded_slave(self) -> SlaveConnection | None:
        """Return the connected slave with the fewest pending work units."""
        candidates = [
            s for s in self._slaves
            if s.connected
            and s.pending_count < self.config.max_concurrent_per_slave
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.pending_count)

    async def dispatch(self, unit: WorkUnit) -> bool:
        """Dispatch a work unit to the least-loaded slave.

        Returns True if dispatched, False if no slave available.
        """
        slave = self._least_loaded_slave()
        if slave is None:
            logger.warning("No available slaves for dispatch")
            return False

        try:
            await slave.send_work(unit)
            self._pending_work[unit.work_id] = unit
            logger.debug(
                f"Dispatched {unit.work_id[:8]} to {slave.slave_id}"
            )
            return True
        except ConnectionError:
            logger.warning(f"Dispatch failed to {slave.slave_id}")
            return False

    async def collect_result(
        self, timeout: float = 30.0,
    ) -> WorkResult | None:
        """Collect one result from any slave. Returns None on timeout."""
        try:
            msg = await asyncio.wait_for(
                self._result_queue.get(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            return None

        msg_type = msg.get("type")
        payload = msg.get("payload", {})

        if msg_type == "work_result":
            result = WorkResult.from_dict(payload)
            self._pending_work.pop(result.work_id, None)
            return result

        elif msg_type == "work_error":
            work_id = payload.get("work_id", "")
            logger.error(
                f"Slave error for {work_id[:8]}: {payload.get('error')}"
            )
            # Retry logic
            unit = self._pending_work.pop(work_id, None)
            if unit is not None and unit.priority < self.config.max_retries:
                unit.priority += 1
                await self.dispatch(unit)
                logger.info(f"Retrying {work_id[:8]} (attempt {unit.priority})")
            return None

        return None

    async def collect_all_pending(
        self, timeout: float = 30.0,
    ) -> list[WorkResult]:
        """Collect all available results within timeout."""
        results: list[WorkResult] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            result = await self.collect_result(timeout=min(remaining, 1.0))
            if result is not None:
                results.append(result)
            elif not self._pending_work:
                break
        return results

    async def update_config(
        self,
        detection_config: dict[str, Any],
        genome_dict: dict[str, Any],
    ) -> None:
        """Push updated config to all connected slaves."""
        tasks = []
        for slave in self._slaves:
            if slave.connected:
                tasks.append(slave.send_config_update(detection_config, genome_dict))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"Config update pushed to {len(tasks)} slave(s)")

    @property
    def total_pending(self) -> int:
        """Total pending work units across all slaves."""
        return sum(s.pending_count for s in self._slaves if s.connected)

    @property
    def n_connected(self) -> int:
        """Number of currently connected slaves."""
        return sum(1 for s in self._slaves if s.connected)

    async def shutdown_all(self) -> None:
        """Send shutdown to all slaves and close connections."""
        for slave in self._slaves:
            await slave.send_shutdown()

        for task in self._collector_tasks:
            task.cancel()
        if self._collector_tasks:
            await asyncio.gather(*self._collector_tasks, return_exceptions=True)

        for slave in self._slaves:
            await slave.close()

        self._slaves.clear()
        self._collector_tasks.clear()
        logger.info("All slave connections closed")
