"""Synchronous bridge between AutonomousDiscovery and the async distributed layer.

Runs the asyncio event loop in a daemon thread, exposes a synchronous API
for the main pipeline thread.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import asdict
from typing import Any

from star_pattern.core.config import PipelineConfig
from star_pattern.core.sky_region import SkyRegion
from star_pattern.distributed.config import DistributedConfig
from star_pattern.distributed.master import MasterDispatcher
from star_pattern.distributed.protocol import WorkUnit, WorkResult
from star_pattern.evaluation.metrics import Anomaly, PatternResult
from star_pattern.utils.logging import get_logger

logger = get_logger("distributed.bridge")


def _result_dict_to_pattern_result(d: dict[str, Any]) -> PatternResult:
    """Reconstruct a PatternResult from its to_dict() output."""
    pr = PatternResult(
        region_ra=d.get("ra", 0.0),
        region_dec=d.get("dec", 0.0),
        detection_type=d.get("type", "unknown"),
        anomaly_score=d.get("anomaly_score", 0.0),
        significance=d.get("significance", 0.0),
        novelty=d.get("novelty", 0.0),
    )
    pr.hypothesis = d.get("hypothesis")
    pr.debate_verdict = d.get("debate_verdict")
    pr.consensus_score = d.get("consensus_score")
    pr.metadata = d.get("metadata", {})
    pr.cross_matches = d.get("cross_matches", [])

    # Reconstruct anomalies
    for a_dict in d.get("anomalies", []):
        pr.anomalies.append(Anomaly(
            anomaly_type=a_dict.get("anomaly_type", ""),
            detector=a_dict.get("detector", ""),
            pixel_x=a_dict.get("pixel_x"),
            pixel_y=a_dict.get("pixel_y"),
            sky_ra=a_dict.get("sky_ra"),
            sky_dec=a_dict.get("sky_dec"),
            score=a_dict.get("score", 0.0),
            properties=a_dict.get("properties", {}),
        ))
    return pr


class DistributedBridge:
    """Synchronous API for distributed work dispatch from the main pipeline thread.

    Internally manages the asyncio event loop in a daemon thread.
    Thread safety: all async operations are scheduled via
    asyncio.run_coroutine_threadsafe().
    """

    def __init__(self, config: DistributedConfig) -> None:
        self.config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._dispatcher: MasterDispatcher | None = None
        self._started = False

    def _run_loop(self) -> None:
        """Run the asyncio event loop in the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro: Any, timeout: float = 30.0) -> Any:
        """Schedule a coroutine on the event loop and wait for the result."""
        if self._loop is None or not self._started:
            raise RuntimeError("Bridge not started")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def start(self) -> int:
        """Start the background event loop and connect to all slaves.

        Returns the number of connected slaves.
        """
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="distributed-bridge",
        )
        self._thread.start()

        # Wait for the loop to be running
        while self._loop is None or not self._loop.is_running():
            pass

        self._dispatcher = MasterDispatcher(self.config)
        self._started = True

        n_connected = self._submit(
            self._dispatcher.connect_all(),
            timeout=self.config.connect_timeout * len(self.config.slave_addresses) + 10,
        )
        logger.info(f"Bridge started: {n_connected} slave(s) connected")
        return n_connected

    def submit_region(
        self,
        region: SkyRegion,
        config: PipelineConfig,
        genome_dict: dict[str, Any] | None = None,
        include_temporal: bool = False,
    ) -> str:
        """Submit a sky region for remote processing. Returns the work_id."""
        unit = WorkUnit(
            region={"ra": region.ra, "dec": region.dec, "radius": region.radius},
            detection_config=asdict(config.detection),
            genome_dict=genome_dict or {},
            data_config=asdict(config.data),
            temporal_config=asdict(config.temporal) if include_temporal else {},
            include_temporal=include_temporal,
        )

        dispatched = self._submit(
            self._dispatcher.dispatch(unit),
            timeout=10.0,
        )
        if not dispatched:
            raise RuntimeError("No slave available to accept work")

        return unit.work_id

    def collect_results(self, timeout: float = 30.0) -> list[PatternResult]:
        """Collect all available results from slaves.

        Returns a list of PatternResult objects reconstructed from
        the slave's serialized output.
        """
        work_results: list[WorkResult] = self._submit(
            self._dispatcher.collect_all_pending(timeout=timeout),
            timeout=timeout + 5.0,
        )

        pattern_results: list[PatternResult] = []
        for wr in work_results:
            if wr.error:
                logger.warning(
                    f"Work {wr.work_id[:8]} returned error: {wr.error}"
                )
                continue

            for pr_dict in wr.pattern_results:
                try:
                    pr = _result_dict_to_pattern_result(pr_dict)
                    pattern_results.append(pr)
                except Exception as e:
                    logger.warning(f"Failed to deserialize result: {e}")

            if wr.detection_summaries:
                logger.info(
                    f"Slave {wr.slave_id}: {len(wr.pattern_results)} results "
                    f"in {wr.elapsed_seconds:.1f}s"
                )

        return pattern_results

    def pending_count(self) -> int:
        """Number of work units still being processed by slaves."""
        if self._dispatcher is None:
            return 0
        return self._dispatcher.total_pending

    def push_config_update(
        self,
        detection_config: dict[str, Any],
        genome_dict: dict[str, Any],
    ) -> None:
        """Push updated detection config and genome to all slaves."""
        self._submit(
            self._dispatcher.update_config(detection_config, genome_dict),
            timeout=10.0,
        )

    def shutdown(self) -> None:
        """Shut down all slave connections and the background thread."""
        if not self._started:
            return

        try:
            self._submit(self._dispatcher.shutdown_all(), timeout=15.0)
        except Exception as e:
            logger.warning(f"Shutdown error: {e}")

        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        self._started = False
        logger.info("Bridge shut down")
