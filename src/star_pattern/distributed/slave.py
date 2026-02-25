"""Slave worker node: listens for work from a master and executes detection."""

from __future__ import annotations

import asyncio
import platform
import os
import time
import traceback
from dataclasses import asdict
from typing import Any

from star_pattern.core.config import PipelineConfig, DetectionConfig, DataConfig
from star_pattern.core.sky_region import SkyRegion
from star_pattern.data.pipeline import DataPipeline
from star_pattern.detection.ensemble import EnsembleDetector
from star_pattern.detection.local_classifier import LocalClassifier
from star_pattern.detection.local_evaluator import LocalEvaluator
from star_pattern.distributed.protocol import (
    PROTOCOL_VERSION,
    WorkUnit,
    WorkResult,
    make_auth,
    recv_message,
    send_message,
    verify_auth,
)
from star_pattern.evaluation.metrics import PatternResult
from star_pattern.pipeline.autonomous import _extract_anomalies
from star_pattern.utils.logging import get_logger

logger = get_logger("distributed.slave")


class SlaveServer:
    """TCP server that receives work units and executes detection pipelines."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.slave_id = f"{platform.node()}:{os.getpid()}"

        # Core pipeline components
        self.data_pipeline = DataPipeline(config.data)
        self.local_classifier = LocalClassifier()
        self.local_evaluator = LocalEvaluator()

        # Detector is rebuilt when config arrives from master
        self._detector: EnsembleDetector | None = None
        self._detection_config_hash: str = ""

        self._server: asyncio.Server | None = None
        self._active_tasks: set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

    def _ensure_detector(self, detection_config: dict[str, Any]) -> None:
        """Create or rebuild EnsembleDetector if config changed."""
        config_repr = str(sorted(detection_config.items()))
        if config_repr != self._detection_config_hash:
            det_config = DetectionConfig(**detection_config)
            self._detector = EnsembleDetector(det_config)
            self._detection_config_hash = config_repr
            logger.info("Rebuilt EnsembleDetector with updated config")

    async def start(self) -> None:
        """Start the slave TCP server and block until shutdown."""
        dist = self.config.distributed
        self._server = await asyncio.start_server(
            self._handle_connection,
            host=dist.listen_host,
            port=dist.listen_port,
        )
        addrs = [s.getsockname() for s in self._server.sockets]
        logger.info(f"Slave {self.slave_id} listening on {addrs}")

        async with self._server:
            await self._shutdown_event.wait()

        # Cancel any remaining work
        for task in self._active_tasks:
            task.cancel()
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)

        logger.info("Slave server shut down")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single master connection."""
        peer = writer.get_extra_info("peername")
        logger.info(f"Connection from {peer}")

        try:
            # Handshake
            msg = await recv_message(reader, timeout=30.0)
            if msg.get("type") != "handshake":
                logger.warning(f"Expected handshake, got {msg.get('type')}")
                writer.close()
                return

            if not verify_auth(
                self.config.distributed.auth_token,
                msg.get("timestamp", 0),
                msg.get("auth_digest", ""),
            ):
                logger.warning(f"Auth failed from {peer}")
                writer.close()
                return

            # Send ack
            n_workers = self.config.distributed.max_concurrent_per_slave
            await send_message(writer, {
                "type": "handshake_ack",
                "slave_id": self.slave_id,
                "n_workers": n_workers,
            })
            logger.info(f"Handshake complete with {peer}, workers={n_workers}")

            # Message loop
            while not self._shutdown_event.is_set():
                try:
                    msg = await recv_message(reader, timeout=None)
                except (asyncio.IncompleteReadError, ConnectionError):
                    logger.info(f"Connection closed by {peer}")
                    break

                msg_type = msg.get("type")

                if msg_type == "work_dispatch":
                    unit = WorkUnit.from_dict(msg.get("payload", {}))
                    task = asyncio.create_task(
                        self._execute_and_send(unit, writer)
                    )
                    self._active_tasks.add(task)
                    task.add_done_callback(self._active_tasks.discard)

                elif msg_type == "config_update":
                    payload = msg.get("payload", {})
                    det_config = payload.get("detection_config", {})
                    if det_config:
                        self._ensure_detector(det_config)

                elif msg_type == "heartbeat":
                    await send_message(writer, {
                        "type": "heartbeat",
                        "timestamp": time.time(),
                    })

                elif msg_type == "shutdown":
                    logger.info("Shutdown command received")
                    self._shutdown_event.set()
                    break

                else:
                    logger.warning(f"Unknown message type: {msg_type}")

        except Exception as e:
            logger.error(f"Connection handler error: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _execute_and_send(
        self,
        unit: WorkUnit,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Execute a work unit and send the result back."""
        start = time.time()
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._execute_work, unit,
            )
            result.elapsed_seconds = time.time() - start
            await send_message(writer, {
                "type": "work_result",
                "payload": result.to_dict(),
            })
        except Exception as e:
            logger.error(f"Work {unit.work_id} failed: {e}")
            await send_message(writer, {
                "type": "work_error",
                "payload": {
                    "work_id": unit.work_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            })

    def _execute_work(self, unit: WorkUnit) -> WorkResult:
        """Execute a work unit synchronously (runs in thread pool)."""
        logger.info(f"Executing work {unit.work_id[:8]}...")

        # Ensure detector is up to date
        if unit.detection_config:
            self._ensure_detector(unit.detection_config)

        if self._detector is None:
            # Build with default config if nothing received yet
            self._detector = EnsembleDetector(self.config.detection)

        # Build region
        region = SkyRegion(
            ra=unit.region.get("ra", 0.0),
            dec=unit.region.get("dec", 0.0),
            radius=unit.region.get("radius", 3.0),
        )

        # Build temporal config
        temporal_config = None
        if unit.include_temporal and unit.temporal_config:
            from star_pattern.core.config import TemporalConfig
            temporal_config = TemporalConfig(
                max_epochs=unit.temporal_config.get("max_epochs", 10),
                min_baseline_days=unit.temporal_config.get("min_baseline_days", 1.0),
                max_baseline_days=unit.temporal_config.get("max_baseline_days", 2000.0),
                snr_threshold=unit.temporal_config.get("snr_threshold", 5.0),
                min_epochs=unit.temporal_config.get("min_epochs", 2),
            )

        # Fetch data (slave has its own cache)
        region_data = self.data_pipeline.fetch_region(
            region,
            include_temporal=unit.include_temporal,
            temporal_config=temporal_config,
        )

        if not region_data.has_images():
            return WorkResult(
                work_id=unit.work_id,
                slave_id=self.slave_id,
                region=unit.region,
                pattern_results=[],
                detection_summaries=[],
            )

        # Merge catalogs for catalog-based detectors
        merged_catalog = None
        if region_data.catalogs:
            from star_pattern.core.catalog import StarCatalog, CatalogEntry
            all_entries: list[CatalogEntry] = []
            for cat in region_data.catalogs.values():
                all_entries.extend(cat.entries)
            if all_entries:
                merged_catalog = StarCatalog(entries=all_entries, source="merged")

        pattern_results: list[dict[str, Any]] = []
        detection_summaries: list[dict[str, Any]] = []

        for band, image in region_data.images.items():
            try:
                # Gather temporal images
                temporal_imgs = None
                if region_data.has_temporal_images():
                    band_key = band.split("_")[-1]
                    temporal_imgs = region_data.temporal_images.get(band_key)
                    if not temporal_imgs:
                        for t_band, t_imgs in region_data.temporal_images.items():
                            if len(t_imgs) >= 2:
                                temporal_imgs = t_imgs
                                break

                detection = self._detector.detect(
                    image, catalog=merged_catalog,
                    temporal_images=temporal_imgs,
                )

                anomaly_score = detection.get(
                    "meta_score", detection.get("anomaly_score", 0)
                )

                classification = self.local_classifier.classify(detection)
                evaluation = self.local_evaluator.evaluate(detection, image)

                result = PatternResult(
                    region_ra=region.ra,
                    region_dec=region.dec,
                    detection_type=classification["classification"],
                    anomaly_score=anomaly_score,
                    significance=detection.get("lens", {}).get("lens_score", 0),
                )
                result.anomalies = _extract_anomalies(detection, image)
                result.hypothesis = classification["rationale"]
                result.debate_verdict = evaluation["verdict"]
                result.metadata["local_classification"] = classification
                result.metadata["local_evaluation"] = evaluation
                result.metadata["significance_rating"] = evaluation["significance_rating"]

                pattern_results.append(result.to_dict())

                # Compact detection summary (avoid sending full detection dict)
                detection_summaries.append({
                    "band": band,
                    "anomaly_score": anomaly_score,
                    "classification": classification["classification"],
                    "verdict": evaluation["verdict"],
                    "n_anomalies": len(result.anomalies),
                })

            except Exception as e:
                logger.warning(f"Detection failed on {band}: {e}")

        return WorkResult(
            work_id=unit.work_id,
            slave_id=self.slave_id,
            region=unit.region,
            pattern_results=pattern_results,
            detection_summaries=detection_summaries,
        )
