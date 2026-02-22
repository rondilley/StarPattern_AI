"""Batch processing of sky regions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from star_pattern.core.config import PipelineConfig
from star_pattern.core.sky_region import SkyRegion
from star_pattern.data.pipeline import DataPipeline
from star_pattern.detection.ensemble import EnsembleDetector
from star_pattern.evaluation.metrics import PatternResult
from star_pattern.utils.run_manager import RunManager
from star_pattern.utils.logging import get_logger

logger = get_logger("pipeline.batch")


class BatchProcessor:
    """Process a batch of sky regions through the detection pipeline."""

    def __init__(self, config: PipelineConfig, run_manager: RunManager | None = None):
        self.config = config
        self.run_manager = run_manager or RunManager(base_dir=config.output_dir)
        self.data_pipeline = DataPipeline(config.data)
        self.detector = EnsembleDetector(config.detection)

    def process_regions(
        self,
        regions: list[SkyRegion],
        save_results: bool = True,
    ) -> list[PatternResult]:
        """Process a batch of sky regions.

        Args:
            regions: List of SkyRegion to process.
            save_results: Whether to save results to disk.

        Returns:
            List of all PatternResult objects.
        """
        all_results: list[PatternResult] = []

        for i, region in enumerate(regions):
            logger.info(f"Processing region {i + 1}/{len(regions)}: {region}")

            try:
                region_data = self.data_pipeline.fetch_region(region)

                if not region_data.has_images():
                    logger.info("  No images, skipping")
                    continue

                for band, image in region_data.images.items():
                    detection = self.detector.detect(image)
                    score = detection.get("anomaly_score", 0)

                    result = PatternResult(
                        region_ra=region.ra,
                        region_dec=region.dec,
                        detection_type="ensemble",
                        anomaly_score=score,
                        details=detection,
                    )
                    all_results.append(result)

            except Exception as e:
                logger.error(f"  Failed: {e}")

        # Sort by score
        all_results.sort(key=lambda r: r.anomaly_score, reverse=True)

        if save_results:
            self.run_manager.save_result(
                "batch_results",
                {
                    "n_regions": len(regions),
                    "n_results": len(all_results),
                    "results": [r.to_dict() for r in all_results],
                },
            )

        logger.info(f"Batch complete: {len(all_results)} results from {len(regions)} regions")
        return all_results

    def process_random(
        self,
        n_regions: int,
        min_gal_lat: float = 20.0,
        radius: float = 3.0,
    ) -> list[PatternResult]:
        """Process random sky regions."""
        rng = np.random.default_rng()
        regions = [
            SkyRegion.random(min_galactic_lat=min_gal_lat, radius=radius, rng=rng)
            for _ in range(n_regions)
        ]
        return self.process_regions(regions)

    def process_from_file(self, path: str | Path) -> list[PatternResult]:
        """Process regions from a JSON file.

        File format: [{"ra": 180.0, "dec": 45.0, "radius": 3.0}, ...]
        """
        path = Path(path)
        with open(path) as f:
            region_dicts = json.load(f)

        regions = [
            SkyRegion(
                ra=r["ra"],
                dec=r["dec"],
                radius=r.get("radius", self.config.data.default_radius_arcmin),
            )
            for r in region_dicts
        ]

        return self.process_regions(regions)
