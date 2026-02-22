#!/usr/bin/env python3
"""Top-level script for batch survey scanning."""

import sys
from pathlib import Path

from star_pattern.core.config import PipelineConfig
from star_pattern.pipeline.batch import BatchProcessor
from star_pattern.visualization.report import DiscoveryReport
from star_pattern.utils.run_manager import RunManager
from star_pattern.utils.logging import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger("survey")


def main():
    config_path = Path("config.json")
    if config_path.exists():
        config = PipelineConfig.from_file(config_path)
    else:
        config = PipelineConfig()

    n_regions = 100
    regions_file = None
    for arg in sys.argv[1:]:
        if arg.startswith("--n="):
            n_regions = int(arg.split("=")[1])
        elif arg.startswith("--regions="):
            regions_file = arg.split("=")[1]

    run_mgr = RunManager(base_dir=config.output_dir)
    processor = BatchProcessor(config, run_manager=run_mgr)

    if regions_file:
        logger.info(f"Processing regions from {regions_file}")
        findings = processor.process_from_file(regions_file)
    else:
        logger.info(f"Processing {n_regions} random regions")
        findings = processor.process_random(n_regions)

    # Report
    report = DiscoveryReport(run_mgr.reports_dir)
    paths = report.generate_full_report(
        findings,
        run_metadata={
            "run_name": run_mgr.run_name,
            "n_regions": n_regions,
            "regions_file": regions_file,
        },
    )

    logger.info(f"\nSurvey complete!")
    logger.info(f"  Findings: {len(findings)}")
    for name, path in paths.items():
        logger.info(f"  {name}: {path}")


if __name__ == "__main__":
    main()
