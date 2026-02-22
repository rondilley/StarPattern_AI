#!/usr/bin/env python3
"""Top-level script for autonomous discovery."""

import sys
from pathlib import Path

from star_pattern.core.config import PipelineConfig
from star_pattern.pipeline.autonomous import AutonomousDiscovery
from star_pattern.visualization.report import DiscoveryReport
from star_pattern.utils.run_manager import RunManager
from star_pattern.utils.logging import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger("main")


def main():
    # Load config
    config_path = Path("config.json")
    if config_path.exists():
        config = PipelineConfig.from_file(config_path)
    else:
        config = PipelineConfig()

    # Parse command line args (simple)
    max_hours = None
    use_llm = False
    for arg in sys.argv[1:]:
        if arg.startswith("--hours="):
            max_hours = float(arg.split("=")[1])
        elif arg == "--with-llm":
            use_llm = True
        elif arg.startswith("--cycles="):
            config.max_cycles = int(arg.split("=")[1])

    run_mgr = RunManager(base_dir=config.output_dir)
    logger.info(f"Starting autonomous discovery run: {run_mgr.run_name}")

    # Run discovery
    pipeline = AutonomousDiscovery(config, run_manager=run_mgr, use_llm=use_llm)
    findings = pipeline.run(max_hours=max_hours)

    # Generate report
    report = DiscoveryReport(run_mgr.reports_dir)
    paths = report.generate_full_report(
        findings,
        run_metadata={
            "run_name": run_mgr.run_name,
            "n_cycles": pipeline.cycle,
            "n_regions": len(pipeline.searched_regions),
            "use_llm": use_llm,
        },
    )

    logger.info(f"\nDiscovery complete!")
    logger.info(f"  Findings: {len(findings)}")
    logger.info(f"  Reports: {', '.join(str(p) for p in paths.values())}")


if __name__ == "__main__":
    main()
