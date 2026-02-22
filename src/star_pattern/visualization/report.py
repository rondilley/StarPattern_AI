"""PDF/HTML report generation for discovery results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from star_pattern.evaluation.metrics import PatternResult
from star_pattern.utils.logging import get_logger

logger = get_logger("visualization.report")


class DiscoveryReport:
    """Generate reports from discovery results."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_text_report(
        self,
        findings: list[PatternResult],
        run_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Generate a text summary report."""
        report_path = self.output_dir / "report.txt"

        lines = [
            "=" * 70,
            "STAR PATTERN AI - DISCOVERY REPORT",
            "=" * 70,
            "",
        ]

        if run_metadata:
            lines.append(f"Run: {run_metadata.get('run_name', 'unknown')}")
            lines.append(f"Cycles: {run_metadata.get('n_cycles', 'N/A')}")
            lines.append(f"Regions: {run_metadata.get('n_regions', 'N/A')}")
            lines.append("")

        lines.append(f"Total findings: {len(findings)}")
        if findings:
            scores = [f.anomaly_score for f in findings]
            lines.append(f"Score range: [{min(scores):.4f}, {max(scores):.4f}]")
            lines.append(f"Mean score: {sum(scores) / len(scores):.4f}")
            lines.append("")

        # Top findings
        sorted_findings = sorted(findings, key=lambda f: f.anomaly_score, reverse=True)
        lines.append("-" * 70)
        lines.append("TOP FINDINGS")
        lines.append("-" * 70)

        for i, f in enumerate(sorted_findings[:20]):
            lines.append(f"\n#{i + 1}: {f.detection_type}")
            lines.append(f"  Location: RA={f.region_ra:.4f}, Dec={f.region_dec:.4f}")
            lines.append(f"  Anomaly Score: {f.anomaly_score:.4f}")
            lines.append(f"  Significance: {f.significance:.4f}")
            lines.append(f"  Combined Score: {f.combined_score:.4f}")

            if f.cross_matches:
                lines.append(f"  Cross-matches: {len(f.cross_matches)}")
                for m in f.cross_matches[:3]:
                    lines.append(f"    - {m.get('name', 'N/A')} ({m.get('object_type', '')})")

            if f.hypothesis:
                lines.append(f"  Hypothesis: {f.hypothesis[:200]}")

            if f.debate_verdict:
                lines.append(f"  Debate Verdict: {f.debate_verdict}")

            if f.consensus_score is not None:
                lines.append(f"  Consensus Score: {f.consensus_score:.1f}/10")

        # Statistics
        lines.append("\n" + "-" * 70)
        lines.append("STATISTICS")
        lines.append("-" * 70)

        by_type: dict[str, list[float]] = {}
        for f in findings:
            by_type.setdefault(f.detection_type, []).append(f.anomaly_score)

        for dtype, scores in sorted(by_type.items()):
            import numpy as np
            lines.append(
                f"  {dtype}: {len(scores)} findings, "
                f"mean={np.mean(scores):.4f}, max={np.max(scores):.4f}"
            )

        # Evolution history
        if run_metadata:
            evolution = run_metadata.get("evolution_history", [])
            if evolution:
                lines.append("\n" + "-" * 70)
                lines.append("EVOLUTION HISTORY")
                lines.append("-" * 70)
                for entry in evolution:
                    lines.append(
                        f"  Cycle {entry['cycle']}: fitness={entry['fitness']:.4f}"
                    )

        lines.append("\n" + "=" * 70)
        lines.append("END OF REPORT")

        report_path.write_text("\n".join(lines))
        logger.info(f"Report saved to {report_path}")
        return report_path

    def generate_json_report(
        self,
        findings: list[PatternResult],
        run_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Generate a JSON report."""
        report_path = self.output_dir / "report.json"
        data = {
            "metadata": run_metadata or {},
            "n_findings": len(findings),
            "findings": [f.to_dict() for f in findings],
        }
        report_path.write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"JSON report saved to {report_path}")
        return report_path

    def generate_full_report(
        self,
        findings: list[PatternResult],
        run_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Generate all report formats."""
        paths = {}
        paths["text"] = self.generate_text_report(findings, run_metadata)
        paths["json"] = self.generate_json_report(findings, run_metadata)

        # Generate plots
        try:
            from star_pattern.visualization.mosaic import (
                create_discovery_mosaic,
                create_score_histogram,
            )

            mosaic = create_discovery_mosaic(findings)
            mosaic_path = self.output_dir / "mosaic.png"
            mosaic.savefig(str(mosaic_path), dpi=150, bbox_inches="tight")
            paths["mosaic"] = mosaic_path
            import matplotlib.pyplot as plt
            plt.close(mosaic)

            hist = create_score_histogram(findings)
            hist_path = self.output_dir / "scores_histogram.png"
            hist.savefig(str(hist_path), dpi=150, bbox_inches="tight")
            paths["histogram"] = hist_path
            plt.close(hist)

        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")

        return paths
