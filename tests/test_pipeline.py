"""Tests for pipeline orchestration."""

import json
import pytest
from pathlib import Path

import numpy as np

from star_pattern.core.config import PipelineConfig, DetectionConfig
from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.sky_region import SkyRegion, RegionData
from star_pattern.discovery.genome import DetectionGenome
from star_pattern.discovery.fitness import FitnessEvaluator
from star_pattern.evaluation.metrics import Anomaly, PatternResult
from star_pattern.visualization.report import DiscoveryReport
from star_pattern.utils.run_manager import RunManager


class TestRunManager:
    def test_create_run(self, tmp_path):
        mgr = RunManager(base_dir=tmp_path / "runs")
        assert mgr.run_dir.exists()
        assert mgr.images_dir.exists()

    def test_save_load_checkpoint(self, tmp_path):
        mgr = RunManager(base_dir=tmp_path / "runs")
        data = {"key": "value", "number": 42}
        mgr.save_checkpoint("test_cp", data)
        loaded = mgr.load_checkpoint("test_cp")
        assert loaded == data

    def test_save_load_result(self, tmp_path):
        mgr = RunManager(base_dir=tmp_path / "runs")
        data = {"findings": [1, 2, 3]}
        mgr.save_result("test_result", data)
        loaded = mgr.load_result("test_result")
        assert loaded == data

    def test_update_state(self, tmp_path):
        mgr = RunManager(base_dir=tmp_path / "runs")
        mgr.update_state(cycle=5, status="running")
        assert mgr.state["cycle"] == 5

    def test_load_nonexistent_checkpoint(self, tmp_path):
        mgr = RunManager(base_dir=tmp_path / "runs")
        assert mgr.load_checkpoint("nonexistent") is None

    def test_load_nonexistent_result(self, tmp_path):
        mgr = RunManager(base_dir=tmp_path / "runs")
        assert mgr.load_result("nonexistent") is None

    def test_named_run(self, tmp_path):
        mgr = RunManager(base_dir=tmp_path / "runs", run_name="my_run")
        assert mgr.run_name == "my_run"
        assert "my_run" in str(mgr.run_dir)


class TestDiscoveryReport:
    def test_text_report(self, tmp_path):
        report = DiscoveryReport(tmp_path / "reports")
        findings = [
            PatternResult(180.0, 45.0, "lens", 0.8, 0.6),
            PatternResult(90.0, 30.0, "morphology", 0.3, 0.2),
        ]
        path = report.generate_markdown_report(findings)
        assert path.exists()
        content = path.read_text()
        assert "Lens" in content or "lens" in content
        assert "180.0" in content

    def test_json_report(self, tmp_path):
        report = DiscoveryReport(tmp_path / "reports")
        findings = [PatternResult(180.0, 45.0, "lens", 0.8)]
        path = report.generate_json_report(findings)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["n_findings"] == 1

    def test_empty_report(self, tmp_path):
        report = DiscoveryReport(tmp_path / "reports")
        path = report.generate_markdown_report([])
        assert path.exists()

    def test_full_report(self, tmp_path):
        report = DiscoveryReport(tmp_path / "reports")
        findings = [
            PatternResult(180.0, 45.0, "lens", 0.8, 0.6),
        ]
        paths = report.generate_full_report(findings, run_metadata={"run_name": "test"})
        assert "markdown" in paths
        assert "json" in paths


class TestPipelineConfig:
    def test_from_file(self, tmp_path):
        config_data = {
            "data": {"sources": ["sdss"], "cache_dir": str(tmp_path)},
            "detection": {"source_extraction_threshold": 2.5},
            "evolution": {"population_size": 20},
            "llm": {"temperature": 0.5},
            "pipeline": {"output_dir": str(tmp_path), "max_cycles": 100},
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data))

        config = PipelineConfig.from_file(config_path)
        assert config.data.sources == ["sdss"]
        assert config.detection.source_extraction_threshold == 2.5
        assert config.evolution.population_size == 20
        assert config.max_cycles == 100

    def test_to_dict(self):
        config = PipelineConfig()
        d = config.to_dict()
        assert "data" in d
        assert "detection" in d
        assert "evolution" in d

    def test_defaults(self):
        config = PipelineConfig()
        assert config.data.sources == ["sdss", "gaia", "mast", "ztf"]
        assert config.evolution.population_size == 30
        assert config.llm.temperature == 0.7


class TestPatternResult:
    def test_combined_score(self):
        result = PatternResult(180.0, 45.0, "lens", 0.8, significance=0.6, novelty=0.5)
        score = result.combined_score
        assert 0 < score < 1

    def test_to_dict(self):
        result = PatternResult(180.0, 45.0, "lens", 0.8)
        d = result.to_dict()
        assert d["ra"] == 180.0
        assert d["dec"] == 45.0
        assert d["type"] == "lens"
        assert d["anomaly_score"] == 0.8

    def test_repr(self):
        result = PatternResult(180.0, 45.0, "lens", 0.8)
        r = repr(result)
        assert "lens" in r
        assert "180" in r

    def test_metadata(self):
        result = PatternResult(180.0, 45.0, "lens", 0.8)
        result.metadata["test_key"] = "test_value"
        d = result.to_dict()
        assert d["metadata"]["test_key"] == "test_value"


class TestDetectionConfigFromGenome:
    def test_from_genome_dict(self):
        """Create a genome dict manually, verify all fields are set."""
        genome_dict = {
            "source_extraction": {"threshold": 4.5, "min_area": 7},
            "gabor": {"frequencies": [0.03, 0.08, 0.15], "n_orientations": 12},
            "anomaly": {"contamination": 0.08, "n_estimators": 150},
            "lens": {"arc_min_length": 15, "snr_threshold": 3.0},
            "morphology": {"smoothing_sigma": 2.5},
            "distribution": {"overdensity_sigma": 3.5},
            "galaxy": {
                "tidal_threshold": 0.4,
                "color_sigma": 3.0,
                "asymmetry_threshold": 0.35,
            },
            "kinematic": {
                "pm_min": 8.0,
                "cluster_eps": 3.5,
                "cluster_min": 7,
                "stream_min_length": 10,
            },
            "transient": {"noise_threshold": 4.0, "parallax_snr": 6.0},
            "ensemble_weights": {"classical": 0.2, "morphology": 0.15},
        }
        config = DetectionConfig.from_genome_dict(genome_dict)
        assert config.source_extraction_threshold == 4.5
        assert config.gabor_orientations == 12
        assert config.anomaly_contamination == 0.08
        assert config.galaxy_tidal_threshold == 0.4
        assert config.kinematic_pm_min == 8.0
        assert config.transient_noise_threshold == 4.0
        assert config.ensemble_weights["classical"] == 0.2

    def test_from_genome_dict_defaults(self):
        """Empty dict should use defaults without crashing."""
        config = DetectionConfig.from_genome_dict({})
        assert config.source_extraction_threshold == 3.0
        assert config.gabor_orientations == 8

    def test_genome_roundtrip(self):
        """DetectionGenome -> to_detection_config -> from_genome_dict produces valid config."""
        genome = DetectionGenome(rng=np.random.default_rng(42))
        genome_dict = genome.to_detection_config()
        config = DetectionConfig.from_genome_dict(genome_dict)
        assert config.source_extraction_threshold > 0
        assert len(config.gabor_frequencies) > 0
        assert 0 < config.anomaly_contamination < 1


class TestFitnessEvaluatorWithGenome:
    def test_fitness_evaluator_uses_all_genome_fields(self, synthetic_image):
        """Create a genome, evaluate with FitnessEvaluator, verify no crash."""
        genome = DetectionGenome(rng=np.random.default_rng(42))
        genome_config = genome.to_detection_config()
        evaluator = FitnessEvaluator()
        result = evaluator.evaluate(genome_config, [synthetic_image])
        assert "fitness" in result
        assert result["fitness"] >= 0
        assert "anomaly" in result
        assert "significance" in result
        assert "novelty" in result
        assert "diversity" in result


class TestAutonomousSavesImages:
    def test_autonomous_saves_images(
        self, sample_config, sample_region_data, tmp_path,
    ):
        """Run detection on a region and verify images are saved."""
        from star_pattern.pipeline.autonomous import AutonomousDiscovery

        run_mgr = RunManager(base_dir=tmp_path / "runs")
        sample_config.max_cycles = 1
        pipeline = AutonomousDiscovery(sample_config, run_manager=run_mgr)

        # Process the region directly to avoid needing network
        pipeline.cycle = 1
        results = pipeline._process_region(sample_region_data)

        # If there were findings above threshold, images should be saved
        images_dir = run_mgr.images_dir
        if results:
            png_files = list(images_dir.glob("*.png"))
            assert len(png_files) > 0, "Expected overlay images to be saved"

    def test_autonomous_generates_report(
        self, sample_config, tmp_path,
    ):
        """Verify report generation produces files."""
        from star_pattern.pipeline.autonomous import AutonomousDiscovery

        run_mgr = RunManager(base_dir=tmp_path / "runs")
        sample_config.max_cycles = 0
        pipeline = AutonomousDiscovery(sample_config, run_manager=run_mgr)

        # Manually add a finding so report has content
        pipeline.findings.append(
            PatternResult(180.0, 45.0, "lens", 0.8, 0.6)
        )
        pipeline.cycle = 1

        report_paths = pipeline._generate_report()
        assert "markdown" in report_paths
        assert "json" in report_paths
        assert report_paths["markdown"].exists()
        assert report_paths["json"].exists()


class TestWideFieldConfig:
    def test_wide_field_config_defaults(self):
        """WideFieldConfig has sensible defaults."""
        from star_pattern.core.config import WideFieldConfig

        wf = WideFieldConfig()
        assert wf.tile_radius_arcmin == 3.0
        assert wf.overlap_fraction == 0.2
        assert wf.max_tiles == 500
        assert wf.mosaic_pixel_scale_arcsec == 0.4
        assert wf.mosaic_combine == "mean"
        assert wf.fetch_all_sources is True
        assert wf.max_mast_observations == 10

    def test_pipeline_config_has_wide_field(self):
        """PipelineConfig includes wide_field field."""
        config = PipelineConfig()
        assert hasattr(config, "wide_field")
        assert config.wide_field.tile_radius_arcmin == 3.0

    def test_from_dict_with_wide_field(self, tmp_path):
        """from_dict parses wide_field section."""
        config_data = {
            "wide_field": {
                "tile_radius_arcmin": 5.0,
                "overlap_fraction": 0.3,
                "max_tiles": 200,
            },
        }
        config = PipelineConfig.from_dict(config_data)
        assert config.wide_field.tile_radius_arcmin == 5.0
        assert config.wide_field.overlap_fraction == 0.3
        assert config.wide_field.max_tiles == 200

    def test_data_config_default_sources(self):
        """DataConfig defaults to all three sources."""
        from star_pattern.core.config import DataConfig

        dc = DataConfig()
        assert "sdss" in dc.sources
        assert "gaia" in dc.sources
        assert "mast" in dc.sources

    def test_to_dict_includes_wide_field(self):
        """to_dict serializes wide_field."""
        config = PipelineConfig()
        d = config.to_dict()
        assert "wide_field" in d
        assert d["wide_field"]["tile_radius_arcmin"] == 3.0


class TestStarCatalogMerge:
    def test_merge_deduplicates(self, sample_catalog):
        """Merge deduplicates by source_id."""
        from star_pattern.core.catalog import StarCatalog, CatalogEntry

        # Create a second catalog with some overlapping IDs
        entries = [
            CatalogEntry(ra=181.0, dec=46.0, source_id="test_0", source="test"),
            CatalogEntry(ra=182.0, dec=47.0, source_id="new_1", source="test"),
        ]
        other = StarCatalog(entries=entries, source="test")

        merged = sample_catalog.merge(other)
        # test_0 was already in sample_catalog, so it should not be duplicated
        ids = [e.source_id for e in merged.entries if e.source_id == "test_0"]
        assert len(ids) == 1
        # new_1 should be added
        ids_new = [e.source_id for e in merged.entries if e.source_id == "new_1"]
        assert len(ids_new) == 1
        # Total should be original + 1 new
        assert len(merged) == len(sample_catalog) + 1

    def test_merge_empty(self, sample_catalog):
        """Merge with empty catalog returns original entries."""
        from star_pattern.core.catalog import StarCatalog

        empty = StarCatalog(source="empty")
        merged = sample_catalog.merge(empty)
        assert len(merged) == len(sample_catalog)


class TestPixelScaleDetection:
    def test_ensemble_passes_pixel_scale(self, synthetic_image_with_wcs):
        """EnsembleDetector extracts pixel scale from WCS image."""
        from star_pattern.detection.ensemble import EnsembleDetector

        detector = EnsembleDetector()
        result = detector.detect(synthetic_image_with_wcs)
        # Should have pixel_scale_arcsec in results
        assert "pixel_scale_arcsec" in result
        ps = result["pixel_scale_arcsec"]
        assert ps is not None
        assert abs(ps - 0.4) < 0.01  # Our fixture uses 0.4 arcsec/pixel

    def test_lens_adapts_radii(self):
        """LensDetector adapts radii when pixel_scale_arcsec is given."""
        from star_pattern.detection.lens_detector import LensDetector

        det = LensDetector()
        # Default radii
        assert det.ring_min_radius == 10
        assert det.ring_max_radius == 80

        # Create synthetic image data
        data = np.random.default_rng(42).normal(100, 10, (256, 256))

        # Detect with a pixel scale of 0.2 arcsec/pixel
        det.detect(data.astype(np.float32), pixel_scale_arcsec=0.2)
        # 3.0 arcsec / 0.2 = 15 pixels min
        assert det.ring_min_radius == 15
        # 25.0 arcsec / 0.2 = 125 pixels max
        assert det.ring_max_radius == 125


class TestReportEvolutionHistory:
    def test_evolution_history_in_report(self, tmp_path):
        """Verify evolution history appears in text report."""
        report = DiscoveryReport(tmp_path / "reports")
        findings = [PatternResult(180.0, 45.0, "lens", 0.8, 0.6)]
        metadata = {
            "run_name": "test",
            "evolution_history": [
                {"cycle": 25, "fitness": 0.4321},
                {"cycle": 50, "fitness": 0.5678},
            ],
        }
        path = report.generate_markdown_report(findings, run_metadata=metadata)
        content = path.read_text()
        assert "Parameter Evolution" in content
        assert "0.4321" in content
        assert "0.5678" in content


class TestSubDetectionExtraction:
    """Test _extract_sub_detections and _total_sub_detections."""

    def test_empty_details(self):
        from star_pattern.visualization.mosaic import (
            _extract_sub_detections,
            _total_sub_detections,
        )
        assert _extract_sub_detections({}) == {}
        assert _extract_sub_detections(None) == {}
        assert _total_sub_detections({}) == 0

    def test_realistic_details(self):
        from star_pattern.visualization.mosaic import (
            _extract_sub_detections,
            _total_sub_detections,
        )

        details = {
            "sources": {"n_sources": 247, "positions": [[0, 0]] * 247},
            "wavelet": {
                "wavelet_score": 0.6,
                "detections": [{"x": i, "y": i, "scale": 0} for i in range(42)],
                "multiscale_objects": [{"x": 0, "y": 0}],
            },
            "lens": {
                "lens_score": 0.4,
                "arcs": [{"radius": 10}, {"radius": 20}],
                "rings": [],
            },
            "distribution": {
                "distribution_score": 0.3,
                "overdensities": [
                    {"x": 50, "y": 50, "radius_px": 10},
                    {"x": 100, "y": 100, "radius_px": 15},
                    {"x": 150, "y": 150, "radius_px": 12},
                ],
            },
            "sersic": {
                "sersic_score": 0.2,
                "residual_features": [{"x": 0, "y": 0, "area_px": 50}],
            },
            "transient": {
                "transient_score": 0.3,
                "flux_outliers": [{"x": 0, "y": 0}],
                "color_outliers": [{"x": 1, "y": 1}, {"x": 2, "y": 2}],
            },
        }
        counts = _extract_sub_detections(details)
        assert counts["sources"] == 247
        assert counts["wavelet features"] == 42
        assert counts["multiscale objects"] == 1
        assert counts["lens arcs"] == 2
        assert counts["overdensities"] == 3
        assert counts["Sersic residuals"] == 1
        assert counts["transient outliers"] == 3

        total = _total_sub_detections(details)
        # 42 + 1 + 2 + 3 + 1 + 3 = 52 (sources excluded)
        assert total == 52

    def test_kinematic_details(self):
        from star_pattern.visualization.mosaic import _extract_sub_detections

        details = {
            "kinematic": {
                "comoving_groups": [{"id": 1}, {"id": 2}],
                "stream_candidates": [{"id": 3}],
                "runaway_stars": [],
            },
        }
        counts = _extract_sub_detections(details)
        assert counts["comoving groups"] == 2
        assert counts["stream candidates"] == 1
        assert "runaway stars" not in counts  # zero excluded

    def test_variability_population(self):
        from star_pattern.visualization.mosaic import _extract_sub_detections

        details = {
            "variability": {
                "variable_candidates": [{"id": i} for i in range(5)],
                "periodic_candidates": [{"id": i} for i in range(3)],
            },
            "population": {
                "n_blue_stragglers": 7,
                "n_red_giants": 12,
            },
        }
        counts = _extract_sub_detections(details)
        assert counts["variable candidates"] == 5
        assert counts["periodic candidates"] == 3
        assert counts["blue stragglers"] == 7
        assert counts["red giants"] == 12


class TestCategorizeFindings:
    """Test _categorize_findings three-way split."""

    def test_three_way_split(self):
        from star_pattern.visualization.mosaic import _categorize_findings

        # New discovery: no cross-matches, decent evaluation
        f_new = PatternResult(180.0, 45.0, "lens", 0.8, 0.6)
        f_new.metadata["local_evaluation"] = {
            "verdict": "real",
            "significance_rating": 7,
        }

        # Known object: has cross-matches
        f_known = PatternResult(90.0, 30.0, "galaxy", 0.7, 0.5)
        f_known.metadata["local_evaluation"] = {
            "verdict": "real",
            "significance_rating": 6,
        }
        f_known.cross_matches = [{"name": "NGC 1234", "object_type": "Galaxy"}]

        # Low confidence: artifact verdict
        f_artifact = PatternResult(45.0, 15.0, "noise", 0.2, 0.1)
        f_artifact.metadata["local_evaluation"] = {
            "verdict": "artifact",
            "significance_rating": 1,
        }

        # Low confidence: low sig_rating (<=2)
        f_low_sig = PatternResult(60.0, 20.0, "morphology", 0.4, 0.3)
        f_low_sig.metadata["local_evaluation"] = {
            "verdict": "uncertain",
            "significance_rating": 2,
        }

        new, known, low = _categorize_findings(
            [f_artifact, f_known, f_new, f_low_sig],
        )
        assert len(new) == 1
        assert new[0] is f_new
        assert len(known) == 1
        assert known[0] is f_known
        assert len(low) == 2
        assert f_artifact in low
        assert f_low_sig in low

    def test_sorted_by_score(self):
        from star_pattern.visualization.mosaic import _categorize_findings

        f1 = PatternResult(180.0, 45.0, "lens", 0.6)
        f1.metadata["local_evaluation"] = {
            "verdict": "real",
            "significance_rating": 5,
        }
        f2 = PatternResult(90.0, 30.0, "wavelet", 0.9)
        f2.metadata["local_evaluation"] = {
            "verdict": "real",
            "significance_rating": 8,
        }

        new, _, _ = _categorize_findings([f1, f2])
        assert len(new) == 2
        # Higher score first
        assert new[0].anomaly_score > new[1].anomaly_score

    def test_empty(self):
        from star_pattern.visualization.mosaic import _categorize_findings
        new, known, low = _categorize_findings([])
        assert new == []
        assert known == []
        assert low == []


class TestReportSubDetections:
    """Test that reports include sub-detection counts."""

    def test_markdown_has_feature_descriptions(self, tmp_path):
        report = DiscoveryReport(tmp_path / "reports")
        f = PatternResult(180.0, 45.0, "lens", 0.8, 0.6)
        f.metadata["local_evaluation"] = {
            "verdict": "real",
            "significance_rating": 7,
        }
        f.details = {
            "sources": {"n_sources": 100},
            "wavelet": {
                "detections": [{"x": i, "y": i, "scale": 0} for i in range(30)],
            },
            "lens": {
                "arcs": [{"radius": 10}, {"radius": 20}],
            },
        }
        path = report.generate_markdown_report(
            [f], run_metadata={"n_regions": 1},
        )
        content = path.read_text()
        # Feature descriptions should appear as itemized list
        assert "100 extracted sources" in content
        assert "wavelet features" in content
        assert "lens arcs" in content
        assert "anomalous sub-detections" in content

    def test_low_confidence_has_assessment_column(self, tmp_path):
        report = DiscoveryReport(tmp_path / "reports")
        f = PatternResult(45.0, 15.0, "noise", 0.2, 0.1)
        f.metadata["local_evaluation"] = {
            "verdict": "artifact",
            "significance_rating": 1,
        }
        f.details = {
            "wavelet": {
                "detections": [{"x": i, "y": i, "scale": 0} for i in range(5)],
            },
        }
        path = report.generate_markdown_report([f])
        content = path.read_text()
        assert "Assessment" in content
        assert "Confidence" in content

    def test_narrative_summary(self, tmp_path):
        report = DiscoveryReport(tmp_path / "reports")
        f1 = PatternResult(180.0, 45.0, "lens", 0.8, 0.6)
        f1.metadata["local_evaluation"] = {
            "verdict": "real", "significance_rating": 7,
        }
        f1.details = {
            "sources": {"n_sources": 50},
            "wavelet": {
                "detections": [{"x": i, "y": i} for i in range(20)],
            },
        }
        f2 = PatternResult(90.0, 30.0, "galaxy", 0.7, 0.5)
        f2.metadata["local_evaluation"] = {
            "verdict": "real", "significance_rating": 6,
        }
        f2.details = {
            "sources": {"n_sources": 80},
            "lens": {"arcs": [{"radius": 10}]},
        }
        path = report.generate_markdown_report(
            [f1, f2], run_metadata={"n_regions": 2},
        )
        content = path.read_text()
        # Summary should have narrative counts, not raw source totals
        assert "## Summary" in content
        assert "follow-up" in content
        assert "F1" in content


class TestReportFindingStructure:
    """Test the new finding-number and narrative structure."""

    def _make_finding(self, ra, dec, dtype, score, sig_rating, verdict,
                      cross_matches=None):
        f = PatternResult(ra, dec, dtype, score, score * 0.8)
        f.metadata["local_evaluation"] = {
            "verdict": verdict,
            "significance_rating": sig_rating,
        }
        if cross_matches:
            f.cross_matches = cross_matches
        return f

    def test_finding_numbers_in_report(self, tmp_path):
        report = DiscoveryReport(tmp_path / "reports")
        f1 = self._make_finding(180.0, 45.0, "lens", 0.9, 8, "real")
        f2 = self._make_finding(90.0, 30.0, "galaxy", 0.7, 6, "real",
                                cross_matches=[{"name": "NGC 1234"}])
        f3 = self._make_finding(45.0, 15.0, "noise", 0.2, 1, "artifact")

        path = report.generate_markdown_report([f1, f2, f3])
        content = path.read_text()
        # Sequential finding numbers
        assert "F1" in content
        assert "F2" in content
        assert "F3" in content

    def test_finding_narrative_structure(self, tmp_path):
        report = DiscoveryReport(tmp_path / "reports")
        f = self._make_finding(180.0, 45.0, "lens", 0.9, 8, "real")
        f.details = {
            "wavelet": {
                "detections": [{"x": i, "y": i, "scale": 0} for i in range(10)],
            },
        }

        path = report.generate_markdown_report([f])
        content = path.read_text()
        # Each finding should have these narrative sections
        assert "What was detected" in content
        assert "Catalog cross-reference" in content
        assert "Evidence" in content

    def test_mosaic_has_finding_labels(self, tmp_path):
        from star_pattern.visualization.mosaic import create_discovery_mosaic

        f = PatternResult(180.0, 45.0, "lens", 0.8, 0.6)
        f.metadata["local_evaluation"] = {
            "verdict": "real",
            "significance_rating": 7,
        }
        fig = create_discovery_mosaic([f])
        # Check that "F1" appears in the figure's text elements
        texts = [t.get_text() for t in fig.texts]
        # Also check axes text
        for ax in fig.get_axes():
            texts.extend([t.get_text() for t in ax.texts])
        assert any("F1" in t for t in texts)


class TestAnomalyDataclass:
    """Test the Anomaly dataclass and its serialization."""

    def test_create_anomaly(self):
        a = Anomaly(
            anomaly_type="lens_arc",
            detector="lens",
            pixel_x=100.5,
            pixel_y=200.3,
            sky_ra=192.654,
            sky_dec=26.609,
            score=4.2,
            properties={"radius": 15, "angle_span": 120},
        )
        assert a.anomaly_type == "lens_arc"
        assert a.detector == "lens"
        assert a.pixel_x == 100.5
        assert a.score == 4.2
        assert a.properties["radius"] == 15

    def test_anomaly_defaults(self):
        a = Anomaly(anomaly_type="overdensity", detector="distribution")
        assert a.pixel_x is None
        assert a.pixel_y is None
        assert a.sky_ra is None
        assert a.sky_dec is None
        assert a.score == 0.0
        assert a.properties == {}

    def test_anomaly_to_dict(self):
        a = Anomaly(
            anomaly_type="comoving_group",
            detector="kinematic",
            sky_ra=180.0,
            sky_dec=45.0,
            score=3.5,
            properties={"n_members": 7},
        )
        d = a.to_dict()
        assert d["anomaly_type"] == "comoving_group"
        assert d["detector"] == "kinematic"
        assert d["pixel_x"] is None
        assert d["sky_ra"] == 180.0
        assert d["score"] == 3.5
        assert d["properties"]["n_members"] == 7

    def test_pattern_result_anomalies_field(self):
        result = PatternResult(180.0, 45.0, "lens", 0.8)
        assert result.anomalies == []
        result.anomalies.append(
            Anomaly(anomaly_type="lens_arc", detector="lens", score=4.0)
        )
        assert len(result.anomalies) == 1

    def test_pattern_result_to_dict_with_anomalies(self):
        result = PatternResult(180.0, 45.0, "lens", 0.8)
        result.anomalies = [
            Anomaly(anomaly_type="lens_arc", detector="lens", score=4.0),
            Anomaly(anomaly_type="overdensity", detector="distribution", score=3.2),
        ]
        d = result.to_dict()
        assert "anomalies" in d
        assert len(d["anomalies"]) == 2
        assert d["anomalies"][0]["anomaly_type"] == "lens_arc"
        assert d["anomalies"][1]["score"] == 3.2


class TestExtractAnomalies:
    """Test _extract_anomalies() function."""

    def test_empty_detection(self):
        from star_pattern.pipeline.autonomous import _extract_anomalies
        result = _extract_anomalies({}, None)
        assert result == []

    def test_lens_arcs_extracted(self):
        from star_pattern.pipeline.autonomous import _extract_anomalies
        detection = {
            "lens": {
                "central_source": {"x": 128, "y": 128},
                "arcs": [
                    {"radius": 30, "snr": 4.5, "angle_span": 120},
                    {"radius": 15, "snr": 2.1, "angle_span": 60},
                ],
                "rings": [{"radius": 25, "completeness": 0.8}],
            },
        }
        anomalies = _extract_anomalies(detection, None)
        assert len(anomalies) == 3
        types = [a.anomaly_type for a in anomalies]
        assert "lens_arc" in types
        assert "lens_ring" in types
        # All should be from lens detector
        assert all(a.detector == "lens" for a in anomalies)

    def test_catalog_based_detectors(self):
        from star_pattern.pipeline.autonomous import _extract_anomalies
        detection = {
            "kinematic": {
                "comoving_groups": [
                    {"mean_ra": 180.0, "mean_dec": 45.0, "n_members": 7,
                     "mean_pmra": 3.2, "mean_pmdec": -1.1},
                ],
                "stream_candidates": [
                    {"mean_ra": 180.1, "mean_dec": 45.1, "n_members": 12},
                ],
                "runaway_stars": [
                    {"ra": 180.2, "dec": 45.2, "pm_total": 25.3, "deviation_sigma": 4.5},
                ],
            },
            "variability": {
                "variable_candidates": [
                    {"ra": 180.3, "dec": 45.3, "score": 0.85,
                     "classification": "eclipsing_binary"},
                ],
                "periodic_candidates": [
                    {"ra": 180.4, "dec": 45.4, "period": 2.34, "power": 0.92},
                ],
            },
        }
        anomalies = _extract_anomalies(detection, None)
        assert len(anomalies) == 5
        types = {a.anomaly_type for a in anomalies}
        assert "comoving_group" in types
        assert "stellar_stream" in types
        assert "runaway_star" in types
        assert "variable_star" in types
        assert "periodic_variable" in types

        # Check RA/Dec carried through
        comov = [a for a in anomalies if a.anomaly_type == "comoving_group"][0]
        assert comov.sky_ra == 180.0
        assert comov.sky_dec == 45.0
        assert comov.properties["n_members"] == 7

    def test_mixed_detectors(self):
        from star_pattern.pipeline.autonomous import _extract_anomalies
        detection = {
            "lens": {
                "arcs": [{"radius": 20, "snr": 3.0}],
                "rings": [],
            },
            "distribution": {
                "overdensities": [
                    {"x": 50, "y": 50, "sigma": 3.8, "radius_px": 20, "n_sources": 12},
                ],
            },
            "galaxy": {
                "tidal_features": [
                    {"x": 100, "y": 100, "snr": 2.1, "length": 45},
                ],
                "merger_nuclei": [],
            },
            "sersic": {
                "residual_features": [
                    {"x": 60, "y": 60, "snr": 1.5, "extent_px": 30},
                ],
            },
        }
        anomalies = _extract_anomalies(detection, None)
        assert len(anomalies) == 4
        types = {a.anomaly_type for a in anomalies}
        assert types == {"lens_arc", "overdensity", "tidal_feature", "sersic_residual"}
        # Should be sorted by score descending
        scores = [a.score for a in anomalies]
        assert scores == sorted(scores, reverse=True)

    def test_cap_at_max(self):
        from star_pattern.pipeline.autonomous import (
            _extract_anomalies, _MAX_PER_DETECTOR,
        )
        detection = {
            "distribution": {
                "overdensities": [
                    {"x": i, "y": i, "sigma": float(i)}
                    for i in range(60)
                ],
            },
        }
        anomalies = _extract_anomalies(detection, None)
        # Single detector is capped at _MAX_PER_DETECTOR
        assert len(anomalies) == _MAX_PER_DETECTOR

    def test_classical_arcs_use_center_keys(self):
        from star_pattern.pipeline.autonomous import _extract_anomalies
        detection = {
            "classical": {
                "hough_arcs": [
                    {"center_x": 120, "center_y": 80, "radius": 25, "strength": 3.1},
                ],
            },
        }
        anomalies = _extract_anomalies(detection, None)
        assert len(anomalies) == 1
        a = anomalies[0]
        assert a.anomaly_type == "classical_arc"
        assert a.pixel_x == 120
        assert a.pixel_y == 80
        # Score is normalized to [0, 1] within detector (3.1/3.1 = 1.0)
        assert a.score == 1.0

    def test_classical_arcs_skip_missing_coords(self):
        from star_pattern.pipeline.autonomous import _extract_anomalies
        detection = {
            "classical": {
                "hough_arcs": [
                    {"radius": 25, "strength": 3.1},  # No center_x/y
                ],
            },
        }
        anomalies = _extract_anomalies(detection, None)
        assert len(anomalies) == 0  # Skipped, not defaulted to (0,0)

    def test_population_candidates(self):
        from star_pattern.pipeline.autonomous import _extract_anomalies
        detection = {
            "population": {
                "blue_stragglers": {
                    "n_blue_stragglers": 3,
                    "bs_fraction": 0.05,
                    "candidates": [
                        {"color": 0.2, "mag": 18.5, "color_offset": 0.3},
                    ],
                },
                "n_red_giants": 12,
                "red_giants": {
                    "rgb_fraction": 0.15,
                },
            },
        }
        anomalies = _extract_anomalies(detection, None)
        assert len(anomalies) == 2
        types = {a.anomaly_type for a in anomalies}
        assert "blue_straggler" in types
        assert "red_giant" in types
        bs = [a for a in anomalies if a.anomaly_type == "blue_straggler"][0]
        assert bs.properties["n_blue_stragglers"] == 3


class TestReportAnomalyTable:
    """Test that reports enumerate anomalies in table format."""

    def test_anomaly_table_in_report(self, tmp_path):
        report = DiscoveryReport(tmp_path / "reports")
        f = PatternResult(180.0, 45.0, "lens", 0.8, 0.6)
        f.metadata["local_evaluation"] = {
            "verdict": "real",
            "significance_rating": 7,
        }
        f.anomalies = [
            Anomaly(
                anomaly_type="lens_arc", detector="lens",
                pixel_x=100, pixel_y=200,
                sky_ra=192.654, sky_dec=26.609,
                score=4.2, properties={"radius": 15, "angle_span": 120},
            ),
            Anomaly(
                anomaly_type="overdensity", detector="distribution",
                pixel_x=50, pixel_y=50,
                sky_ra=192.658, sky_dec=26.612,
                score=3.8, properties={"radius_px": 20, "n_sources": 12},
            ),
        ]
        path = report.generate_markdown_report(
            [f], run_metadata={"n_regions": 1},
        )
        content = path.read_text()
        assert "Detected anomalies (2 total)" in content
        assert "Lens arc" in content
        assert "Overdensity" in content
        assert "192.654" in content
        assert "distribution" in content

    def test_anomaly_summary_in_report(self, tmp_path):
        report = DiscoveryReport(tmp_path / "reports")
        f1 = PatternResult(180.0, 45.0, "lens", 0.8, 0.6)
        f1.metadata["local_evaluation"] = {
            "verdict": "real", "significance_rating": 7,
        }
        f1.anomalies = [
            Anomaly(anomaly_type="lens_arc", detector="lens", score=4.0),
            Anomaly(anomaly_type="lens_arc", detector="lens", score=3.0),
            Anomaly(anomaly_type="overdensity", detector="distribution", score=3.5),
        ]
        f2 = PatternResult(90.0, 30.0, "galaxy", 0.7, 0.5)
        f2.metadata["local_evaluation"] = {
            "verdict": "real", "significance_rating": 6,
        }
        f2.anomalies = [
            Anomaly(anomaly_type="tidal_feature", detector="galaxy", score=2.0),
        ]
        path = report.generate_markdown_report(
            [f1, f2], run_metadata={"n_regions": 2},
        )
        content = path.read_text()
        assert "Anomaly summary:" in content
        assert "4 individual anomalies" in content

    def test_no_anomaly_table_when_empty(self, tmp_path):
        report = DiscoveryReport(tmp_path / "reports")
        f = PatternResult(180.0, 45.0, "lens", 0.8, 0.6)
        f.metadata["local_evaluation"] = {
            "verdict": "real", "significance_rating": 7,
        }
        # No anomalies set
        path = report.generate_markdown_report([f])
        content = path.read_text()
        assert "Detected anomalies" not in content


class TestMosaicAnomalyCutouts:
    """Test that mosaic shows anomaly cutouts when available."""

    def test_mosaic_with_anomalies(self):
        from star_pattern.visualization.mosaic import create_discovery_mosaic
        import matplotlib.pyplot as plt

        f = PatternResult(180.0, 45.0, "lens", 0.8, 0.6)
        f.metadata["local_evaluation"] = {
            "verdict": "real",
            "significance_rating": 7,
        }
        f.anomalies = [
            Anomaly(
                anomaly_type="lens_arc", detector="lens",
                pixel_x=100, pixel_y=200,
                score=4.2,
            ),
            Anomaly(
                anomaly_type="overdensity", detector="distribution",
                pixel_x=50, pixel_y=50,
                score=3.8,
            ),
        ]
        fig = create_discovery_mosaic([f])
        texts = []
        for ax in fig.get_axes():
            texts.extend([t.get_text() for t in ax.texts])
        # Should have anomaly labels
        assert any("A1" in t for t in texts)
        assert any("A2" in t for t in texts)
        # Should have overview panel
        assert any("Overview" in t for t in texts)
        plt.close(fig)

    def test_mosaic_fallback_no_anomalies(self):
        from star_pattern.visualization.mosaic import create_discovery_mosaic
        import matplotlib.pyplot as plt

        f = PatternResult(180.0, 45.0, "lens", 0.8, 0.6)
        f.metadata["local_evaluation"] = {
            "verdict": "real",
            "significance_rating": 7,
        }
        # No anomalies -- should fall back to legacy behavior
        fig = create_discovery_mosaic([f])
        texts = []
        for ax in fig.get_axes():
            texts.extend([t.get_text() for t in ax.texts])
        assert any("F1" in t for t in texts)
        plt.close(fig)

    def test_mosaic_catalog_only_anomalies(self):
        from star_pattern.visualization.mosaic import create_discovery_mosaic
        import matplotlib.pyplot as plt

        f = PatternResult(180.0, 45.0, "kinematic", 0.7)
        f.metadata["local_evaluation"] = {
            "verdict": "real",
            "significance_rating": 6,
        }
        f.anomalies = [
            Anomaly(
                anomaly_type="comoving_group", detector="kinematic",
                sky_ra=180.0, sky_dec=45.0,
                score=3.5,
                properties={"n_members": 7, "mean_pmra": 3.2, "mean_pmdec": -1.1},
            ),
        ]
        fig = create_discovery_mosaic([f])
        texts = []
        for ax in fig.get_axes():
            texts.extend([t.get_text() for t in ax.texts])
        # Should show text info for catalog-based anomaly
        assert any("A1" in t for t in texts)
        plt.close(fig)
