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
from star_pattern.evaluation.metrics import PatternResult
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
        path = report.generate_text_report(findings)
        assert path.exists()
        content = path.read_text()
        assert "lens" in content
        assert "180.0000" in content

    def test_json_report(self, tmp_path):
        report = DiscoveryReport(tmp_path / "reports")
        findings = [PatternResult(180.0, 45.0, "lens", 0.8)]
        path = report.generate_json_report(findings)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["n_findings"] == 1

    def test_empty_report(self, tmp_path):
        report = DiscoveryReport(tmp_path / "reports")
        path = report.generate_text_report([])
        assert path.exists()

    def test_full_report(self, tmp_path):
        report = DiscoveryReport(tmp_path / "reports")
        findings = [
            PatternResult(180.0, 45.0, "lens", 0.8, 0.6),
        ]
        paths = report.generate_full_report(findings, run_metadata={"run_name": "test"})
        assert "text" in paths
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
        assert "text" in report_paths
        assert "json" in report_paths
        assert report_paths["text"].exists()
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
        path = report.generate_text_report(findings, run_metadata=metadata)
        content = path.read_text()
        assert "EVOLUTION HISTORY" in content
        assert "0.4321" in content
        assert "0.5678" in content
