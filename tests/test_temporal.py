"""Tests for temporal change detection via multi-epoch image differencing.

Uses synthetic images (no network calls) to verify detection of new sources,
disappeared sources, brightenings, moving objects, and reference construction.
Also includes ZTF integration tests (skipped when IRSA is unavailable) and
genome compatibility tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io.fits import Header
from astropy.wcs import WCS

from star_pattern.core.config import DetectionConfig, PipelineConfig, TemporalConfig
from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.sky_region import EpochImage, RegionData, SkyRegion
from star_pattern.detection.temporal import TemporalDetector
from star_pattern.discovery.genome import DetectionGenome, GENE_DEFINITIONS


# --- Helpers ---

def _make_wcs(crval_ra: float = 180.0, crval_dec: float = 45.0,
              cdelt: float = -0.0002777, naxis: int = 200) -> WCS:
    """Create a simple TAN WCS centered on (crval_ra, crval_dec)."""
    header = Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = naxis
    header["NAXIS2"] = naxis
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = naxis / 2
    header["CRPIX2"] = naxis / 2
    header["CRVAL1"] = crval_ra
    header["CRVAL2"] = crval_dec
    header["CDELT1"] = cdelt  # ~1 arcsec/pixel
    header["CDELT2"] = abs(cdelt)
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    return WCS(header)


def _make_fits(data: np.ndarray, wcs: WCS | None = None) -> FITSImage:
    """Create a FITSImage from array + optional WCS."""
    img = FITSImage.__new__(FITSImage)
    img.data = data.astype(np.float64)
    img.header = Header()
    img.wcs = wcs
    img._file_path = None
    return img


def _add_gaussian(data: np.ndarray, cx: float, cy: float,
                  amplitude: float = 100.0, sigma: float = 3.0) -> None:
    """Add a 2D Gaussian source to an image (in-place)."""
    ny, nx = data.shape
    y, x = np.mgrid[0:ny, 0:nx]
    gauss = amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    data += gauss


def _make_epoch(data: np.ndarray, mjd: float, wcs: WCS | None = None,
                band: str = "r") -> EpochImage:
    """Create an EpochImage from array."""
    return EpochImage(
        image=_make_fits(data, wcs),
        mjd=mjd,
        band=band,
        source="test",
    )


def _make_base_sky(shape: tuple[int, int] = (200, 200),
                   n_stars: int = 20, seed: int = 42) -> np.ndarray:
    """Create a base sky with random stars and noise."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 1.0, shape)
    for _ in range(n_stars):
        cx = rng.uniform(20, shape[1] - 20)
        cy = rng.uniform(20, shape[0] - 20)
        _add_gaussian(data, cx, cy, amplitude=rng.uniform(30, 200), sigma=2.5)
    return data


# --- Synthetic Image Tests ---

class TestTemporalDetectorSynthetic:
    """Tests using synthetic multi-epoch images (no network)."""

    def test_graceful_degradation_single_epoch(self):
        """Single epoch returns score=0, no crash."""
        detector = TemporalDetector()
        data = _make_base_sky()
        epochs = [_make_epoch(data, mjd=59000.0)]
        result = detector.analyze(epochs)
        assert result["temporal_score"] == 0

    def test_graceful_degradation_empty(self):
        """Empty list returns score=0."""
        detector = TemporalDetector()
        result = detector.analyze([])
        assert result["temporal_score"] == 0

    def test_detect_new_source(self):
        """A source added in epoch 2 should be detected as new/brightening."""
        wcs = _make_wcs()
        base = _make_base_sky()

        epoch1 = _make_epoch(base.copy(), mjd=59000.0, wcs=wcs)

        data2 = base.copy()
        _add_gaussian(data2, 150, 150, amplitude=500, sigma=3.0)
        epoch2 = _make_epoch(data2, mjd=59100.0, wcs=wcs)

        detector = TemporalDetector(DetectionConfig(temporal_snr_threshold=3.0))
        result = detector.analyze([epoch1, epoch2])

        assert result["temporal_score"] > 0
        # Should detect the new source (either as new_source or brightening)
        total_detections = (
            result.get("n_new_sources", 0) + result.get("n_brightenings", 0)
        )
        assert total_detections > 0

    def test_detect_disappeared_source(self):
        """A source removed in epoch 2 should be detected as disappeared/fading."""
        wcs = _make_wcs()
        base = _make_base_sky()

        data1 = base.copy()
        _add_gaussian(data1, 100, 100, amplitude=500, sigma=3.0)
        epoch1 = _make_epoch(data1, mjd=59000.0, wcs=wcs)

        epoch2 = _make_epoch(base.copy(), mjd=59100.0, wcs=wcs)

        detector = TemporalDetector(DetectionConfig(temporal_snr_threshold=3.0))
        result = detector.analyze([epoch1, epoch2])

        assert result["temporal_score"] > 0
        total_detections = (
            result.get("n_disappeared", 0) + result.get("n_fadings", 0)
        )
        assert total_detections > 0

    def test_detect_brightening(self):
        """A source that increases in flux across epochs should be detected."""
        wcs = _make_wcs()
        base = _make_base_sky()

        data1 = base.copy()
        _add_gaussian(data1, 120, 120, amplitude=50, sigma=3.0)
        epoch1 = _make_epoch(data1, mjd=59000.0, wcs=wcs)

        data2 = base.copy()
        _add_gaussian(data2, 120, 120, amplitude=500, sigma=3.0)
        epoch2 = _make_epoch(data2, mjd=59100.0, wcs=wcs)

        data3 = base.copy()
        _add_gaussian(data3, 120, 120, amplitude=800, sigma=3.0)
        epoch3 = _make_epoch(data3, mjd=59200.0, wcs=wcs)

        detector = TemporalDetector(DetectionConfig(temporal_snr_threshold=3.0))
        result = detector.analyze([epoch1, epoch2, epoch3])

        assert result["temporal_score"] > 0
        assert result.get("n_epochs_analyzed", 0) == 3

    def test_detect_moving_object(self):
        """A source that shifts position between epochs should produce residuals."""
        wcs = _make_wcs()
        base = _make_base_sky()

        data1 = base.copy()
        _add_gaussian(data1, 100, 100, amplitude=400, sigma=3.0)
        epoch1 = _make_epoch(data1, mjd=59000.0, wcs=wcs)

        data2 = base.copy()
        _add_gaussian(data2, 104, 100, amplitude=400, sigma=3.0)
        epoch2 = _make_epoch(data2, mjd=59100.0, wcs=wcs)

        detector = TemporalDetector(DetectionConfig(
            temporal_snr_threshold=3.0,
            temporal_dipole_max_sep=5.0,
        ))
        result = detector.analyze([epoch1, epoch2])

        assert result["temporal_score"] > 0
        # Should detect something -- either moving object or new+disappeared
        total = (
            result.get("n_moving", 0)
            + result.get("n_new_sources", 0)
            + result.get("n_disappeared", 0)
            + result.get("n_brightenings", 0)
        )
        assert total > 0

    def test_reference_rejects_transient(self):
        """Median stack should suppress a single-epoch transient in reference."""
        wcs = _make_wcs()
        base = _make_base_sky()

        epochs = []
        for i in range(5):
            data = base.copy() + np.random.default_rng(i + 10).normal(0, 0.5, base.shape)
            epochs.append(_make_epoch(data, mjd=59000.0 + i * 30, wcs=wcs))

        # Add a bright transient only in epoch 3
        transient_data = base.copy()
        _add_gaussian(transient_data, 80, 80, amplitude=1000, sigma=3.0)
        epochs[2] = _make_epoch(transient_data, mjd=59060.0, wcs=wcs)

        detector = TemporalDetector(DetectionConfig(temporal_snr_threshold=5.0))

        # Build reference and verify the transient is NOT in it
        ref, ref_wcs = detector._build_reference(epochs)
        assert ref is not None

        # The transient position in the reference should not be excessively bright
        # (median of 5 epochs where only 1 has the transient = non-transient value)
        ref_value_at_transient = ref[80, 80]
        epoch3_value = epochs[2].image.data[80, 80]
        assert ref_value_at_transient < epoch3_value * 0.5

    def test_noise_estimation_robust(self):
        """MAD noise estimate should not be biased by real signals."""
        wcs = _make_wcs()
        base = _make_base_sky()

        data1 = base.copy()
        epoch1 = _make_epoch(data1, mjd=59000.0, wcs=wcs)

        data2 = base.copy() + np.random.default_rng(99).normal(0, 2.0, base.shape)
        # Add a very bright real transient
        _add_gaussian(data2, 100, 100, amplitude=5000, sigma=3.0)
        epoch2 = _make_epoch(data2, mjd=59100.0, wcs=wcs)

        detector = TemporalDetector()
        ref, ref_wcs = detector._build_reference([epoch1, epoch2])
        assert ref is not None

        diff, noise = detector._subtract_epoch(epoch2, ref, ref_wcs)
        assert diff is not None
        assert noise > 0

        # Noise should reflect the background, not be dominated by the transient
        # With base noise ~1.0 and added noise ~2.0, MAD should be reasonable
        # (not inflated to thousands by the transient)
        assert noise < 100, f"Noise={noise} is unreasonably high"

    def test_wcs_alignment(self):
        """Epochs with slightly offset WCS should still align via reprojection."""
        try:
            from reproject import reproject_interp  # noqa: F401
        except ImportError:
            pytest.skip("reproject not available")

        wcs1 = _make_wcs(crval_ra=180.0, crval_dec=45.0)
        wcs2 = _make_wcs(crval_ra=180.001, crval_dec=45.001)  # Slight offset

        base = _make_base_sky()
        _add_gaussian(base, 100, 100, amplitude=300, sigma=3.0)

        epoch1 = _make_epoch(base.copy(), mjd=59000.0, wcs=wcs1)
        epoch2 = _make_epoch(base.copy(), mjd=59100.0, wcs=wcs2)

        detector = TemporalDetector(DetectionConfig(temporal_snr_threshold=3.0))
        result = detector.analyze([epoch1, epoch2])

        # With aligned WCS, subtracting identical sky should produce minimal residuals
        # (any detections would be from the WCS offset alignment residuals, not real changes)
        # The key test is that it doesn't crash and produces a result
        assert "temporal_score" in result

    def test_snr_threshold_filtering(self):
        """Higher SNR threshold should detect fewer residuals.

        With a very high threshold, a moderate-amplitude source that would
        be detected at a low threshold should be filtered out.
        """
        wcs = _make_wcs()
        base = _make_base_sky()

        epoch1 = _make_epoch(base.copy(), mjd=59000.0, wcs=wcs)
        data2 = base.copy()
        _add_gaussian(data2, 150, 150, amplitude=50.0, sigma=3.0)
        epoch2 = _make_epoch(data2, mjd=59100.0, wcs=wcs)

        # Low threshold should detect the source
        det_low = TemporalDetector(DetectionConfig(temporal_snr_threshold=3.0))
        result_low = det_low.analyze([epoch1, epoch2])
        total_low = (
            result_low.get("n_new_sources", 0)
            + result_low.get("n_brightenings", 0)
        )

        # Very high threshold should detect fewer or zero
        det_high = TemporalDetector(DetectionConfig(temporal_snr_threshold=50.0))
        result_high = det_high.analyze([epoch1, epoch2])
        total_high = (
            result_high.get("n_new_sources", 0)
            + result_high.get("n_brightenings", 0)
        )

        assert total_high <= total_low

    def test_no_wcs_pixel_fallback(self):
        """Without WCS, detector falls back to direct pixel stacking."""
        base = _make_base_sky()

        epoch1 = _make_epoch(base.copy(), mjd=59000.0, wcs=None)

        data2 = base.copy()
        _add_gaussian(data2, 100, 100, amplitude=500, sigma=3.0)
        epoch2 = _make_epoch(data2, mjd=59100.0, wcs=None)

        detector = TemporalDetector(DetectionConfig(temporal_snr_threshold=3.0))
        result = detector.analyze([epoch1, epoch2])

        assert result["temporal_score"] > 0

    def test_baseline_info(self):
        """Result should include correct baseline in days."""
        wcs = _make_wcs()
        base = _make_base_sky()

        epoch1 = _make_epoch(base.copy(), mjd=59000.0, wcs=wcs)

        data2 = base.copy()
        _add_gaussian(data2, 100, 100, amplitude=500, sigma=3.0)
        epoch2 = _make_epoch(data2, mjd=59365.0, wcs=wcs)

        detector = TemporalDetector(DetectionConfig(temporal_snr_threshold=3.0))
        result = detector.analyze([epoch1, epoch2])

        assert abs(result.get("baseline_days", 0) - 365.0) < 0.1


class TestEnsembleTemporalIntegration:
    """Test that temporal results flow through ensemble scoring."""

    def test_ensemble_with_temporal(self):
        """EnsembleDetector accepts temporal_images and includes temporal results."""
        from star_pattern.detection.ensemble import EnsembleDetector

        config = DetectionConfig(temporal_snr_threshold=3.0)
        detector = EnsembleDetector(config)

        wcs = _make_wcs()
        base = _make_base_sky()
        image = _make_fits(base, wcs)

        # Create epoch images with a new source in epoch 2
        data2 = base.copy()
        _add_gaussian(data2, 150, 150, amplitude=500, sigma=3.0)
        temporal_imgs = [
            _make_epoch(base.copy(), mjd=59000.0, wcs=wcs),
            _make_epoch(data2, mjd=59100.0, wcs=wcs),
        ]

        result = detector.detect(image, temporal_images=temporal_imgs)

        assert "temporal" in result
        assert "temporal_score" in result["temporal"]

    def test_ensemble_without_temporal(self):
        """EnsembleDetector works normally without temporal images."""
        from star_pattern.detection.ensemble import EnsembleDetector

        detector = EnsembleDetector()
        base = _make_base_sky()
        image = _make_fits(base)

        result = detector.detect(image)

        assert "temporal" in result
        assert result["temporal"].get("no_temporal_images", False)


class TestAnomalyExtraction:
    """Test that temporal findings produce correct Anomaly objects."""

    def test_anomaly_extraction_from_detection(self):
        """_extract_anomalies produces temporal Anomaly objects."""
        from star_pattern.pipeline.autonomous import _extract_anomalies

        detection = {
            "temporal": {
                "temporal_score": 0.6,
                "n_new_sources": 1,
                "n_disappeared": 1,
                "n_brightenings": 0,
                "n_fadings": 0,
                "n_moving": 1,
                "new_sources": [{
                    "sky_ra": 180.1, "sky_dec": 45.1,
                    "cx": 100, "cy": 100, "peak_snr": 8.5,
                    "n_epochs_detected": 1,
                }],
                "disappeared": [{
                    "sky_ra": 180.2, "sky_dec": 45.2,
                    "cx": 50, "cy": 50, "peak_snr": 6.0,
                    "n_epochs_detected": 1,
                }],
                "brightenings": [],
                "fadings": [],
                "moving_objects": [{
                    "sky_ra": 180.3, "sky_dec": 45.3,
                    "cx": 75, "cy": 75, "peak_snr": 10.0,
                    "n_epochs_detected": 2,
                }],
            },
            # Minimal other detector results to avoid errors
            "sources": {"n_sources": 0},
            "lens": {},
            "wavelet": {},
            "galaxy": {},
            "distribution": {},
            "classical": {},
            "sersic": {},
            "kinematic": {},
            "transient": {},
            "variability": {},
            "population": {},
        }

        anomalies = _extract_anomalies(detection, image=None)

        temporal_anomalies = [a for a in anomalies if a.detector == "temporal"]
        assert len(temporal_anomalies) == 3

        types = {a.anomaly_type for a in temporal_anomalies}
        assert "temporal_new_source" in types
        assert "temporal_disappeared" in types
        assert "temporal_moving" in types


# --- Genome Tests ---

class TestGenomeTemporalGenes:
    """Test that temporal genes are correctly integrated."""

    def test_genome_has_72_genes(self):
        """Gene count expanded from 60 to 72 (12 enable gates added)."""
        assert len(GENE_DEFINITIONS) == 72

    def test_temporal_genes_exist(self):
        """All 6 temporal genes are defined."""
        gene_names = {g.name for g in GENE_DEFINITIONS}
        expected = {
            "temporal_snr_threshold",
            "temporal_min_epochs",
            "temporal_max_baseline",
            "temporal_min_baseline",
            "temporal_dipole_max_sep",
            "weight_temporal",
        }
        assert expected.issubset(gene_names)

    def test_old_genome_compatibility(self):
        """A 54-gene genome loads correctly and pads new genes."""
        old_genes = np.random.default_rng(42).random(54)
        genome = DetectionGenome.from_dict({"genes": old_genes.tolist()})
        assert genome.n_genes == 72
        assert len(genome.genes) == 72
        # First 54 genes should be preserved
        np.testing.assert_array_almost_equal(genome.genes[:54], old_genes)

    def test_temporal_in_detection_config(self):
        """to_detection_config includes temporal section."""
        genome = DetectionGenome()
        config = genome.to_detection_config()
        assert "temporal" in config
        assert "snr_threshold" in config["temporal"]
        assert "min_epochs" in config["temporal"]

    def test_temporal_weight_in_ensemble(self):
        """Ensemble weights include temporal."""
        genome = DetectionGenome()
        config = genome.to_detection_config()
        assert "temporal" in config["ensemble_weights"]


# --- Config Tests ---

class TestTemporalConfig:
    """Test TemporalConfig integration."""

    def test_pipeline_config_has_temporal(self):
        """PipelineConfig includes temporal field."""
        config = PipelineConfig()
        assert hasattr(config, "temporal")
        assert isinstance(config.temporal, TemporalConfig)

    def test_temporal_config_defaults(self):
        """TemporalConfig has sensible defaults."""
        tc = TemporalConfig()
        assert tc.enabled is True
        assert tc.max_epochs == 10
        assert tc.min_epochs == 2
        assert tc.snr_threshold == 5.0

    def test_pipeline_config_from_dict(self):
        """PipelineConfig.from_dict handles temporal section."""
        d = {
            "temporal": {
                "enabled": False,
                "max_epochs": 5,
                "snr_threshold": 8.0,
            }
        }
        config = PipelineConfig.from_dict(d)
        assert config.temporal.enabled is False
        assert config.temporal.max_epochs == 5
        assert config.temporal.snr_threshold == 8.0

    def test_detection_config_temporal_fields(self):
        """DetectionConfig has temporal fields."""
        dc = DetectionConfig()
        assert dc.temporal_snr_threshold == 5.0
        assert dc.temporal_min_epochs == 2
        assert dc.temporal_dipole_max_sep == 5.0


# --- Data Types Tests ---

class TestEpochImageAndRegionData:
    """Test EpochImage dataclass and RegionData temporal support."""

    def test_epoch_image_creation(self):
        """EpochImage stores image, mjd, band correctly."""
        data = np.zeros((100, 100))
        img = _make_fits(data)
        epoch = EpochImage(image=img, mjd=59000.0, band="r", source="ztf")
        assert epoch.mjd == 59000.0
        assert epoch.band == "r"
        assert epoch.source == "ztf"

    def test_region_data_has_temporal(self):
        """RegionData has temporal_images field."""
        rd = RegionData(region=SkyRegion(ra=180.0, dec=45.0, radius=3.0))
        assert rd.temporal_images == {}
        assert rd.has_temporal_images() is False

    def test_region_data_has_temporal_images_true(self):
        """has_temporal_images returns True when multi-epoch data exists."""
        rd = RegionData(region=SkyRegion(ra=180.0, dec=45.0, radius=3.0))
        data = np.zeros((100, 100))
        img = _make_fits(data)
        rd.temporal_images["r"] = [
            EpochImage(image=img, mjd=59000.0, band="r"),
            EpochImage(image=img, mjd=59100.0, band="r"),
        ]
        assert rd.has_temporal_images() is True


# --- Feature Fusion Tests ---

class TestFeatureFusionTemporal:
    """Test that temporal features are in the feature schema."""

    def test_temporal_features_in_schema(self):
        """Feature schema includes temporal features."""
        from star_pattern.detection.feature_fusion import FeatureFusionExtractor

        ff = FeatureFusionExtractor()
        names = ff.feature_names
        assert "temporal.temporal_score" in names
        assert "temporal.n_new_sources" in names
        assert "temporal.n_disappeared" in names
        assert "temporal.n_brightenings" in names
        assert "temporal.n_moving" in names
        assert "temporal.baseline_days" in names

    def test_temporal_features_extracted(self):
        """Temporal features are correctly extracted from detection dict."""
        from star_pattern.detection.feature_fusion import FeatureFusionExtractor

        ff = FeatureFusionExtractor()
        detection = {
            "temporal": {
                "temporal_score": 0.75,
                "n_new_sources": 2,
                "n_disappeared": 1,
                "n_brightenings": 3,
                "n_moving": 1,
                "baseline_days": 365.0,
            },
        }
        features = ff.extract(detection)
        # Find the temporal_score index
        idx = ff.feature_names.index("temporal.temporal_score")
        assert features[idx] == 0.75
        idx_new = ff.feature_names.index("temporal.n_new_sources")
        assert features[idx_new] == 2


# --- ZTF Integration Tests (network required) ---

class TestZTFEpochImages:
    """ZTF epoch image fetch tests (skipped when IRSA unavailable)."""

    def _get_ztf_source(self):
        """Get a ZTF data source, skip if unavailable."""
        try:
            from star_pattern.data.ztf import ZTFDataSource
            from star_pattern.data.cache import DataCache
            import tempfile
            cache = DataCache(tempfile.mkdtemp())
            ztf = ZTFDataSource(cache=cache)
            return ztf
        except Exception:
            pytest.skip("ZTF/IRSA not available")

    def test_ztf_epoch_search(self):
        """IBE metadata query returns results for a well-observed field."""
        ztf = self._get_ztf_source()
        region = SkyRegion(ra=150.0, dec=2.0, radius=2.0)
        try:
            result = ztf.fetch_epoch_images(region, bands=["r"], max_epochs=3)
        except Exception:
            pytest.skip("IRSA IBE unavailable")

        if not result:
            pytest.skip("No ZTF epoch images found (may be IRSA downtime)")
        assert "r" in result
        assert len(result["r"]) >= 1

    def test_ztf_cutout_is_valid_fits(self):
        """Downloaded cutout is a valid FITS image."""
        ztf = self._get_ztf_source()
        region = SkyRegion(ra=150.0, dec=2.0, radius=2.0)
        try:
            result = ztf.fetch_epoch_images(region, bands=["r"], max_epochs=2)
        except Exception:
            pytest.skip("IRSA IBE unavailable")

        if not result or "r" not in result or not result["r"]:
            pytest.skip("No ZTF epoch images found")

        epoch = result["r"][0]
        assert epoch.image.data is not None
        assert epoch.image.data.shape[0] > 0
        assert epoch.mjd > 0

    def test_epoch_caching(self):
        """Second fetch for same region hits cache."""
        ztf = self._get_ztf_source()
        region = SkyRegion(ra=150.0, dec=2.0, radius=2.0)
        try:
            result1 = ztf.fetch_epoch_images(region, bands=["r"], max_epochs=2)
        except Exception:
            pytest.skip("IRSA IBE unavailable")

        if not result1:
            pytest.skip("No ZTF epoch images found")

        initial_cache_size = ztf._cache.size
        result2 = ztf.fetch_epoch_images(region, bands=["r"], max_epochs=2)

        # Cache should not have grown (all hits)
        assert ztf._cache.size == initial_cache_size


# --- Cache Tests ---

class TestCacheEpochSupport:
    """Test that cache supports epoch parameter."""

    def test_epoch_key_differs(self):
        """Different epochs produce different cache keys."""
        from star_pattern.data.cache import DataCache
        key1 = DataCache._make_key("ztf", 180.0, 45.0, 3.0, band="r", epoch="20200101")
        key2 = DataCache._make_key("ztf", 180.0, 45.0, 3.0, band="r", epoch="20200201")
        key3 = DataCache._make_key("ztf", 180.0, 45.0, 3.0, band="r")
        assert key1 != key2
        assert key1 != key3

    def test_epoch_empty_preserves_key(self):
        """Empty epoch produces same key as no epoch (backward compat)."""
        from star_pattern.data.cache import DataCache
        key1 = DataCache._make_key("ztf", 180.0, 45.0, 3.0, band="r", epoch="")
        key2 = DataCache._make_key("ztf", 180.0, 45.0, 3.0, band="r")
        assert key1 == key2


# --- Presets Test ---

class TestPresetsIncludeTemporal:
    """Test that presets include temporal-focused option."""

    def test_preset_count(self):
        """Presets include the temporal-focused preset (12 total)."""
        from star_pattern.discovery.presets import get_preset_genomes
        presets = get_preset_genomes()
        assert len(presets) == 12

    def test_temporal_preset_has_high_weight(self):
        """Last preset gives temporal the highest raw gene weight."""
        from star_pattern.discovery.presets import get_preset_genomes
        presets = get_preset_genomes(rng=np.random.default_rng(42))
        temporal_preset = presets[-1]
        # Check raw gene value (not normalized) -- weight_temporal should be 0.40
        temporal_weight = temporal_preset.get("weight_temporal")
        assert temporal_weight > 0.3


# --- Local Classifier Tests ---

class TestLocalClassifierTemporal:
    """Test temporal classification mapping."""

    def test_temporal_classification(self):
        """Temporal-dominated detection classifies as temporal_change."""
        from star_pattern.detection.local_classifier import LocalClassifier

        classifier = LocalClassifier()
        detection = {
            "temporal": {"temporal_score": 0.8},
            "classical": {"classical_score": 0.1},
            "morphology": {"morphology_score": 0.1},
            "anomaly": {"anomaly_score": 0.1},
            "lens": {"lens_score": 0.0},
            "distribution": {"distribution_score": 0.1},
            "galaxy": {"galaxy_score": 0.0},
            "kinematic": {"kinematic_score": 0.0},
            "transient": {"transient_score": 0.1},
            "sersic": {"sersic_score": 0.0},
            "wavelet": {"wavelet_score": 0.0},
            "population": {"population_score": 0.0},
            "variability": {"variability_score": 0.1},
        }
        result = classifier.classify(detection)
        assert result["classification"] == "temporal_change"
        assert result["dominant_detector"] == "temporal"


# --- Diagnostic Tests ---

class TestTemporalDiagnostics:
    """Test that diagnostics are stored correctly after analyze()."""

    def test_diagnostics_available_after_analyze(self):
        """After a successful analyze(), diagnostics contains all expected keys."""
        wcs = _make_wcs()
        base = _make_base_sky()

        epoch1 = _make_epoch(base.copy(), mjd=59000.0, wcs=wcs)

        data2 = base.copy()
        _add_gaussian(data2, 150, 150, amplitude=500, sigma=3.0)
        epoch2 = _make_epoch(data2, mjd=59100.0, wcs=wcs)

        detector = TemporalDetector(DetectionConfig(temporal_snr_threshold=3.0))
        result = detector.analyze([epoch1, epoch2])

        assert result["temporal_score"] > 0
        diag = detector.diagnostics
        assert diag is not None

        # Check all expected keys
        assert "reference_image" in diag
        assert "reference_wcs" in diag
        assert "diff_images" in diag
        assert "snr_maps" in diag
        assert "n_residuals_per_epoch" in diag

        # Check shapes and types
        ref = diag["reference_image"]
        assert isinstance(ref, np.ndarray)
        assert ref.ndim == 2
        assert ref.shape == (200, 200)

        # Check diff_images entries
        assert len(diag["diff_images"]) == 2
        for entry in diag["diff_images"]:
            assert "mjd" in entry
            assert "data" in entry
            assert "noise" in entry
            assert isinstance(entry["data"], np.ndarray)
            assert entry["data"].shape == ref.shape
            assert entry["noise"] > 0

        # Check snr_maps entries
        assert len(diag["snr_maps"]) == 2
        for entry in diag["snr_maps"]:
            assert "mjd" in entry
            assert "data" in entry
            assert isinstance(entry["data"], np.ndarray)
            assert entry["data"].shape == ref.shape

        # Check n_residuals_per_epoch
        assert len(diag["n_residuals_per_epoch"]) == 2
        for n in diag["n_residuals_per_epoch"]:
            assert isinstance(n, int)
            assert n >= 0

    def test_diagnostics_none_on_insufficient_epochs(self):
        """Single epoch returns None diagnostics."""
        detector = TemporalDetector()
        data = _make_base_sky()
        epochs = [_make_epoch(data, mjd=59000.0)]
        result = detector.analyze(epochs)
        assert result["temporal_score"] == 0
        assert detector.diagnostics is None

    def test_diagnostics_cleared_between_calls(self):
        """A failed call clears diagnostics from a prior successful call."""
        wcs = _make_wcs()
        base = _make_base_sky()

        # First call: successful (2 epochs)
        epoch1 = _make_epoch(base.copy(), mjd=59000.0, wcs=wcs)
        data2 = base.copy()
        _add_gaussian(data2, 150, 150, amplitude=500, sigma=3.0)
        epoch2 = _make_epoch(data2, mjd=59100.0, wcs=wcs)

        detector = TemporalDetector(DetectionConfig(temporal_snr_threshold=3.0))
        detector.analyze([epoch1, epoch2])
        assert detector.diagnostics is not None

        # Second call: fails (1 epoch)
        detector.analyze([epoch1])
        assert detector.diagnostics is None

    def test_snr_maps_match_diff_images(self):
        """SNR = |diff|/noise consistency check."""
        wcs = _make_wcs()
        base = _make_base_sky()

        epoch1 = _make_epoch(base.copy(), mjd=59000.0, wcs=wcs)
        data2 = base.copy()
        _add_gaussian(data2, 100, 100, amplitude=500, sigma=3.0)
        epoch2 = _make_epoch(data2, mjd=59100.0, wcs=wcs)

        detector = TemporalDetector(DetectionConfig(temporal_snr_threshold=3.0))
        detector.analyze([epoch1, epoch2])

        diag = detector.diagnostics
        assert diag is not None

        for diff_entry, snr_entry in zip(diag["diff_images"], diag["snr_maps"]):
            expected_snr = np.abs(diff_entry["data"]) / diff_entry["noise"]
            np.testing.assert_array_almost_equal(snr_entry["data"], expected_snr)


class TestTemporalOverlay:
    """Test that overlay_temporal_analysis produces a valid figure."""

    def test_overlay_produces_figure(self):
        """overlay_temporal_analysis returns a Figure with 4+ axes."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from star_pattern.visualization.pattern_overlay import overlay_temporal_analysis

        wcs = _make_wcs()
        base = _make_base_sky()

        epoch1 = _make_epoch(base.copy(), mjd=59000.0, wcs=wcs)
        data2 = base.copy()
        _add_gaussian(data2, 150, 150, amplitude=500, sigma=3.0)
        epoch2 = _make_epoch(data2, mjd=59100.0, wcs=wcs)

        detector = TemporalDetector(DetectionConfig(temporal_snr_threshold=3.0))
        result = detector.analyze([epoch1, epoch2])

        assert detector.diagnostics is not None

        try:
            fig = overlay_temporal_analysis(detector.diagnostics, result)
            assert isinstance(fig, Figure)
            # 2x2 grid = 4 axes (plus possible twin axis)
            assert len(fig.axes) >= 4
        finally:
            plt.close("all")
