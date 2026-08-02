"""Multi-detector ensemble scoring."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from star_pattern.core.catalog import StarCatalog
from star_pattern.core.config import DetectionConfig
from star_pattern.core.fits_handler import FITSImage
from star_pattern.detection.anomaly import AnomalyDetector
from star_pattern.detection.base import (
    RECOVERABLE_DETECTOR_EXCEPTIONS,
    DetectionContext,
    DetectorSpec,
)
from star_pattern.detection.classical import ClassicalDetector
from star_pattern.detection.distribution import DistributionAnalyzer
from star_pattern.detection.feature_fusion import FeatureFusionExtractor
from star_pattern.detection.galaxy_detector import GalaxyDetector
from star_pattern.detection.lens_detector import LensDetector
from star_pattern.detection.morphology import MorphologyAnalyzer
from star_pattern.detection.proper_motion import ProperMotionAnalyzer
from star_pattern.detection.sersic import SersicAnalyzer
from star_pattern.detection.source_extraction import SourceExtractor
from star_pattern.detection.stellar_population import StellarPopulationAnalyzer
from star_pattern.detection.temporal import TemporalDetector
from star_pattern.detection.transient import TransientDetector
from star_pattern.detection.variability import VariabilityAnalyzer
from star_pattern.detection.wavelet import WaveletAnalyzer
from star_pattern.utils.logging import get_logger

logger = get_logger("detection.ensemble")


def _summarize_classical(raw: dict[str, Any]) -> dict[str, Any]:
    # Direct indexing, not .get(): a missing key here means the detector
    # changed its contract, and that should surface as an error entry
    # rather than a silent zero. Preserved from the original code.
    return {
        "gabor_score": raw["gabor_score"],
        "fft_score": raw["fft_score"],
        "arc_score": raw["arc_score"],
        "n_arcs": len(raw.get("arcs", [])),
    }


def _summarize_morphology(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "concentration": raw["concentration"],
        "asymmetry": raw["asymmetry"],
        "smoothness": raw["smoothness"],
        "gini": raw["gini"],
        "m20": raw["m20"],
        "morphology_score": raw["morphology_score"],
    }


def _summarize_lens(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "lens_score": raw["lens_score"],
        "n_arcs": len(raw.get("arcs", [])),
        "n_rings": len(raw.get("rings", [])),
        "is_candidate": raw.get("is_candidate", False),
    }


def _summarize_distribution(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "voronoi_cv": raw.get("voronoi_cv", 0),
        "clark_evans_r": raw.get("clark_evans_r", 1.0),
        "n_overdensities": len(raw.get("overdensities", [])),
        "distribution_score": raw.get("distribution_score", 0),
    }


def _summarize_galaxy(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "galaxy_score": raw.get("galaxy_score", 0),
        "n_tidal": len(raw.get("tidal_features", [])),
        "n_mergers": len(raw.get("merger_candidates", [])),
        "n_color_outliers": len(raw.get("color_outliers", [])),
    }


def _summarize_kinematic(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "kinematic_score": raw.get("kinematic_score", 0),
        "n_comoving_groups": len(raw.get("comoving_groups", [])),
        "n_streams": len(raw.get("stream_candidates", [])),
        "n_runaways": len(raw.get("runaway_stars", [])),
    }


def _summarize_transient(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "transient_score": raw.get("transient_score", 0),
        "n_astrometric": len(raw.get("astrometric_outliers", [])),
        "n_photometric": len(raw.get("photometric_outliers", [])),
        "n_parallax": len(raw.get("parallax_anomalies", [])),
    }


def _summarize_sersic(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "sersic_score": raw.get("sersic_score", 0),
        "sersic_n": raw.get("fit", {}).get("n", 0),
        "r_e": raw.get("fit", {}).get("r_e", 0),
        "morphology_class": raw.get("morphology_class", "unknown"),
        "n_residual_features": len(raw.get("residual_features", [])),
    }


def _summarize_wavelet(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "wavelet_score": raw.get("wavelet_score", 0),
        "n_detections": len(raw.get("detections", [])),
        "n_multiscale": len(raw.get("multiscale_objects", [])),
        "mean_scale": raw.get("mean_scale", 0),
    }


def _summarize_population(raw: dict[str, Any]) -> dict[str, Any]:
    """Population needs bespoke nested extraction, so it owns its details."""
    entry: dict[str, Any] = {
        "population_score": raw.get("population_score", 0),
        "n_photometric": raw.get("n_photometric", 0),
        "n_blue_stragglers": raw.get("blue_stragglers", {}).get("n_blue_stragglers", 0),
        "n_red_giants": raw.get("red_giants", {}).get("n_red_giants", 0),
        "multiple_populations": raw.get("multiple_populations", {}).get(
            "is_multiple_population", False
        ),
    }
    blue = raw.get("blue_stragglers", {})
    if blue.get("candidates"):
        entry["blue_straggler_candidates"] = _to_serializable(blue["candidates"])
    red = raw.get("red_giants", {})
    if red.get("candidates"):
        entry["red_giant_candidates"] = _to_serializable(red["candidates"])
    cmd = raw.get("cmd_density", {})
    if cmd.get("peaks"):
        entry["cmd_peaks"] = _to_serializable(cmd["peaks"])
    if "turnoff" in raw:
        entry["turnoff"] = _to_serializable(raw["turnoff"])
    if "tip_rgb" in raw:
        entry["tip_rgb"] = _to_serializable(raw["tip_rgb"])
    return entry


def _summarize_variability(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "variability_score": raw.get("variability_score", 0),
        "n_variables": len(raw.get("variable_candidates", [])),
        "n_periodic": len(raw.get("periodic_candidates", [])),
        "n_transients": len(raw.get("transient_candidates", [])),
    }


def _summarize_temporal(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "temporal_score": raw.get("temporal_score", 0),
        "n_epochs_analyzed": raw.get("n_epochs_analyzed", 0),
        "baseline_days": raw.get("baseline_days", 0),
        "n_new_sources": raw.get("n_new_sources", 0),
        "n_disappeared": raw.get("n_disappeared", 0),
        "n_brightenings": raw.get("n_brightenings", 0),
        "n_fadings": raw.get("n_fadings", 0),
        "n_moving": raw.get("n_moving", 0),
    }


def _needs_catalog(ctx: DetectionContext) -> str | None:
    return None if ctx.catalog is not None else "no_catalog"


def _to_serializable(obj: Any) -> Any:
    """Convert numpy arrays and types to JSON-safe Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


class EnsembleDetector:
    """Combine multiple detectors into a single scoring pipeline."""

    def __init__(
        self,
        config: DetectionConfig | None = None,
        meta_detector: Any | None = None,
    ):
        self.config = config or DetectionConfig()
        self._feature_fusion = FeatureFusionExtractor()
        self._meta_detector = meta_detector
        self.classical = ClassicalDetector(
            gabor_frequencies=self.config.gabor_frequencies,
            gabor_orientations=self.config.gabor_orientations,
        )
        self.source_extractor = SourceExtractor(threshold=self.config.source_extraction_threshold)
        self.morphology = MorphologyAnalyzer()
        self.anomaly = AnomalyDetector(contamination=self.config.anomaly_contamination)
        self.lens = LensDetector()
        self.distribution = DistributionAnalyzer()
        self.galaxy = GalaxyDetector(self.config)
        self.proper_motion = ProperMotionAnalyzer(self.config)
        self.transient = TransientDetector(self.config)
        self.sersic = SersicAnalyzer(
            max_radius_frac=self.config.sersic_max_radius_frac,
            residual_sigma=self.config.sersic_residual_sigma,
        )
        self.wavelet = WaveletAnalyzer(
            n_scales=self.config.wavelet_n_scales,
            significance_threshold=self.config.wavelet_significance,
        )
        self.stellar_population = StellarPopulationAnalyzer(
            ms_width=self.config.population_ms_width,
            blue_straggler_offset=self.config.population_blue_straggler_offset,
        )
        self.variability = VariabilityAnalyzer(self.config)
        self.temporal = TemporalDetector(self.config)

        self._specs = self._build_specs()

    def _is_enabled(self, detector_name: str) -> bool:
        """Check if a detector is enabled via config gates."""
        gates = self.config.enabled_detectors
        if not gates:
            return True  # No gates = all enabled (backward compat)
        return gates.get(detector_name, True)

    def _build_specs(self) -> list[DetectorSpec]:
        """Describe every ensemble member once, in results-dict order.

        The order of this list IS the key order of the results dict, which
        the report and mosaic layers consume as serialized JSON.
        """
        return [
            DetectorSpec(
                name="classical",
                run=lambda ctx: self.classical.detect(ctx.data, pixel_scale_arcsec=ctx.pixel_scale),
                score_key="classical_score",
                weight_name="classical",
                default_weight=0.09,
                summarize=_summarize_classical,
                detail_keys=("arcs",),
                parallel=True,
                count_detections=lambda raw: len(raw.get("arcs", [])),
                feature_index=0,
            ),
            DetectorSpec(
                name="morphology",
                run=lambda ctx: self.morphology.analyze(ctx.data),
                score_key="morphology_score",
                weight_name="morphology",
                default_weight=0.09,
                summarize=_summarize_morphology,
                parallel=True,
                feature_index=1,
            ),
            DetectorSpec(
                name="lens",
                run=lambda ctx: self.lens.detect(ctx.data, pixel_scale_arcsec=ctx.pixel_scale),
                score_key="lens_score",
                weight_name="lens",
                default_weight=0.09,
                summarize=_summarize_lens,
                detail_keys=("central_source", "arcs", "rings"),
                parallel=True,
                count_detections=lambda raw: len(raw.get("arcs", [])),
                feature_index=2,
            ),
            DetectorSpec(
                name="distribution",
                run=lambda ctx: self.distribution.analyze(
                    ctx.positions, boundary=ctx.data.shape[::-1]
                ),
                score_key="distribution_score",
                weight_name="distribution",
                default_weight=0.11,
                summarize=_summarize_distribution,
                detail_keys=("overdensities",),
                precondition=lambda ctx: (
                    None
                    if ctx.positions is not None and len(ctx.positions) >= 10
                    else "n_sources_too_few"
                ),
                count_detections=lambda raw: len(raw.get("overdensities", [])),
                feature_index=3,
            ),
            DetectorSpec(
                name="galaxy",
                run=lambda ctx: self.galaxy.detect(
                    ctx.data, catalog=ctx.catalog, pixel_scale_arcsec=ctx.pixel_scale
                ),
                score_key="galaxy_score",
                weight_name="galaxy",
                default_weight=0.09,
                summarize=_summarize_galaxy,
                detail_keys=("tidal_features", "merger_candidates", "color_outliers"),
                count_detections=lambda raw: raw.get("n_detections", 0),
                feature_index=4,
            ),
            DetectorSpec(
                name="kinematic",
                run=lambda ctx: self.proper_motion.analyze(ctx.catalog),
                score_key="kinematic_score",
                weight_name="kinematic",
                default_weight=0.09,
                summarize=_summarize_kinematic,
                detail_keys=("comoving_groups", "stream_candidates", "runaway_stars"),
                precondition=_needs_catalog,
                count_detections=lambda raw: raw.get("n_detections", 0),
                feature_index=5,
            ),
            DetectorSpec(
                name="transient",
                run=lambda ctx: self.transient.analyze(ctx.catalog),
                score_key="transient_score",
                weight_name="transient",
                default_weight=0.04,
                summarize=_summarize_transient,
                detail_keys=(
                    "astrometric_outliers",
                    "photometric_outliers",
                    "parallax_anomalies",
                ),
                precondition=_needs_catalog,
                count_detections=lambda raw: raw.get("n_detections", 0),
                feature_index=6,
            ),
            DetectorSpec(
                name="sersic",
                run=lambda ctx: self.sersic.analyze(ctx.data, pixel_scale_arcsec=ctx.pixel_scale),
                score_key="sersic_score",
                weight_name="sersic",
                default_weight=0.07,
                summarize=_summarize_sersic,
                detail_keys=(
                    "fit",
                    "radial_profile",
                    "residual_features",
                    "ellipticity",
                    "position_angle",
                ),
                parallel=True,
                feature_index=7,
            ),
            DetectorSpec(
                name="wavelet",
                run=lambda ctx: self.wavelet.analyze(ctx.data, pixel_scale_arcsec=ctx.pixel_scale),
                score_key="wavelet_score",
                weight_name="wavelet",
                default_weight=0.09,
                summarize=_summarize_wavelet,
                detail_keys=("detections", "multiscale_objects", "scale_spectrum"),
                parallel=True,
                feature_index=8,
            ),
            DetectorSpec(
                name="population",
                run=lambda ctx: self.stellar_population.analyze(ctx.catalog),
                score_key="population_score",
                weight_name="population",
                default_weight=0.06,
                summarize=_summarize_population,
                precondition=_needs_catalog,
                feature_index=9,
            ),
            DetectorSpec(
                name="variability",
                run=lambda ctx: self.variability.analyze(ctx.catalog),
                score_key="variability_score",
                weight_name="variability",
                default_weight=0.09,
                summarize=_summarize_variability,
                detail_keys=(
                    "variable_candidates",
                    "periodic_candidates",
                    "transient_candidates",
                ),
                precondition=_needs_catalog,
                count_detections=lambda raw: len(raw.get("variable_candidates", [])),
                feature_index=10,
            ),
            DetectorSpec(
                name="temporal",
                run=lambda ctx: self.temporal.analyze(
                    ctx.temporal_images, pixel_scale_arcsec=ctx.pixel_scale
                ),
                score_key="temporal_score",
                weight_name="temporal",
                default_weight=0.08,
                summarize=_summarize_temporal,
                detail_keys=(
                    "new_sources",
                    "disappeared",
                    "brightenings",
                    "fadings",
                    "moving_objects",
                ),
                # No enable gene exists for temporal; see discovery/genome.py.
                gated=False,
                precondition=lambda ctx: (
                    None
                    if ctx.temporal_images and len(ctx.temporal_images) >= 2
                    else "no_temporal_images"
                ),
                count_detections=lambda raw: (
                    raw.get("n_new_sources", 0)
                    + raw.get("n_disappeared", 0)
                    + raw.get("n_brightenings", 0)
                    + raw.get("n_moving", 0)
                ),
                feature_index=11,
            ),
        ]

    def _skip_marker(self, spec: DetectorSpec, ctx: DetectionContext) -> str | None:
        """Return the reason this detector must not run, or None to run."""
        if spec.gated and not self._is_enabled(spec.name):
            return "disabled"
        return spec.precondition(ctx)

    def _collect(
        self,
        spec: DetectorSpec,
        ctx: DetectionContext,
        futures: dict[str, Any],
        results: dict[str, Any],
        marker: str | None,
    ) -> dict[str, Any]:
        """Run or collect one detector, fill its results entry, return its raw output."""
        if marker is not None:
            results[spec.name] = {marker: True}
            return {spec.score_key: 0}

        try:
            raw = futures[spec.name].result() if spec.parallel else spec.run(ctx)
            entry = spec.summarize(raw)
            for key in spec.detail_keys:
                if raw.get(key):
                    entry[key] = _to_serializable(raw[key])
            for key in spec.detail_keys_if_present:
                if key in raw:
                    entry[key] = _to_serializable(raw[key])
            results[spec.name] = entry
            return raw
        except RECOVERABLE_DETECTOR_EXCEPTIONS as e:
            logger.warning("%s detector failed (%s): %s", spec.name, type(e).__name__, e)
            logger.debug("%s traceback", spec.name, exc_info=True)
            results[spec.name] = {"error": str(e)}
            return {spec.score_key: 0}

    def _run_source_extraction(
        self, ctx: DetectionContext, results: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract sources first: distribution needs the positions.

        Kept out of the registry loop because its result entry uses
        presence rather than truthiness to decide what to include. The
        values are numpy arrays, and truth-testing an array with more than
        one element raises.
        """
        try:
            sources = self.source_extractor.extract(ctx.data)
            entry: dict[str, Any] = {
                "n_sources": sources["n_sources"],
                "background_rms": sources.get("background_rms", 0),
            }
            if "positions" in sources:
                entry["positions"] = _to_serializable(sources["positions"])
            for key in ("star_mask", "fluxes", "ellipticity", "fwhm"):
                if key in sources:
                    entry[key] = _to_serializable(sources[key])
            results["sources"] = entry
            return sources
        except RECOVERABLE_DETECTOR_EXCEPTIONS as e:
            logger.warning("Source extraction failed (%s): %s", type(e).__name__, e)
            logger.debug("Source extraction traceback", exc_info=True)
            results["sources"] = {"n_sources": 0, "error": str(e)}
            return {"positions": np.empty((0, 2)), "n_sources": 0}

    def detect(
        self,
        image: FITSImage,
        catalog: StarCatalog | None = None,
        temporal_images: list | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run all detectors and produce ensemble scores.

        Args:
            image: FITSImage to analyze.
            catalog: Optional StarCatalog for catalog-based detectors.

        Returns:
            Dict with per-detector results and ensemble anomaly score.
        """
        data = image.data
        pixel_scale = image.pixel_scale()

        n_disabled = sum(
            1
            for name in [
                "classical",
                "morphology",
                "lens",
                "sersic",
                "wavelet",
                "distribution",
                "galaxy",
                "kinematic",
                "transient",
                "population",
                "variability",
                "anomaly",
            ]
            if not self._is_enabled(name)
        )
        disabled_info = f", {n_disabled} disabled" if n_disabled else ""
        logger.info(f"Running ensemble detection on {data.shape}{disabled_info}")

        results: dict[str, Any] = {
            "shape": list(data.shape),
            "pixel_scale_arcsec": pixel_scale,
        }

        # Source extraction (must run first -- distribution needs positions)
        ctx = DetectionContext(
            data=data,
            pixel_scale=pixel_scale,
            catalog=catalog,
            temporal_images=temporal_images,
        )
        sources = self._run_source_extraction(ctx, results)
        ctx.positions = sources.get("positions", np.empty((0, 2)))

        # Decide once which detectors are eligible. The same decision drives
        # dispatch, the results entry, and the score denominator below.
        markers: dict[str, str | None] = {
            spec.name: self._skip_marker(spec, ctx) for spec in self._specs
        }

        # Submit the image-heavy detectors to the pool. scipy and numpy
        # release the GIL in their C paths, so these genuinely overlap.
        futures: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for spec in self._specs:
                if spec.parallel and markers[spec.name] is None:
                    futures[spec.name] = pool.submit(spec.run, ctx)

        # Collect every detector in registry order. That order is the key
        # order of the results dict, which downstream layers read as JSON.
        raw_by_name: dict[str, dict[str, Any]] = {
            spec.name: self._collect(spec, ctx, futures, results, markers[spec.name])
            for spec in self._specs
        }

        n_failed = sum(1 for spec in self._specs if "error" in results[spec.name])
        if n_failed:
            failed = [s.name for s in self._specs if "error" in results[s.name]]
            logger.warning(
                "%d of %d detectors failed: %s",
                n_failed,
                len(self._specs),
                ", ".join(failed),
            )

        # Anomaly detection over the stacked detector scores. This runs
        # regardless of the enable gates: its gene only zeroes the weight.
        detector_scores = np.array(
            [
                raw_by_name[spec.name].get(spec.score_key, 0)
                for spec in sorted(
                    (s for s in self._specs if s.feature_index is not None),
                    key=lambda s: s.feature_index,
                )
            ],
            dtype=np.float64,
        )

        try:
            anomaly_result = self.anomaly.detect(detector_scores.reshape(1, -1))
            anomaly_detector_score = float(anomaly_result.get("mean_anomaly_score", 0))
            results["anomaly"] = {
                "anomaly_score": anomaly_detector_score,
                "n_anomalies": anomaly_result.get("n_anomalies", 0),
            }
        except RECOVERABLE_DETECTOR_EXCEPTIONS as e:
            logger.warning("Anomaly detection failed (%s): %s", type(e).__name__, e)
            anomaly_detector_score = 0.0
            results["anomaly"] = {"error": str(e)}

        # Ensemble scoring, renormalized over the detectors that actually
        # ran.
        #
        # The previous version was a weighted sum with no denominator, so a
        # detector that did not run silently contributed zero while still
        # occupying its share of the weight budget. Two consequences:
        # an image with no catalog scored low because five catalog-based
        # detectors were absent, not because the image was uninteresting;
        # and because the GA toggles detectors through the enable genes and
        # then selects on a fixed interest_threshold, any genome that
        # switched a detector off was penalized for a reason unrelated to
        # detection quality. That is selection pressure on the harness, not
        # on the science.
        #
        # Disabled and input-missing detectors leave both sides of the
        # ratio. A detector that ran and FAILED stays in the denominator
        # with a zero score, so a genome cannot improve its fitness by
        # driving a detector into an exception.
        weights = self.config.ensemble_weights
        numerator = 0.0
        denominator = 0.0
        for spec in self._specs:
            if markers[spec.name] is not None:
                continue
            weight = weights.get(spec.weight_name, spec.default_weight)
            numerator += weight * raw_by_name[spec.name].get(spec.score_key, 0)
            denominator += weight

        # The anomaly meta-detector always runs, so it is always a member.
        anomaly_weight = weights.get("anomaly", 0.09)
        numerator += anomaly_weight * anomaly_detector_score
        denominator += anomaly_weight

        results["anomaly_score"] = (
            float(np.clip(numerator / denominator, 0, 1)) if denominator > 1e-10 else 0.0
        )
        results["n_detections"] = sum(
            spec.count_detections(raw_by_name[spec.name]) for spec in self._specs
        )

        # Rich feature extraction (Phase 1)
        try:
            rich_features = self._feature_fusion.extract(results)
            results["rich_features"] = rich_features
        except RECOVERABLE_DETECTOR_EXCEPTIONS as e:
            logger.warning("Feature fusion failed (%s): %s", type(e).__name__, e)

        # Meta-detector scoring (Phase 2)
        if self._meta_detector is not None and "rich_features" in results:
            try:
                meta_result = self._meta_detector.score(
                    results["rich_features"], results["anomaly_score"]
                )
                results["meta_score"] = meta_result["meta_score"]
                results["meta_details"] = meta_result
            except RECOVERABLE_DETECTOR_EXCEPTIONS as e:
                logger.warning("Meta-detector scoring failed (%s): %s", type(e).__name__, e)

        logger.info(
            f"Ensemble score: {results['anomaly_score']:.4f} "
            f"({results['n_detections']} detections)"
        )
        return results

    def detect_batch(
        self,
        images: list[FITSImage],
        catalogs: list[StarCatalog | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Run detection on a batch of images."""
        results = []
        for i, img in enumerate(images):
            logger.info(f"Processing image {i + 1}/{len(images)}")
            catalog = catalogs[i] if catalogs and i < len(catalogs) else None
            results.append(self.detect(img, catalog=catalog))
        return results
