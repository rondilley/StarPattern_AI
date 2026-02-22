"""Multi-detector ensemble scoring."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from star_pattern.core.catalog import StarCatalog
from star_pattern.core.config import DetectionConfig
from star_pattern.core.fits_handler import FITSImage
from star_pattern.detection.classical import ClassicalDetector
from star_pattern.detection.source_extraction import SourceExtractor
from star_pattern.detection.morphology import MorphologyAnalyzer
from star_pattern.detection.anomaly import AnomalyDetector
from star_pattern.detection.lens_detector import LensDetector
from star_pattern.detection.distribution import DistributionAnalyzer
from star_pattern.detection.galaxy_detector import GalaxyDetector
from star_pattern.detection.proper_motion import ProperMotionAnalyzer
from star_pattern.detection.transient import TransientDetector
from star_pattern.detection.sersic import SersicAnalyzer
from star_pattern.detection.wavelet import WaveletAnalyzer
from star_pattern.detection.stellar_population import StellarPopulationAnalyzer
from star_pattern.detection.variability import VariabilityAnalyzer
from star_pattern.detection.feature_fusion import FeatureFusionExtractor
from star_pattern.utils.logging import get_logger

logger = get_logger("detection.ensemble")


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
        self.source_extractor = SourceExtractor(
            threshold=self.config.source_extraction_threshold
        )
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

    def detect(
        self,
        image: FITSImage,
        catalog: StarCatalog | None = None,
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
        logger.info(f"Running ensemble detection on {data.shape}")

        results: dict[str, Any] = {
            "shape": list(data.shape),
            "pixel_scale_arcsec": pixel_scale,
        }

        # Source extraction (must run first -- distribution needs positions)
        try:
            sources = self.source_extractor.extract(data)
            results["sources"] = {
                "n_sources": sources["n_sources"],
                "background_rms": sources.get("background_rms", 0),
            }
            # Preserve spatial data (positions as list, not raw SEP array)
            if "positions" in sources:
                results["sources"]["positions"] = _to_serializable(
                    sources["positions"]
                )
            for key in ("star_mask", "fluxes", "ellipticity", "fwhm"):
                if key in sources:
                    results["sources"][key] = _to_serializable(sources[key])
        except Exception as e:
            logger.warning(f"Source extraction failed: {e}")
            sources = {"positions": np.empty((0, 2)), "n_sources": 0}
            results["sources"] = {"n_sources": 0, "error": str(e)}

        # Run 4 heavy image-based detectors in parallel threads
        # scipy/numpy release the GIL during C-level computation
        classical: dict[str, Any] = {"classical_score": 0}
        morph: dict[str, Any] = {"morphology_score": 0}
        lens: dict[str, Any] = {"lens_score": 0}

        def _run_classical() -> dict[str, Any]:
            return self.classical.detect(data, pixel_scale_arcsec=pixel_scale)

        def _run_morphology() -> dict[str, Any]:
            return self.morphology.analyze(data)

        def _run_lens() -> dict[str, Any]:
            return self.lens.detect(data, pixel_scale_arcsec=pixel_scale)

        def _run_sersic() -> dict[str, Any]:
            return self.sersic.analyze(data, pixel_scale_arcsec=pixel_scale)

        def _run_wavelet() -> dict[str, Any]:
            return self.wavelet.analyze(data, pixel_scale_arcsec=pixel_scale)

        # Submit heavy detectors to thread pool
        parallel_tasks: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            parallel_tasks["classical"] = pool.submit(_run_classical)
            parallel_tasks["morphology"] = pool.submit(_run_morphology)
            parallel_tasks["lens"] = pool.submit(_run_lens)
            parallel_tasks["sersic"] = pool.submit(_run_sersic)
            parallel_tasks["wavelet"] = pool.submit(_run_wavelet)

        # Collect classical results
        try:
            classical = parallel_tasks["classical"].result()
            results["classical"] = {
                "gabor_score": classical["gabor_score"],
                "fft_score": classical["fft_score"],
                "arc_score": classical["arc_score"],
                "n_arcs": len(classical.get("arcs", [])),
            }
            if classical.get("arcs"):
                results["classical"]["arcs"] = _to_serializable(
                    classical["arcs"]
                )
        except Exception as e:
            logger.warning(f"Classical detection failed: {e}")
            classical = {"classical_score": 0}
            results["classical"] = {"error": str(e)}

        # Collect morphology results
        try:
            morph = parallel_tasks["morphology"].result()
            results["morphology"] = {
                "concentration": morph["concentration"],
                "asymmetry": morph["asymmetry"],
                "smoothness": morph["smoothness"],
                "gini": morph["gini"],
                "m20": morph["m20"],
                "morphology_score": morph["morphology_score"],
            }
        except Exception as e:
            logger.warning(f"Morphology analysis failed: {e}")
            morph = {"morphology_score": 0}
            results["morphology"] = {"error": str(e)}

        # Collect lens results
        try:
            lens = parallel_tasks["lens"].result()
            results["lens"] = {
                "lens_score": lens["lens_score"],
                "n_arcs": len(lens.get("arcs", [])),
                "n_rings": len(lens.get("rings", [])),
                "is_candidate": lens.get("is_candidate", False),
            }
            for key in ("central_source", "arcs", "rings"):
                if lens.get(key):
                    results["lens"][key] = _to_serializable(lens[key])
        except Exception as e:
            logger.warning(f"Lens detection failed: {e}")
            lens = {"lens_score": 0}
            results["lens"] = {"error": str(e)}

        # Distribution analysis (if enough sources)
        positions = sources.get("positions", np.empty((0, 2)))
        if len(positions) >= 10:
            try:
                dist = self.distribution.analyze(positions, boundary=data.shape[::-1])
                results["distribution"] = {
                    "voronoi_cv": dist.get("voronoi_cv", 0),
                    "clark_evans_r": dist.get("clark_evans_r", 1.0),
                    "n_overdensities": len(dist.get("overdensities", [])),
                    "distribution_score": dist.get("distribution_score", 0),
                }
                if dist.get("overdensities"):
                    results["distribution"]["overdensities"] = _to_serializable(
                        dist["overdensities"]
                    )
            except Exception as e:
                logger.warning(f"Distribution analysis failed: {e}")
                dist = {"distribution_score": 0}
                results["distribution"] = {"error": str(e)}
        else:
            dist = {"distribution_score": 0}
            results["distribution"] = {"n_sources_too_few": True}

        # Galaxy detection (image + optional catalog)
        try:
            galaxy = self.galaxy.detect(
                data, catalog=catalog, pixel_scale_arcsec=pixel_scale
            )
            results["galaxy"] = {
                "galaxy_score": galaxy.get("galaxy_score", 0),
                "n_tidal": len(galaxy.get("tidal_features", [])),
                "n_mergers": len(galaxy.get("merger_candidates", [])),
                "n_color_outliers": len(galaxy.get("color_outliers", [])),
            }
            for key in ("tidal_features", "merger_candidates", "color_outliers"):
                if galaxy.get(key):
                    results["galaxy"][key] = _to_serializable(galaxy[key])
        except Exception as e:
            logger.warning(f"Galaxy detection failed: {e}")
            galaxy = {"galaxy_score": 0}
            results["galaxy"] = {"error": str(e)}

        # Proper motion / kinematic analysis (catalog only)
        kinematic: dict[str, Any] = {"kinematic_score": 0}
        if catalog is not None:
            try:
                kinematic = self.proper_motion.analyze(catalog)
                results["kinematic"] = {
                    "kinematic_score": kinematic.get("kinematic_score", 0),
                    "n_comoving_groups": len(kinematic.get("comoving_groups", [])),
                    "n_streams": len(kinematic.get("stream_candidates", [])),
                    "n_runaways": len(kinematic.get("runaway_stars", [])),
                }
                for key in ("comoving_groups", "stream_candidates", "runaway_stars"):
                    if kinematic.get(key):
                        results["kinematic"][key] = _to_serializable(
                            kinematic[key]
                        )
            except Exception as e:
                logger.warning(f"Kinematic analysis failed: {e}")
                kinematic = {"kinematic_score": 0}
                results["kinematic"] = {"error": str(e)}
        else:
            results["kinematic"] = {"no_catalog": True}

        # Transient / variability detection (catalog only)
        transient: dict[str, Any] = {"transient_score": 0}
        if catalog is not None:
            try:
                transient = self.transient.analyze(catalog)
                results["transient"] = {
                    "transient_score": transient.get("transient_score", 0),
                    "n_astrometric": len(transient.get("astrometric_outliers", [])),
                    "n_photometric": len(transient.get("photometric_outliers", [])),
                    "n_parallax": len(transient.get("parallax_anomalies", [])),
                }
                for key in (
                    "astrometric_outliers",
                    "photometric_outliers",
                    "parallax_anomalies",
                ):
                    if transient.get(key):
                        results["transient"][key] = _to_serializable(
                            transient[key]
                        )
            except Exception as e:
                logger.warning(f"Transient detection failed: {e}")
                transient = {"transient_score": 0}
                results["transient"] = {"error": str(e)}
        else:
            results["transient"] = {"no_catalog": True}

        # Sersic profile fitting (galaxy morphology) -- collected from parallel
        sersic: dict[str, Any] = {"sersic_score": 0}
        try:
            sersic = parallel_tasks["sersic"].result()
            results["sersic"] = {
                "sersic_score": sersic.get("sersic_score", 0),
                "sersic_n": sersic.get("fit", {}).get("n", 0),
                "r_e": sersic.get("fit", {}).get("r_e", 0),
                "morphology_class": sersic.get("morphology_class", "unknown"),
                "n_residual_features": len(sersic.get("residual_features", [])),
            }
            for key in (
                "fit", "radial_profile", "residual_features",
                "ellipticity", "position_angle",
            ):
                if key in sersic and sersic[key]:
                    results["sersic"][key] = _to_serializable(sersic[key])
        except Exception as e:
            logger.warning(f"Sersic analysis failed: {e}")
            sersic = {"sersic_score": 0}
            results["sersic"] = {"error": str(e)}

        # Wavelet multi-scale analysis -- collected from parallel
        wavelet: dict[str, Any] = {"wavelet_score": 0}
        try:
            wavelet = parallel_tasks["wavelet"].result()
            results["wavelet"] = {
                "wavelet_score": wavelet.get("wavelet_score", 0),
                "n_detections": len(wavelet.get("detections", [])),
                "n_multiscale": len(wavelet.get("multiscale_objects", [])),
                "mean_scale": wavelet.get("mean_scale", 0),
            }
            for key in ("detections", "multiscale_objects", "scale_spectrum"):
                if wavelet.get(key):
                    results["wavelet"][key] = _to_serializable(wavelet[key])
        except Exception as e:
            logger.warning(f"Wavelet analysis failed: {e}")
            wavelet = {"wavelet_score": 0}
            results["wavelet"] = {"error": str(e)}

        # Stellar population / CMD analysis (catalog only)
        population: dict[str, Any] = {"population_score": 0}
        if catalog is not None:
            try:
                population = self.stellar_population.analyze(catalog)
                results["population"] = {
                    "population_score": population.get("population_score", 0),
                    "n_photometric": population.get("n_photometric", 0),
                    "n_blue_stragglers": population.get(
                        "blue_stragglers", {}
                    ).get("n_blue_stragglers", 0),
                    "n_red_giants": population.get(
                        "red_giants", {}
                    ).get("n_red_giants", 0),
                    "multiple_populations": population.get(
                        "multiple_populations", {}
                    ).get("is_multiple_population", False),
                }
                # CMD summary data (candidates, not raw mask arrays)
                bs = population.get("blue_stragglers", {})
                if bs.get("candidates"):
                    results["population"]["blue_straggler_candidates"] = (
                        _to_serializable(bs["candidates"])
                    )
                rg = population.get("red_giants", {})
                if rg.get("candidates"):
                    results["population"]["red_giant_candidates"] = (
                        _to_serializable(rg["candidates"])
                    )
                cmd = population.get("cmd_density", {})
                if cmd.get("peaks"):
                    results["population"]["cmd_peaks"] = _to_serializable(
                        cmd["peaks"]
                    )
                if "turnoff" in population:
                    results["population"]["turnoff"] = _to_serializable(
                        population["turnoff"]
                    )
                if "tip_rgb" in population:
                    results["population"]["tip_rgb"] = _to_serializable(
                        population["tip_rgb"]
                    )
            except Exception as e:
                logger.warning(f"Population analysis failed: {e}")
                population = {"population_score": 0}
                results["population"] = {"error": str(e)}
        else:
            results["population"] = {"no_catalog": True}

        # Variability / time-domain analysis (catalog only, needs light curves)
        variability: dict[str, Any] = {"variability_score": 0}
        if catalog is not None:
            try:
                variability = self.variability.analyze(catalog)
                results["variability"] = {
                    "variability_score": variability.get("variability_score", 0),
                    "n_variables": len(variability.get("variable_candidates", [])),
                    "n_periodic": len(variability.get("periodic_candidates", [])),
                    "n_transients": len(variability.get("transient_candidates", [])),
                }
                for key in (
                    "variable_candidates",
                    "periodic_candidates",
                    "transient_candidates",
                ):
                    if variability.get(key):
                        results["variability"][key] = _to_serializable(
                            variability[key]
                        )
            except Exception as e:
                logger.warning(f"Variability analysis failed: {e}")
                variability = {"variability_score": 0}
                results["variability"] = {"error": str(e)}
        else:
            results["variability"] = {"no_catalog": True}

        # Anomaly detection: build feature vector from all detector scores
        detector_scores = np.array([
            classical.get("classical_score", 0),
            morph.get("morphology_score", 0),
            lens.get("lens_score", 0),
            dist.get("distribution_score", 0),
            galaxy.get("galaxy_score", 0),
            kinematic.get("kinematic_score", 0),
            transient.get("transient_score", 0),
            sersic.get("sersic_score", 0),
            wavelet.get("wavelet_score", 0),
            population.get("population_score", 0),
            variability.get("variability_score", 0),
        ], dtype=np.float64)

        # Run anomaly detector on stacked feature vectors
        # For single images, uses distance-from-mean fallback
        try:
            features_2d = detector_scores.reshape(1, -1)
            anomaly_result = self.anomaly.detect(features_2d)
            anomaly_detector_score = float(
                anomaly_result.get("mean_anomaly_score", 0)
            )
            results["anomaly"] = {
                "anomaly_score": anomaly_detector_score,
                "n_anomalies": anomaly_result.get("n_anomalies", 0),
            }
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            anomaly_detector_score = 0.0
            results["anomaly"] = {"error": str(e)}

        # Ensemble scoring
        weights = self.config.ensemble_weights
        anomaly_score = (
            weights.get("classical", 0.09) * classical.get("classical_score", 0)
            + weights.get("morphology", 0.09) * morph.get("morphology_score", 0)
            + weights.get("anomaly", 0.09) * anomaly_detector_score
            + weights.get("lens", 0.09) * lens.get("lens_score", 0)
            + weights.get("distribution", 0.11) * dist.get("distribution_score", 0)
            + weights.get("galaxy", 0.09) * galaxy.get("galaxy_score", 0)
            + weights.get("kinematic", 0.09) * kinematic.get("kinematic_score", 0)
            + weights.get("transient", 0.04) * transient.get("transient_score", 0)
            + weights.get("sersic", 0.07) * sersic.get("sersic_score", 0)
            + weights.get("wavelet", 0.09) * wavelet.get("wavelet_score", 0)
            + weights.get("population", 0.06) * population.get("population_score", 0)
            + weights.get("variability", 0.09) * variability.get("variability_score", 0)
        )

        results["anomaly_score"] = float(np.clip(anomaly_score, 0, 1))
        results["n_detections"] = (
            len(classical.get("arcs", []))
            + len(lens.get("arcs", []))
            + len(dist.get("overdensities", []))
            + galaxy.get("n_detections", 0)
            + kinematic.get("n_detections", 0)
            + transient.get("n_detections", 0)
            + len(variability.get("variable_candidates", []))
        )

        # Rich feature extraction (Phase 1)
        try:
            rich_features = self._feature_fusion.extract(results)
            results["rich_features"] = rich_features
        except Exception as e:
            logger.debug(f"Feature fusion failed: {e}")

        # Meta-detector scoring (Phase 2)
        if self._meta_detector is not None and "rich_features" in results:
            try:
                meta_result = self._meta_detector.score(
                    results["rich_features"], results["anomaly_score"]
                )
                results["meta_score"] = meta_result["meta_score"]
                results["meta_details"] = meta_result
            except Exception as e:
                logger.debug(f"Meta-detector scoring failed: {e}")

        logger.info(f"Ensemble score: {results['anomaly_score']:.4f} ({results['n_detections']} detections)")
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
