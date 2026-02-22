"""Autonomous discovery pipeline: main loop with graceful shutdown.

Architecture: LLM-as-strategist with local detection and evaluation.
- Detection cycle (LOCAL, zero tokens): fetch -> detect -> classify -> evaluate
- Strategy session (LLM, ~1,000 tokens, every N cycles): batch review + guidance
- Evolution cycle (LOCAL + LLM-seeded): GA with LLM-informed population variants
"""

from __future__ import annotations

import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from star_pattern.core.config import (
    PipelineConfig, DetectionConfig, EvolutionConfig, SurveyConfig,
)
from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.sky_region import SkyRegion, RegionData
from star_pattern.data.pipeline import DataPipeline
from star_pattern.detection.ensemble import EnsembleDetector
from star_pattern.detection.local_classifier import LocalClassifier
from star_pattern.detection.local_evaluator import LocalEvaluator
from star_pattern.detection.meta_detector import MetaDetector, MetaDetectorConfig
from star_pattern.detection.compositional import ComposedPipeline, OperationSpec
from star_pattern.discovery.evolutionary import EvolutionaryDiscovery
from star_pattern.discovery.genome import DetectionGenome
from star_pattern.evaluation.metrics import PatternResult
from star_pattern.evaluation.cross_reference import CatalogCrossReferencer
from star_pattern.pipeline.active_learning import ActiveLearner
from star_pattern.utils.run_manager import RunManager
from star_pattern.utils.logging import get_logger

logger = get_logger("pipeline.autonomous")


class AutonomousDiscovery:
    """Main autonomous discovery loop.

    Cycle: fetch data -> detect patterns -> local classify -> local evaluate
    -> record -> (periodically) LLM strategy session -> apply strategy

    LLM calls are minimized through:
    - LocalClassifier replaces per-detection LLM hypothesis generation
    - LocalEvaluator replaces per-detection LLM debate
    - StrategyAdvisor reviews batches every strategy_interval cycles
    - TokenTracker enforces session-wide budget
    - LLMCache prevents redundant calls
    """

    def __init__(
        self,
        config: PipelineConfig,
        run_manager: RunManager | None = None,
        use_llm: bool = False,
    ):
        self.config = config
        self.run_manager = run_manager or RunManager(base_dir=config.output_dir)
        self.use_llm = use_llm

        self.data_pipeline = DataPipeline(config.data)

        # Meta-detector (Phase 2: learned non-linear scoring)
        self._meta_detector: MetaDetector | None = None
        if config.meta.enabled:
            meta_config = MetaDetectorConfig(
                blend_weight=config.meta.blend_weight,
                min_samples_gbm=config.meta.min_samples_gbm,
                min_samples_nn=config.meta.min_samples_nn,
                gbm_n_estimators=config.meta.gbm_n_estimators,
                gbm_max_depth=config.meta.gbm_max_depth,
                nn_hidden=list(config.meta.nn_hidden),
            )
            self._meta_detector = MetaDetector(meta_config)

        self.detector = EnsembleDetector(
            config.detection, meta_detector=self._meta_detector
        )
        self.cross_ref = CatalogCrossReferencer()

        # Local processing (zero token cost)
        self.local_classifier = LocalClassifier()
        self.local_evaluator = LocalEvaluator()

        # Representation learning (Phase 3)
        self._repr_manager = None
        if config.representation.enabled:
            try:
                from star_pattern.ml.representation_manager import RepresentationManager
                checkpoint_dir = Path(run_manager.run_dir if run_manager else config.output_dir)
                self._repr_manager = RepresentationManager(
                    config=config.representation,
                    checkpoint_dir=checkpoint_dir,
                )
            except Exception as e:
                logger.debug(f"RepresentationManager init failed: {e}")

        # Compositional detection (Phase 4)
        self._pipeline_genomes: list[Any] = []
        self._best_pipeline: ComposedPipeline | None = None
        if config.compositional.enabled:
            try:
                from star_pattern.discovery.pipeline_presets import get_preset_pipelines
                self._pipeline_genomes = get_preset_pipelines()
            except Exception as e:
                logger.debug(f"Pipeline genome init failed: {e}")

        # LLM components (lazy init)
        self._llm_providers = None
        self._hypothesis_gen = None
        self._search_guide = None
        self._debate = None
        self._consensus = None

        # Strategy components (lazy init)
        self._strategy_advisor = None
        self._token_tracker = None
        self._llm_cache = None
        self._pending_strategy = None
        self._suggested_regions: list[SkyRegion] = []
        self._last_strategy_outcome: dict[str, Any] | None = None
        self._flagged_for_review: list[dict[str, Any]] = []

        # Wide-field pipeline (lazy init)
        self.wide_field_pipeline = None
        self._wide_field_radius: float | None = None

        # HEALPix survey (lazy init)
        self._survey = None

        self.findings: list[PatternResult] = []
        self.searched_regions: list[SkyRegion] = []
        self.cycle: int = 0
        self._shutdown = False

        # Image buffer for evolution (capped at 50)
        self._recent_images: list[FITSImage] = []
        self._current_genome: DetectionGenome | None = None
        self._evolution_history: list[dict] = []
        self._full_generation_histories: list[list[dict]] = []

        # Detection cache: avoids re-running detection in _save_region_summary
        self._last_detection_results: dict[str, dict] = {}
        self._last_classification: dict[str, Any] | None = None
        self._last_evaluation: dict[str, Any] | None = None

        # Active learning
        self.active_learner = ActiveLearner(
            meta_detector=self._meta_detector,
        )

        # Graceful shutdown handler
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def set_wide_field(self, field_radius_arcmin: float) -> None:
        """Enable wide-field mode with the given field radius."""
        from star_pattern.data.wide_field import WideFieldPipeline

        self._wide_field_radius = field_radius_arcmin
        self.wide_field_pipeline = WideFieldPipeline(self.config)
        logger.info(
            f"Wide-field mode enabled: {field_radius_arcmin}' radius"
        )

    def set_survey(self, config: SurveyConfig) -> None:
        """Enable HEALPix grid survey mode."""
        from star_pattern.core.healpix_survey import HEALPixSurvey

        state_dir = Path(self.run_manager.run_dir)
        self._survey = HEALPixSurvey(config, state_dir=state_dir)
        stats = self._survey.coverage_stats()
        logger.info(
            f"Survey mode enabled: NSIDE={config.nside}, "
            f"order={config.order}, "
            f"{stats['n_total']} pixels ({stats['n_remaining']} remaining)"
        )

    def _fetch_wide_field(
        self, region: SkyRegion, field_radius: float
    ) -> RegionData:
        """Fetch wide-field data with tiling and mosaicking."""
        return self.wide_field_pipeline.fetch_wide_field(
            center_ra=region.ra,
            center_dec=region.dec,
            field_radius_arcmin=field_radius,
        )

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        if self._shutdown:
            logger.info("Second interrupt -- exiting immediately.")
            raise SystemExit(1)
        logger.info("Shutdown requested. Stopping after current phase...")
        self._shutdown = True

    def _init_llm(self) -> None:
        """Initialize LLM components if enabled.

        Sets up: providers, token tracker, cache, strategy advisor.
        Legacy components (hypothesis gen, debate) are kept for
        rare escalation cases but are no longer called per-detection.
        """
        if not self.use_llm or self._llm_providers is not None:
            return

        from star_pattern.llm.providers.discovery import ProviderDiscovery
        from star_pattern.llm.hypothesis import HypothesisGenerator
        from star_pattern.llm.search_guide import LLMSearchGuide
        from star_pattern.llm.token_tracker import TokenTracker
        from star_pattern.llm.cache import LLMCache
        from star_pattern.llm.strategy import StrategyAdvisor

        discovery = ProviderDiscovery(key_dir=self.config.llm.key_dir)
        self._llm_providers = discovery.discover()

        if self._llm_providers:
            # Token tracking and caching
            self._token_tracker = TokenTracker(
                budget_tokens=self.config.llm.token_budget,
            )
            cache_dir = Path(self.run_manager.run_dir) / "llm_cache"
            self._llm_cache = LLMCache(cache_dir)

            # Strategy advisor (new primary LLM integration)
            self._strategy_advisor = StrategyAdvisor(
                providers=self._llm_providers,
                tracker=self._token_tracker,
                cache=self._llm_cache,
            )

            # Legacy components kept for escalation
            self._hypothesis_gen = HypothesisGenerator(
                self._llm_providers[0], self.config.llm
            )
            self._search_guide = LLMSearchGuide(
                self._llm_providers[0], self.config.llm
            )

            if len(self._llm_providers) >= 2:
                from star_pattern.llm.debate import PatternDebate
                from star_pattern.llm.consensus import PatternConsensus

                self._debate = PatternDebate(self._llm_providers, self.config.llm)
                self._consensus = PatternConsensus(self._llm_providers, self.config.llm)

            logger.info(
                f"LLM initialized: {len(self._llm_providers)} providers, "
                f"budget={self.config.llm.token_budget} tokens, "
                f"strategy_interval={self.config.llm.strategy_interval}"
            )
        else:
            logger.warning("No LLM providers found. Continuing without LLM.")
            self.use_llm = False

    def _get_next_region(self) -> SkyRegion:
        """Get the next region to search.

        Priority:
        1. LLM strategy-suggested regions (from periodic sessions)
        2. HEALPix survey grid (if enabled)
        3. Random region selection
        """
        # Use strategy-suggested regions first
        if self._suggested_regions:
            region = self._suggested_regions.pop(0)
            logger.info(f"Using strategy-suggested region: {region}")
            return region

        # Use survey grid if enabled
        if self._survey is not None:
            region = self._survey.next_region()
            if region is not None:
                logger.info(f"Using survey pixel {region._healpix_pixel}: {region}")  # type: ignore[attr-defined]
                return region
            logger.info("Survey complete, falling back to random")

        # Random region
        return SkyRegion.random(
            min_galactic_lat=self.config.data.min_galactic_latitude,
            radius=self.config.data.default_radius_arcmin,
        )

    def _process_region(self, region_data: RegionData) -> list[PatternResult]:
        """Run detection and evaluation on a region.

        Uses local classifier and evaluator (zero tokens) instead of
        per-detection LLM calls. Flags ambiguous/novel findings for
        batch review during strategy sessions.
        """
        results = []

        # Merge all catalogs for catalog-based detectors
        merged_catalog = None
        if region_data.catalogs:
            from star_pattern.core.catalog import StarCatalog, CatalogEntry
            all_entries: list[CatalogEntry] = []
            for cat in region_data.catalogs.values():
                all_entries.extend(cat.entries)
            if all_entries:
                merged_catalog = StarCatalog(entries=all_entries, source="merged")

        # Dynamic interest threshold from active learning
        interest_threshold = (
            self.active_learner.get_refined_threshold()
            if self.active_learner.feedback_history
            else 0.1
        )

        # Clear detection cache for this region
        self._last_detection_results = {}
        self._last_classification = None
        self._last_evaluation = None

        for band, image in region_data.images.items():
            if self._shutdown:
                break

            # Buffer images for evolution
            self._recent_images.append(image)
            if len(self._recent_images) > 50:
                self._recent_images = self._recent_images[-50:]

            try:
                detection = self.detector.detect(image, catalog=merged_catalog)

                # Phase 3: Embed image and get anomaly score
                if self._repr_manager is not None:
                    try:
                        embedding = self._repr_manager.embed_image(image)
                        if embedding is not None:
                            emb_score = self._repr_manager.embedding_anomaly_score(embedding)
                            detection["embedding_anomaly_score"] = emb_score
                            self._repr_manager.buffer_image(image, embedding)
                    except Exception as e:
                        logger.debug(f"Representation embedding failed: {e}")

                # Phase 4: Run best composed pipeline
                if self._best_pipeline is not None:
                    try:
                        comp_result = self._best_pipeline.run(image, detection)
                        detection["composed_score"] = comp_result.get("composed_score", 0)
                    except Exception as e:
                        logger.debug(f"Composed pipeline failed: {e}")

                # Re-extract rich features after injecting embedding/composed scores
                if "rich_features" in detection:
                    try:
                        from star_pattern.detection.feature_fusion import FeatureFusionExtractor
                        ff = FeatureFusionExtractor()
                        detection["rich_features"] = ff.extract(detection)
                    except Exception:
                        pass

                # Cache detection for _save_region_summary (avoids re-run)
                self._last_detection_results[band] = detection

                # Use meta_score when available, fall back to anomaly_score
                anomaly_score = detection.get(
                    "meta_score", detection.get("anomaly_score", 0)
                )

                # Only process interesting detections
                if anomaly_score < interest_threshold:
                    logger.debug(
                        f"  Filtered {band}: score={anomaly_score:.4f} "
                        f"< threshold={interest_threshold:.4f}"
                    )
                    continue

                # --- LOCAL classification (zero tokens) ---
                classification = self.local_classifier.classify(detection)

                result = PatternResult(
                    region_ra=region_data.region.ra,
                    region_dec=region_data.region.dec,
                    detection_type=classification["classification"],
                    anomaly_score=anomaly_score,
                    significance=detection.get("lens", {}).get("lens_score", 0),
                    details=detection,
                )

                # Store local classification in metadata
                result.metadata["local_classification"] = classification
                result.hypothesis = classification["rationale"]

                # --- LOCAL evaluation (zero tokens) ---
                evaluation = self.local_evaluator.evaluate(detection, image)
                result.debate_verdict = evaluation["verdict"]
                result.metadata["local_evaluation"] = evaluation
                result.metadata["significance_rating"] = evaluation["significance_rating"]

                # Cache for summary image generation
                self._last_classification = classification
                self._last_evaluation = evaluation

                # Save overlay images
                try:
                    image_paths = self._save_finding_images(
                        image, detection, self.cycle, band,
                    )
                    result.metadata["image_paths"] = [
                        str(p) for p in image_paths
                    ]
                except Exception as e:
                    logger.debug(f"Image saving failed: {e}")

                # Cross-reference
                try:
                    xref = self.cross_ref.cross_reference(
                        region_data.region.ra, region_data.region.dec
                    )
                    result.cross_matches = xref.get("matches", [])
                except Exception as e:
                    logger.debug(f"Cross-reference failed: {e}")

                # Local active learning feedback (no LLM call)
                if self.active_learner.should_query(result):
                    is_interesting = evaluation["verdict"] == "real"
                    rich_feats = detection.get("rich_features")
                    self.active_learner.add_feedback(
                        result, is_interesting,
                        rich_features=rich_feats,
                        detector_scores=classification.get("detector_scores"),
                        notes=f"local: {evaluation['verdict']}",
                    )
                    result.metadata["feedback"] = is_interesting

                # Queue for LLM batch review if flagged
                needs_review = (
                    classification.get("needs_llm_review", False)
                    or evaluation.get("needs_llm_review", False)
                )
                if needs_review:
                    self._flagged_for_review.append({
                        "ra": region_data.region.ra,
                        "dec": region_data.region.dec,
                        "classification": classification["classification"],
                        "confidence": classification["confidence"],
                        "anomaly_score": anomaly_score,
                        "verdict": evaluation["verdict"],
                        "rationale": classification["rationale"],
                    })

                results.append(result)

            except Exception as e:
                logger.warning(f"Detection failed on {band}: {e}")

        return results

    def _save_finding_images(
        self,
        image: FITSImage,
        detection: dict[str, Any],
        cycle: int,
        band: str,
    ) -> list[Path]:
        """Save annotated overlay images for a detection."""
        saved: list[Path] = []
        prefix = f"cycle_{cycle:04d}_{band}"

        from star_pattern.visualization.pattern_overlay import (
            overlay_sources, overlay_lens_detection, overlay_distribution,
            overlay_galaxy_features, overlay_classical_detection,
            overlay_morphology, overlay_sersic_analysis,
            overlay_wavelet_detection, overlay_kinematic_groups,
            overlay_transient_detection, overlay_variability,
            overlay_population_cmd, overlay_anomaly_scores,
        )
        import matplotlib.pyplot as plt

        def _save(name: str, fig: object) -> None:
            try:
                path = self.run_manager.save_image(f"{prefix}_{name}.png", fig)
                saved.append(path)
            finally:
                plt.close(fig)

        def _try_save(name: str, func: object, *args: object) -> None:
            """Create overlay figure and save it, closing on any error."""
            try:
                fig = func(*args)
                _save(name, fig)
            except Exception as e:
                logger.debug(f"Overlay {name} failed: {e}")

        # Source overlay (always)
        sources = detection.get("sources", {})
        if sources:
            _try_save("sources", overlay_sources, image, sources)

        # Classical overlay
        classical = detection.get("classical", {})
        if (classical.get("gabor_score", 0) > 0.15
                or classical.get("arc_score", 0) > 0.15):
            _try_save("classical", overlay_classical_detection, image, classical)

        # Morphology overlay
        morphology = detection.get("morphology", {})
        if morphology.get("morphology_score", 0) > 0.2:
            _try_save("morphology", overlay_morphology, image, morphology)

        # Lens overlay
        lens = detection.get("lens", {})
        if lens.get("lens_score", 0) > 0.2:
            _try_save("lens", overlay_lens_detection, image, lens)

        # Sersic overlay
        sersic = detection.get("sersic", {})
        if sersic.get("sersic_score", 0) > 0.15:
            _try_save("sersic", overlay_sersic_analysis, image, sersic)

        # Distribution overlay
        dist = detection.get("distribution", {})
        if dist.get("distribution_score", 0) > 0.2:
            positions = np.array(sources.get("positions", []))
            _try_save("distribution", overlay_distribution,
                image, dist, positions if len(positions) > 0 else None,
            )

        # Wavelet overlay
        wavelet = detection.get("wavelet", {})
        if wavelet.get("wavelet_score", 0) > 0.15:
            _try_save("wavelet", overlay_wavelet_detection, image, wavelet)

        # Galaxy overlay
        galaxy = detection.get("galaxy", {})
        if galaxy.get("galaxy_score", 0) > 0.2:
            _try_save("galaxy", overlay_galaxy_features, image, galaxy)

        # Kinematic overlay (catalog-based)
        kinematic = detection.get("kinematic", {})
        if (not kinematic.get("no_catalog")
                and kinematic.get("kinematic_score", 0) > 0.15):
            _try_save("kinematic", overlay_kinematic_groups, kinematic)

        # Transient overlay (catalog-based)
        transient = detection.get("transient", {})
        if (not transient.get("no_catalog")
                and transient.get("transient_score", 0) > 0.15):
            _try_save("transient", overlay_transient_detection, image, transient)

        # Variability overlay (catalog-based)
        variability = detection.get("variability", {})
        if (not variability.get("no_catalog")
                and variability.get("variability_score", 0) > 0.15):
            _try_save("variability", overlay_variability, image, variability)

        # Population CMD overlay (catalog-based)
        population = detection.get("population", {})
        if (not population.get("no_catalog")
                and population.get("population_score", 0) > 0.15):
            _try_save("population", overlay_population_cmd, image, population)

        # Anomaly scores overview (always saved)
        _try_save("anomaly_scores", overlay_anomaly_scores, image, detection)

        # Close any leaked figures from overlay functions that raised
        # after creating a figure but before returning it
        plt.close("all")

        return saved

    def _save_region_summary(
        self,
        region_data: RegionData,
        findings: list[PatternResult],
        elapsed_minutes: float,
    ) -> None:
        """Save an annotated summary image using cached detection results."""
        from star_pattern.visualization.pattern_overlay import create_annotated_summary
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for band, image in region_data.images.items():
            # Use cached detection (avoids re-running the full ensemble)
            detection = self._last_detection_results.get(band)
            if detection is None:
                try:
                    detection = self.detector.detect(image)
                except Exception as e:
                    logger.debug(f"Summary detection failed for {band}: {e}")
                    continue

            fig = None
            try:
                fig = create_annotated_summary(
                    image=image,
                    detection=detection,
                    ra=region_data.region.ra,
                    dec=region_data.region.dec,
                    cycle=self.cycle,
                    elapsed_minutes=elapsed_minutes,
                    classification=self._last_classification,
                    evaluation=self._last_evaluation,
                    evolution_history=self._evolution_history or None,
                )
                prefix = f"cycle_{self.cycle:04d}_{band}_summary"
                self.run_manager.save_image(f"{prefix}.png", fig)
            except Exception as e:
                logger.debug(f"Summary image save failed for {band}: {e}")
            finally:
                if fig is not None:
                    plt.close(fig)
                # Close any leaked figures from partial rendering
                plt.close("all")
            break  # One summary per region is enough

    @staticmethod
    def _classify_detection(detection: dict[str, Any]) -> str:
        """Classify the primary type of detection.

        Kept for backward compatibility. LocalClassifier is preferred.
        """
        scores = {
            "lens": detection.get("lens", {}).get("lens_score", 0),
            "morphology": detection.get("morphology", {}).get("morphology_score", 0),
            "distribution": detection.get("distribution", {}).get("distribution_score", 0),
            "galaxy": detection.get("galaxy", {}).get("galaxy_score", 0),
            "kinematic": detection.get("kinematic", {}).get("kinematic_score", 0),
            "transient": detection.get("transient", {}).get("transient_score", 0),
        }
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def run(self, max_hours: float | None = None) -> list[PatternResult]:
        """Run the autonomous discovery loop.

        Args:
            max_hours: Maximum runtime in hours (None = use config.max_cycles).

        Returns:
            List of all pattern results found.
        """
        if self.use_llm:
            self._init_llm()

        start_time = time.time()
        max_seconds = max_hours * 3600 if max_hours else float("inf")
        max_cycles = self.config.max_cycles
        strategy_interval = self.config.llm.strategy_interval

        logger.info(
            f"Starting autonomous discovery "
            f"(max_cycles={max_cycles}, max_hours={max_hours})"
        )

        while self.cycle < max_cycles and not self._shutdown:
            elapsed = time.time() - start_time
            if elapsed > max_seconds:
                logger.info("Time limit reached.")
                break

            self.cycle += 1
            elapsed_min = elapsed / 60
            elapsed_hr = elapsed / 3600
            findings_rate = (
                len(self.findings) / elapsed_hr if elapsed_hr > 0.01 else 0
            )

            # Progress banner
            logger.info(f"\n{'='*60}")
            logger.info(
                f"CYCLE {self.cycle}/{max_cycles} | "
                f"Elapsed: {elapsed_min:.1f} min | "
                f"Findings: {len(self.findings)} ({findings_rate:.1f}/hr) | "
                f"Regions: {len(self.searched_regions)}"
            )
            logger.info(f"{'='*60}")

            # PHASE: FETCHING
            logger.info("PHASE: FETCHING")
            region = self._get_next_region()
            self.searched_regions.append(region)

            # Fetch data (wide-field or standard)
            try:
                if (
                    self.wide_field_pipeline is not None
                    and self._wide_field_radius is not None
                ):
                    region_data = self._fetch_wide_field(
                        region, self._wide_field_radius
                    )
                else:
                    region_data = self.data_pipeline.fetch_region(region)
            except Exception as e:
                logger.error(f"Data fetch failed: {e}")
                continue

            if self._shutdown:
                logger.info("Shutdown: stopping after fetch phase.")
                break

            if not region_data.has_images():
                # In survey mode, mark empty pixels visited without
                # spending a cycle -- SDSS covers ~35% of the sky, so
                # most survey pixels will have no data.
                if self._survey is not None and hasattr(region, "_healpix_pixel"):
                    self._survey.mark_visited(
                        region._healpix_pixel, findings_count=0,  # type: ignore[attr-defined]
                    )
                    self.cycle -= 1  # Don't count empty survey pixels
                    logger.info(
                        f"No images at survey pixel "
                        f"{region._healpix_pixel}, skipping "  # type: ignore[attr-defined]
                        f"(not counted as cycle)"
                    )
                else:
                    logger.info("No images found, skipping.")
                continue

            # PHASE: DETECTING + EVALUATING
            logger.info(
                f"PHASE: DETECTING ({len(region_data.images)} bands) "
                f"at RA={region.ra:.4f} Dec={region.dec:.4f}"
            )
            new_findings = self._process_region(region_data)

            if self._shutdown:
                # Still save partial findings before exiting
                if new_findings:
                    self.findings.extend(new_findings)
                logger.info("Shutdown: stopping after detection phase.")
                break

            # Always save an annotated summary for every region
            self._save_region_summary(
                region_data, new_findings, elapsed_min,
            )

            if new_findings:
                self.findings.extend(new_findings)
                for f in new_findings:
                    logger.info(f"  FOUND: {f}")

                # Save results
                self.run_manager.save_result(
                    f"cycle_{self.cycle:04d}",
                    {"findings": [f.to_dict() for f in new_findings]},
                )
            else:
                logger.info(
                    f"  No findings above threshold "
                    f"(processed {len(region_data.images)} bands)"
                )

            # Mark survey pixel as visited
            if self._survey is not None and hasattr(region, "_healpix_pixel"):
                self._survey.mark_visited(
                    region._healpix_pixel,  # type: ignore[attr-defined]
                    findings_count=len(new_findings),
                )

            if self._shutdown:
                logger.info("Shutdown: stopping before checkpoint/strategy/evolution.")
                break

            # Checkpoint
            if self.cycle % self.config.checkpoint_interval == 0:
                self._save_checkpoint()
                if self._survey is not None:
                    self._survey.save_state()

            # Strategy session (periodic LLM review)
            if (
                self.cycle > 0
                and self.cycle % strategy_interval == 0
                and self._strategy_advisor is not None
                and not self._shutdown
            ):
                logger.info("PHASE: STRATEGY")
                self._run_strategy_session()

            # Adaptive evolution
            if (
                self.cycle > 0
                and self.cycle % self.config.evolve_interval == 0
                and not self._shutdown
            ):
                self._evolve_parameters()

            # Log progress
            token_info = ""
            if self._token_tracker:
                token_info = (
                    f", tokens={self._token_tracker.total_tokens}"
                    f"/{self._token_tracker.budget_tokens}"
                )
            survey_info = ""
            if self._survey is not None:
                stats = self._survey.coverage_stats()
                survey_info = (
                    f", survey={stats['percent_complete']:.1f}% "
                    f"({stats['n_visited']}/{stats['n_total']})"
                )
            logger.info(
                f"Total findings: {len(self.findings)}, "
                f"regions searched: {len(self.searched_regions)}"
                f"{token_info}{survey_info}"
            )

        # Final save
        self._save_checkpoint()
        if self._survey is not None:
            self._survey.save_state()
        self.run_manager.save_result(
            "all_findings",
            {
                "findings": [f.to_dict() for f in self.findings],
                "n_cycles": self.cycle,
                "n_regions": len(self.searched_regions),
            },
        )

        # Save token usage
        if self._token_tracker:
            token_path = Path(self.run_manager.run_dir) / "token_usage.json"
            self._token_tracker.save(token_path)

        # Generate reports
        try:
            self._generate_report()
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")

        logger.info(
            f"\nDiscovery complete: {len(self.findings)} findings in {self.cycle} cycles"
        )
        return self.findings

    def _run_strategy_session(self) -> None:
        """Periodic LLM strategy review. ~1,000-2,500 tokens per session.

        Reviews pipeline performance, flagged findings, and provides
        strategic guidance for detector parameters and region selection.
        """
        if not self._strategy_advisor:
            return

        if self._token_tracker and not self._token_tracker.can_afford(2500):
            logger.info("Token budget too low for strategy session, skipping")
            return

        logger.info("Running LLM strategy session...")

        # Build compact summary
        findings_summary = self._summarize_recent_findings()
        genome_config = (
            self._current_genome.to_detection_config()
            if self._current_genome
            else {}
        )
        al_stats = self.active_learner.get_strategy_summary()
        evo_history = self._evolution_history[-5:]  # Last 5 entries

        # Record outcome of previous strategy
        prev_outcome = self._last_strategy_outcome

        try:
            strategy = self._strategy_advisor.review_session(
                findings_summary=findings_summary,
                current_genome=genome_config,
                active_learning_stats=al_stats,
                evolution_history=evo_history,
                previous_strategy_outcome=prev_outcome,
            )
        except Exception as e:
            logger.warning(f"Strategy session failed: {e}")
            return

        # Apply strategy locally (no more LLM calls)
        self._apply_strategy(strategy)

        # Batch review flagged findings
        if self._flagged_for_review:
            try:
                reviews = self._strategy_advisor.review_flagged_findings(
                    self._flagged_for_review
                )
                for i, review in enumerate(reviews):
                    if i < len(self._flagged_for_review):
                        self._flagged_for_review[i].update(review)
                logger.info(
                    f"Batch reviewed {len(reviews)} flagged findings"
                )
            except Exception as e:
                logger.warning(f"Batch review failed: {e}")
            self._flagged_for_review = []

        # Track outcome for next session
        self._last_strategy_outcome = {
            "n_total": len(self.findings),
            "n_high_confidence": sum(
                1 for f in self.findings[-25:]
                if f.anomaly_score > 0.5
            ),
            "interesting_rate": al_stats.get("interesting_rate", 0),
        }

        logger.info("Strategy session complete")

    def _apply_strategy(self, strategy: Any) -> None:
        """Apply strategy results to all pipeline components. Zero tokens."""
        # Update active learner weights
        self.active_learner.apply_strategy(strategy)

        # Store for next evolution cycle
        self._pending_strategy = strategy

        # Update region selection
        if strategy.focus_regions:
            self._suggested_regions = list(strategy.focus_regions)

    def _summarize_recent_findings(self) -> dict[str, Any]:
        """Build compact findings summary for strategy sessions."""
        recent = self.findings[-25:] if self.findings else []

        type_counts: dict[str, int] = {}
        n_high = 0
        n_artifacts = 0

        for f in recent:
            t = f.detection_type
            type_counts[t] = type_counts.get(t, 0) + 1
            if f.anomaly_score > 0.5:
                n_high += 1
            if f.debate_verdict == "artifact":
                n_artifacts += 1

        return {
            "n_total": len(recent),
            "n_high_confidence": n_high,
            "n_artifacts": n_artifacts,
            "type_counts": type_counts,
            "n_regions": len(self.searched_regions),
        }

    def _evolve_parameters(self) -> None:
        """Run a short evolutionary search to adapt detection parameters."""
        if len(self._recent_images) < 5:
            return

        # Subsample images to cap evolution compute
        max_evo_images = 10
        if len(self._recent_images) > max_evo_images:
            indices = np.random.default_rng().choice(
                len(self._recent_images), size=max_evo_images, replace=False
            )
            evo_images = [self._recent_images[i] for i in indices]
        else:
            evo_images = list(self._recent_images)

        max_seconds = self.config.evolve_max_seconds

        logger.info(
            f"PHASE: EVOLVING -- {len(evo_images)} images "
            f"(from {len(self._recent_images)} buffered), "
            f"time_limit={max_seconds}s"
        )

        evo_config = PipelineConfig(
            detection=self.config.detection,
            evolution=EvolutionConfig(
                population_size=self.config.evolve_population,
                generations=self.config.evolve_generations,
                mutation_rate=self.config.evolution.mutation_rate,
                crossover_rate=self.config.evolution.crossover_rate,
            ),
            evolve_workers=self.config.evolve_workers,
        )

        engine = EvolutionaryDiscovery(evo_config, images=evo_images)

        # Seed population with LLM strategy suggestions
        if self._pending_strategy is not None:
            engine.initialize_population()
            engine.apply_strategy_to_population(self._pending_strategy)

        if self._shutdown:
            logger.info("Shutdown: skipping evolution run.")
            return

        evo_start = time.time()
        best = engine.run(max_seconds=max_seconds)
        evo_elapsed = time.time() - evo_start

        # Merge LLM strategy weights if available
        if self._pending_strategy is not None:
            engine.merge_strategy_weights(best, self._pending_strategy)
            self._pending_strategy = None

        # Apply best genome
        new_config = DetectionConfig.from_genome_dict(best.to_detection_config())
        self.detector = EnsembleDetector(new_config)
        self._current_genome = best
        self.config.detection = new_config

        self._evolution_history.append({
            "cycle": self.cycle,
            "fitness": best.fitness,
            "components": best.fitness_components,
        })

        # Store per-generation history for evolution summary
        self._full_generation_histories.append(
            engine.history if engine.history else []
        )

        logger.info(
            f"Evolution complete in {evo_elapsed:.1f}s: "
            f"fitness={best.fitness:.4f}"
        )
        self.run_manager.save_result(
            f"evolution_cycle_{self.cycle:04d}",
            {"genome": best.to_dict(), "history": engine.history},
        )

        # Phase 3: BYOL retrain during evolution (natural time for GPU work)
        if self._repr_manager is not None:
            try:
                retrain_result = self._repr_manager.maybe_retrain_backbone()
                if retrain_result:
                    logger.info("BYOL backbone retrained during evolution phase")
            except Exception as e:
                logger.debug(f"BYOL retrain failed: {e}")

        # Phase 4: Evolve pipeline genomes
        if self._pipeline_genomes and self.config.compositional.enabled:
            try:
                self._evolve_pipeline_genomes(evo_images)
            except Exception as e:
                logger.debug(f"Pipeline genome evolution failed: {e}")

        # Save evolution summary image
        fig = None
        try:
            from star_pattern.visualization.pattern_overlay import (
                create_evolution_summary,
            )
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = create_evolution_summary(
                generation_histories=self._full_generation_histories,
                evolution_history=self._evolution_history,
            )
            self.run_manager.save_image(
                f"evolution_summary_cycle_{self.cycle:04d}.png", fig,
            )
        except Exception as e:
            logger.debug(f"Evolution summary image failed: {e}")
        finally:
            try:
                import matplotlib.pyplot as plt
                if fig is not None:
                    plt.close(fig)
                plt.close("all")
            except Exception:
                pass

    def _evolve_pipeline_genomes(
        self, images: list[FITSImage]
    ) -> None:
        """Evolve the compositional pipeline genome population.

        Evaluates each pipeline genome on the image buffer, then applies
        tournament selection, crossover, and mutation.
        """
        from star_pattern.discovery.pipeline_genome import PipelineGenome

        comp_config = self.config.compositional
        rng = np.random.default_rng()

        # Evaluate current population
        for genome in self._pipeline_genomes:
            pipeline = ComposedPipeline(genome.to_pipeline_spec())
            scores = []
            for img in images[:5]:  # Cap eval cost
                try:
                    result = pipeline.run(img)
                    scores.append(result.get("composed_score", 0))
                except Exception:
                    scores.append(0)
            genome.fitness = float(np.mean(scores)) if scores else 0.0

        # Sort by fitness
        self._pipeline_genomes.sort(key=lambda g: g.fitness, reverse=True)

        # Set best pipeline for use in detection
        if self._pipeline_genomes:
            best_spec = self._pipeline_genomes[0].to_pipeline_spec()
            self._best_pipeline = ComposedPipeline(best_spec)
            logger.info(
                f"Best composed pipeline: "
                f"fitness={self._pipeline_genomes[0].fitness:.4f}, "
                f"{self._pipeline_genomes[0].describe()}"
            )

        # Evolve next generation
        pop_size = comp_config.evolve_population
        elite_count = max(1, pop_size // 5)
        new_pop = list(self._pipeline_genomes[:elite_count])

        while len(new_pop) < pop_size:
            # Tournament selection
            candidates = rng.choice(
                self._pipeline_genomes,
                size=min(3, len(self._pipeline_genomes)),
                replace=False,
            )
            parent1 = max(candidates, key=lambda g: g.fitness)

            candidates = rng.choice(
                self._pipeline_genomes,
                size=min(3, len(self._pipeline_genomes)),
                replace=False,
            )
            parent2 = max(candidates, key=lambda g: g.fitness)

            if rng.random() < 0.7:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1 = PipelineGenome(
                    [OperationSpec(name=op.name, params=dict(op.params))
                     for op in parent1.operations],
                    parent1.score_method, rng,
                )
                child2 = PipelineGenome(
                    [OperationSpec(name=op.name, params=dict(op.params))
                     for op in parent2.operations],
                    parent2.score_method, rng,
                )

            new_pop.append(child1.mutate(0.2))
            if len(new_pop) < pop_size:
                new_pop.append(child2.mutate(0.2))

        self._pipeline_genomes = new_pop

    def _generate_report(self) -> dict[str, Path]:
        """Generate full discovery report with text, JSON, mosaic, and histogram."""
        from star_pattern.visualization.report import DiscoveryReport

        report = DiscoveryReport(self.run_manager.reports_dir)
        metadata = {
            "run_name": self.run_manager.run_name,
            "n_cycles": self.cycle,
            "n_regions": len(self.searched_regions),
            "n_findings": len(self.findings),
            "evolution_runs": len(self._evolution_history),
            "evolution_history": self._evolution_history,
        }

        # Include token usage if available
        if self._token_tracker:
            metadata["token_usage"] = self._token_tracker.summary()

        paths = report.generate_full_report(self.findings, metadata)

        logger.info(f"Reports saved to {self.run_manager.reports_dir}")
        return paths

    def _save_checkpoint(self) -> None:
        self.run_manager.save_checkpoint(
            f"autonomous_cycle_{self.cycle}",
            {
                "cycle": self.cycle,
                "n_findings": len(self.findings),
                "findings": [f.to_dict() for f in self.findings[-50:]],
                "regions_searched": len(self.searched_regions),
            },
        )
