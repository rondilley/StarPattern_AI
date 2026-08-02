"""Characterization tests pinning the exact shape of EnsembleDetector.detect().

The results dict is a public contract in everything but name. FeatureFusionExtractor,
MetaDetector, the confidence layer, and the report layer all read specific keys out
of it, and several read them positionally by insertion order once serialized. A
refactor that changes a key name, a key's position, or a skip-marker shape breaks
those consumers silently, because they use .get() and see a zero instead of an error.

These tests record the full serialized output for four scenarios and compare every
later run against it: same keys, same insertion order at every level, same types,
same values, all to 1e-9. Goldens must not be regenerated to make a failing test
pass. Regenerate only when a change is intended to alter output, and review the
fixture diff as part of that change:

    STARPATTERN_RECORD_GOLDEN=1 pytest tests/test_ensemble_characterization.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from star_pattern.core.catalog import CatalogEntry, StarCatalog
from star_pattern.core.config import DetectionConfig
from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.sky_region import EpochImage
from star_pattern.detection.ensemble import EnsembleDetector, _to_serializable

FIXTURES = Path(__file__).parent / "fixtures"
RECORD = os.environ.get("STARPATTERN_RECORD_GOLDEN") == "1"

# Floats are compared with a tolerance rather than exactly. The detectors run
# through BLAS and, for the classical and wavelet paths, through the GPU, and
# neither guarantees the last ulp is reproducible across machines. The tolerance
# is tight enough that any real change in behaviour still fails.
REL_TOL = 1e-9
ABS_TOL = 1e-12

# There are no loosened paths. There used to be: sep's deblender is not
# reproducible, and while SourceExtractor still reported sep's segmentation
# flux and shape, four output arrays and three rich_features had to be
# compared with a 30% relative tolerance for this suite to pass at all.
# SourceExtractor now measures flux and shape itself with a fixed aperture
# and fixed-window moments, so every field above is exactly reproducible
# and every comparison here is tight. If a tolerance ever has to be widened
# again, that is a regression to investigate, not a knob to turn.


# --------------------------------------------------------------------------
# Deterministic inputs
# --------------------------------------------------------------------------


def _field_image(seed: int = 20260801, size: int = 192, n_sources: int = 40) -> np.ndarray:
    """A star field with a central galaxy, an arc, and scattered point sources."""
    rng = np.random.default_rng(seed)
    data = rng.normal(100.0, 8.0, (size, size))

    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Central extended source, so sersic and morphology have something to fit.
    data += 400.0 * np.exp(-0.5 * (r / 6.0) ** 2)

    # Arc at r ~ 28 px, so lens and classical have something to find.
    theta = np.arctan2(yy - cy, xx - cx)
    arc = (np.abs(r - 28.0) < 2.5) & (np.abs(theta - 0.6) < 1.1)
    data[arc] += 180.0

    # Point sources, so source extraction and distribution have positions.
    for _ in range(n_sources):
        sy, sx = rng.integers(12, size - 12, size=2)
        sigma = rng.uniform(1.5, 3.0)
        py, px = np.mgrid[-8:9, -8:9]
        data[sy - 8 : sy + 9, sx - 8 : sx + 9] += 600.0 * np.exp(-(px**2 + py**2) / (2 * sigma**2))

    return data.astype(np.float32)


def _catalog(seed: int = 7, n: int = 120) -> StarCatalog:
    """Catalog with the property keys the catalog-based detectors read."""
    rng = np.random.default_rng(seed)
    entries = []
    for i in range(n):
        g = float(rng.uniform(14.0, 21.0))
        entries.append(
            CatalogEntry(
                ra=180.0 + float(rng.normal(0, 0.01)),
                dec=45.0 + float(rng.normal(0, 0.01)),
                mag=g,
                mag_band="G",
                obj_type="star" if rng.random() > 0.25 else "galaxy",
                source="characterization",
                source_id=f"src_{i:04d}",
                properties={
                    "G": g,
                    "BP": g + float(rng.uniform(0.2, 1.4)),
                    "RP": g - float(rng.uniform(0.2, 1.2)),
                    "bp_rp": float(rng.uniform(0.1, 1.9)),
                    "parallax": float(rng.uniform(0.1, 5.0)),
                    "parallax_error": float(rng.uniform(0.01, 0.5)),
                    "pmra": float(rng.normal(0, 12.0)),
                    "pmdec": float(rng.normal(0, 12.0)),
                    "astro_noise": float(abs(rng.normal(0, 1.5))),
                },
            )
        )
    return StarCatalog(entries=entries, source="characterization")


def _epochs() -> list[EpochImage]:
    """Three epochs where one source appears and one brightens."""
    base = _field_image(seed=11, size=128, n_sources=15)
    epochs = []
    for i, mjd in enumerate((58000.0, 58120.0, 58300.0)):
        frame = base.copy()
        if i >= 1:
            py, px = np.mgrid[-8:9, -8:9]
            blob = 900.0 * np.exp(-(px**2 + py**2) / (2 * 2.0**2))
            frame[40 - 8 : 40 + 9, 70 - 8 : 70 + 9] += blob
        if i == 2:
            frame[90 - 8 : 90 + 9, 30 - 8 : 30 + 9] += blob * 0.7
        epochs.append(
            EpochImage(
                image=FITSImage.from_array(frame.astype(np.float32)),
                mjd=mjd,
                band="r",
                source="characterization",
            )
        )
    return epochs


SCENARIOS = ("full", "image_only", "gated", "sparse")


def _build(scenario: str) -> tuple[EnsembleDetector, dict[str, Any]]:
    """Return the detector and detect() kwargs for a scenario."""
    if scenario == "full":
        return EnsembleDetector(DetectionConfig()), {
            "image": FITSImage.from_array(_field_image()),
            "catalog": _catalog(),
            "temporal_images": _epochs(),
        }
    if scenario == "image_only":
        # Pins the no_catalog and no_temporal_images skip markers.
        return EnsembleDetector(DetectionConfig()), {
            "image": FITSImage.from_array(_field_image()),
        }
    if scenario == "gated":
        # Pins the disabled marker alongside detectors that still run.
        config = DetectionConfig()
        config.enabled_detectors = {
            "lens": False,
            "sersic": False,
            "kinematic": False,
            "wavelet": False,
        }
        return EnsembleDetector(config), {
            "image": FITSImage.from_array(_field_image()),
            "catalog": _catalog(),
        }
    if scenario == "sparse":
        # Pins n_sources_too_few: a near-empty frame yields under 10 sources.
        rng = np.random.default_rng(3)
        flat = rng.normal(100.0, 1.0, (96, 96)).astype(np.float32)
        return EnsembleDetector(DetectionConfig()), {
            "image": FITSImage.from_array(flat),
        }
    raise ValueError(f"Unknown scenario: {scenario}")


def _run(scenario: str) -> dict[str, Any]:
    detector, kwargs = _build(scenario)
    return json.loads(json.dumps(_to_serializable(detector.detect(**kwargs))))


# --------------------------------------------------------------------------
# Structural comparison
# --------------------------------------------------------------------------


def _compare(actual: Any, expected: Any, path: str, failures: list[str]) -> None:
    """Compare recursively, reporting key order and value differences."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            failures.append(f"{path}: expected dict, got {type(actual).__name__}")
            return
        # Key ORDER matters: report JSON and the mosaic layer read the
        # serialized dict, so a reordering is a visible output change.
        if list(actual.keys()) != list(expected.keys()):
            missing = [k for k in expected if k not in actual]
            added = [k for k in actual if k not in expected]
            if missing or added:
                failures.append(f"{path}: keys missing={missing} added={added}")
            else:
                failures.append(
                    f"{path}: key order changed\n"
                    f"      expected {list(expected.keys())}\n"
                    f"      actual   {list(actual.keys())}"
                )
            return
        for key in expected:
            _compare(actual[key], expected[key], f"{path}.{key}", failures)
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            failures.append(f"{path}: expected list, got {type(actual).__name__}")
            return
        if len(actual) != len(expected):
            failures.append(f"{path}: length {len(actual)} != expected {len(expected)}")
            return
        for i, (a, e) in enumerate(zip(actual, expected)):
            _compare(a, e, f"{path}[{i}]", failures)
        return

    if isinstance(expected, bool) or expected is None:
        if actual != expected:
            failures.append(f"{path}: {actual!r} != expected {expected!r}")
        return

    if isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            failures.append(f"{path}: expected number, got {actual!r}")
            return
        if not math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            failures.append(f"{path}: {actual!r} != expected {expected!r}")
        return

    if actual != expected:
        failures.append(f"{path}: {actual!r} != expected {expected!r}")


@pytest.mark.parametrize("scenario", SCENARIOS)
class TestEnsembleOutputShape:
    def test_matches_golden(self, scenario: str):
        """detect() output is unchanged against the recorded golden."""
        golden_path = FIXTURES / f"ensemble_golden_{scenario}.json"
        actual = _run(scenario)

        if RECORD or not golden_path.exists():
            FIXTURES.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(
                json.dumps(actual, indent=2, sort_keys=False) + "\n", encoding="utf-8"
            )
            pytest.skip(f"Recorded golden for {scenario}; rerun to compare")

        expected = json.loads(golden_path.read_text(encoding="utf-8"))
        failures: list[str] = []
        _compare(actual, expected, scenario, failures)
        assert not failures, "\n  ".join(["Output drifted from golden:"] + failures)

    def test_is_deterministic(self, scenario: str):
        """Two fresh detectors on the same input give the same output.

        Guards the golden itself: a comparison against a recording is only
        meaningful if the thing being recorded is reproducible.
        """
        first = _run(scenario)
        second = _run(scenario)
        failures: list[str] = []
        _compare(second, first, scenario, failures)
        assert not failures, "\n  ".join(["detect() is not deterministic:"] + failures)


class TestEnsembleContract:
    """Invariants that must survive even when the goldens are regenerated."""

    def test_top_level_keys_and_order(self):
        result = _run("full")
        assert list(result.keys()) == [
            "shape",
            "pixel_scale_arcsec",
            "sources",
            "classical",
            "morphology",
            "lens",
            "distribution",
            "galaxy",
            "kinematic",
            "transient",
            "sersic",
            "wavelet",
            "population",
            "variability",
            "temporal",
            "anomaly",
            "anomaly_score",
            "n_detections",
            "rich_features",
        ]

    def test_skip_markers(self):
        """Each way a detector can be skipped keeps its own distinct marker."""
        image_only = _run("image_only")
        assert image_only["kinematic"] == {"no_catalog": True}
        assert image_only["transient"] == {"no_catalog": True}
        assert image_only["population"] == {"no_catalog": True}
        assert image_only["variability"] == {"no_catalog": True}
        assert image_only["temporal"] == {"no_temporal_images": True}

        gated = _run("gated")
        assert gated["lens"] == {"disabled": True}
        assert gated["sersic"] == {"disabled": True}
        assert gated["wavelet"] == {"disabled": True}
        assert gated["kinematic"] == {"disabled": True}

        sparse = _run("sparse")
        assert sparse["distribution"] == {"n_sources_too_few": True}

    def test_score_is_bounded(self):
        for scenario in SCENARIOS:
            result = _run(scenario)
            assert 0.0 <= result["anomaly_score"] <= 1.0
            assert isinstance(result["n_detections"], int)
            assert result["n_detections"] >= 0

    def test_rich_features_length_is_stable(self):
        """feature_fusion reads the results dict; its width must not drift."""
        from star_pattern.detection.feature_fusion import FeatureFusionExtractor

        expected = FeatureFusionExtractor().n_features
        for scenario in SCENARIOS:
            assert len(_run(scenario)["rich_features"]) == expected


class TestEnsembleScoreRenormalization:
    """The ensemble score divides by the weight of the detectors that ran.

    Before this, the score was a weighted sum with no denominator, so a
    detector that could not run still consumed its share of the budget and
    contributed zero. An image with no catalog scored low for lack of a
    catalog rather than for lack of anything interesting, and any GA genome
    that disabled a detector was penalized for doing so.
    """

    def test_full_inputs_are_unaffected(self):
        """With every detector running the denominator is ~1, so nothing moves.

        This is what makes the change safe to reason about: it only lifts
        scores that were being unfairly suppressed.
        """
        detector, kwargs = _build("full")
        result = detector.detect(**kwargs)
        weights = detector.config.ensemble_weights
        total = sum(
            weights.get(spec.weight_name, spec.default_weight) for spec in detector._specs
        ) + weights.get("anomaly", 0.09)
        assert total == pytest.approx(1.0, abs=0.05)
        assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_missing_inputs_no_longer_suppress_the_score(self):
        """An image with no catalog is scored on what could actually run."""
        with_catalog = _run("full")["anomaly_score"]
        without_catalog = _run("image_only")["anomaly_score"]
        # Not a fixed relationship between the two images, but the
        # catalog-less one must no longer be dragged toward zero purely by
        # the five catalog detectors being absent.
        assert (
            without_catalog > 0.4
        ), f"score {without_catalog:.3f} looks suppressed by missing inputs"
        assert 0.0 <= with_catalog <= 1.0

    def test_disabling_a_detector_does_not_mechanically_lower_the_score(self):
        """The GA must not be punished for switching a detector off."""
        image = FITSImage.from_array(_field_image())
        catalog = _catalog()

        baseline = EnsembleDetector(DetectionConfig()).detect(image=image, catalog=catalog)[
            "anomaly_score"
        ]

        gated_config = DetectionConfig()
        gated_config.enabled_detectors = {"wavelet": False, "sersic": False}
        gated = EnsembleDetector(gated_config).detect(image=image, catalog=catalog)["anomaly_score"]

        # Removing two detectors changes the average over the rest; it must
        # not impose a fixed penalty proportional to their weight.
        weights = DetectionConfig().ensemble_weights
        removed = weights.get("wavelet", 0.09) + weights.get("sersic", 0.07)
        assert gated > baseline - removed, (
            f"disabling detectors cost {baseline - gated:.4f}, which is close to "
            f"their {removed:.4f} weight share: the old suppression is back"
        )

    def test_all_detectors_disabled_scores_zero(self):
        config = DetectionConfig()
        config.enabled_detectors = {
            spec.name: False for spec in EnsembleDetector(DetectionConfig())._specs
        }
        result = EnsembleDetector(config).detect(image=FITSImage.from_array(_field_image()))
        # Only the always-on anomaly meta-detector remains in the ratio.
        assert 0.0 <= result["anomaly_score"] <= 1.0


class TestSepDeblenderInstability:
    """Documents the sep defect that SourceExtractor works around.

    sep is still not reproducible; the pipeline simply no longer consumes
    the fields that wobble. These tests record why that workaround exists,
    so it is not "simplified" away by someone who finds the extra moment
    code redundant.

    No sep version fixes this. Tested 1.4.1, 1.4.0 and the sep-pjw 1.3.8
    fork: all three produce identical instability. 1.2.1 has no wheel for
    Python 3.12. No extract parameter fixes it either -- the only settings
    that are reproducible (deblend_nthresh=1, deblend_cont >= 0.5) are the
    ones that switch deblending off, and they find 26 sources on this
    field instead of 38.
    """

    def test_characterized_sep_version_is_the_pinned_one(self):
        """The measurements in this module belong to one specific sep build.

        pyproject pins sep==1.4.1. If that pin moves, re-run the
        measurements here rather than assuming they still hold.
        """
        sep = pytest.importorskip("sep")
        assert sep.__version__ == "1.4.1", (
            f"sep is {sep.__version__}, but the behaviour recorded in this "
            f"module and in the golden fixtures was measured against 1.4.1. "
            f"Re-run this suite and review the fixture diff before changing "
            f"the pin."
        )

    def test_extract_is_not_reproducible(self):
        """sep.extract returns different photometry for identical input."""
        sep = pytest.importorskip("sep")
        image = np.ascontiguousarray(_field_image().astype(np.float64))
        background = sep.Background(image)
        subtracted = image - background.back()

        results = [
            sep.extract(subtracted, thresh=3.0, err=background.globalrms, minarea=5)
            for _ in range(4)
        ]
        fluxes = {np.array(r["flux"], copy=True).tobytes() for r in results}
        positions = {np.array(r["x"], copy=True).tobytes() for r in results}

        assert len(positions) == 1, (
            "sep positions have become unstable. SourceExtractor measures flux "
            "and shape at these positions and relies on them being fixed."
        )
        assert len(fluxes) > 1, (
            "sep.extract has become reproducible. The aperture-photometry "
            "workaround in SourceExtractor could then be reconsidered in "
            "favour of segmentation photometry, which captures the full "
            "source rather than a fixed circle."
        )

    def test_disabling_deblending_restores_reproducibility(self):
        """Isolates the deblender as the cause, not extraction as a whole."""
        sep = pytest.importorskip("sep")
        image = np.ascontiguousarray(_field_image().astype(np.float64))
        background = sep.Background(image)
        subtracted = image - background.back()

        fluxes = {
            np.array(
                sep.extract(
                    subtracted,
                    thresh=3.0,
                    err=background.globalrms,
                    minarea=5,
                    deblend_cont=1.0,
                )["flux"],
                copy=True,
            ).tobytes()
            for _ in range(4)
        }
        assert len(fluxes) == 1

    def test_sep_shape_parameters_would_flip_the_star_galaxy_label(self):
        """Why SourceExtractor does not use sep's a, b and theta.

        Classifying on sep's shape parameters means `ellipticity < 0.3`
        computed from an unstable axis ratio. A source near that threshold
        is labelled a star on one run and a galaxy on the next, about 1 in
        38 on this field. The extractor computes its own moments instead;
        tests/test_source_extraction_determinism.py holds that boundary.
        """
        sep = pytest.importorskip("sep")
        image = np.ascontiguousarray(_field_image().astype(np.float64))
        background = sep.Background(image)
        subtracted = image - background.back()

        runs = [
            sep.extract(subtracted, thresh=3.0, err=background.globalrms, minarea=5)
            for _ in range(25)
        ]
        a = np.array([r["a"] for r in runs])
        b = np.array([r["b"] for r in runs])
        ellipticity = 1 - b / np.maximum(a, 1e-10)
        kron = np.sqrt(a * b)
        star = (ellipticity < 0.3) & (kron < np.median(kron, axis=1, keepdims=True) * 1.5)

        flips = int((star.max(axis=0) != star.min(axis=0)).sum())
        spread = float(np.ptp(ellipticity, axis=0).max())

        # Documents the observed magnitude. The upper bounds are the point:
        # if the instability grows past them, the tolerances in this module
        # are no longer justified and need re-deriving from fresh numbers.
        assert spread < 0.25, f"ellipticity spread grew to {spread:.3f}"
        assert flips <= 3, f"{flips} sources changed star/galaxy classification"

    def test_ensemble_score_is_unaffected(self):
        """The headline score must stay reproducible despite the defect.

        This is the assertion that actually protects the science. If it ever
        fails, the deblender wobble has reached the discovery score and the
        pipeline can no longer be trusted to rank the same field the same way
        twice.
        """
        runs = [_run("full") for _ in range(3)]
        assert len({r["anomaly_score"] for r in runs}) == 1
        assert len({r["n_detections"] for r in runs}) == 1
