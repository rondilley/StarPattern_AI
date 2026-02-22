"""Tests for HEALPix grid survey mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("astropy_healpix")

from star_pattern.core.config import SurveyConfig
from star_pattern.core.healpix_survey import HEALPixSurvey, _nside2npix
from star_pattern.core.sky_region import SkyRegion


@pytest.fixture
def small_config() -> SurveyConfig:
    """SurveyConfig with NSIDE=1 (12 pixels) for fast tests."""
    return SurveyConfig(
        nside=1,
        min_galactic_lat=0.0,  # No filtering for small tests
        radius_arcmin=3.0,
        order="galactic_latitude",
        state_file="test_survey_state.json",
    )


@pytest.fixture
def filtered_config() -> SurveyConfig:
    """SurveyConfig with galactic plane filtering."""
    return SurveyConfig(
        nside=2,
        min_galactic_lat=20.0,
        radius_arcmin=3.0,
        order="galactic_latitude",
        state_file="test_survey_state.json",
    )


def test_pixel_list_filters_galactic_plane(filtered_config: SurveyConfig, tmp_path: Path) -> None:
    """Filtered count < total count, all pass |b| >= threshold."""
    survey = HEALPixSurvey(filtered_config, state_dir=tmp_path)
    total_pixels = _nside2npix(filtered_config.nside)
    n_filtered = len(survey._all_pixels_set)

    # Some pixels should be filtered out (galactic plane)
    assert n_filtered < total_pixels
    assert n_filtered > 0

    # All remaining pixels should pass the galactic latitude filter
    for ipix in survey._all_pixels_set:
        region = survey.pixel_to_region(ipix)
        assert abs(region.galactic_lat) >= filtered_config.min_galactic_lat


def test_next_region_returns_sky_region(small_config: SurveyConfig, tmp_path: Path) -> None:
    """Returns valid SkyRegion with correct radius."""
    survey = HEALPixSurvey(small_config, state_dir=tmp_path)
    region = survey.next_region()

    assert region is not None
    assert isinstance(region, SkyRegion)
    assert region.radius == small_config.radius_arcmin


def test_mark_visited_advances(small_config: SurveyConfig, tmp_path: Path) -> None:
    """After mark_visited, next_region returns different pixel."""
    survey = HEALPixSurvey(small_config, state_dir=tmp_path)

    region1 = survey.next_region()
    assert region1 is not None
    pixel1 = region1._healpix_pixel  # type: ignore[attr-defined]
    survey.mark_visited(pixel1)

    region2 = survey.next_region()
    assert region2 is not None
    pixel2 = region2._healpix_pixel  # type: ignore[attr-defined]
    assert pixel2 != pixel1


def test_survey_complete_returns_none(small_config: SurveyConfig, tmp_path: Path) -> None:
    """After visiting all pixels, next_region returns None."""
    survey = HEALPixSurvey(small_config, state_dir=tmp_path)

    # Visit all pixels
    while True:
        region = survey.next_region()
        if region is None:
            break
        survey.mark_visited(region._healpix_pixel)  # type: ignore[attr-defined]

    assert survey.next_region() is None


def test_save_and_load_state(small_config: SurveyConfig, tmp_path: Path) -> None:
    """Round-trip persistence via JSON."""
    survey = HEALPixSurvey(small_config, state_dir=tmp_path)

    # Visit some pixels
    region = survey.next_region()
    assert region is not None
    survey.mark_visited(region._healpix_pixel, findings_count=3)  # type: ignore[attr-defined]
    survey.save_state()

    # Verify file exists
    state_path = tmp_path / small_config.state_file
    assert state_path.exists()

    # Load into new survey
    survey2 = HEALPixSurvey(small_config, state_dir=tmp_path)
    assert region._healpix_pixel in survey2._visited  # type: ignore[attr-defined]
    assert survey2._findings_per_pixel.get(region._healpix_pixel) == 3  # type: ignore[attr-defined]


def test_coverage_stats(small_config: SurveyConfig, tmp_path: Path) -> None:
    """Correct n_total, n_visited, percent_complete."""
    survey = HEALPixSurvey(small_config, state_dir=tmp_path)
    stats = survey.coverage_stats()

    assert stats["n_total"] == 12  # NSIDE=1 has 12 pixels
    assert stats["n_visited"] == 0
    assert stats["n_remaining"] == 12
    assert stats["percent_complete"] == 0.0
    assert stats["n_with_findings"] == 0

    # Visit one pixel with findings
    region = survey.next_region()
    assert region is not None
    survey.mark_visited(region._healpix_pixel, findings_count=2)  # type: ignore[attr-defined]
    stats = survey.coverage_stats()

    assert stats["n_visited"] == 1
    assert stats["n_remaining"] == 11
    assert stats["n_with_findings"] == 1
    assert stats["percent_complete"] > 0


def test_ordering_galactic_latitude(tmp_path: Path) -> None:
    """First pixels have higher |b| than last."""
    config = SurveyConfig(
        nside=2,
        min_galactic_lat=0.0,
        order="galactic_latitude",
        state_file="test_state.json",
    )
    survey = HEALPixSurvey(config, state_dir=tmp_path)

    pixels = survey._pending
    assert len(pixels) > 2

    # First pixel should have higher |galactic lat| than last
    first_region = survey.pixel_to_region(pixels[0])
    last_region = survey.pixel_to_region(pixels[-1])
    assert abs(first_region.galactic_lat) >= abs(last_region.galactic_lat)


def test_ordering_dec_sweep(tmp_path: Path) -> None:
    """Pixels ordered by |dec|."""
    config = SurveyConfig(
        nside=2,
        min_galactic_lat=0.0,
        order="dec_sweep",
        state_file="test_state.json",
    )
    survey = HEALPixSurvey(config, state_dir=tmp_path)

    pixels = survey._pending
    assert len(pixels) > 2

    # First pixel should have lower |dec| than last
    first_region = survey.pixel_to_region(pixels[0])
    last_region = survey.pixel_to_region(pixels[-1])
    assert abs(first_region.dec) <= abs(last_region.dec)


def test_pixel_to_region_valid_coords(small_config: SurveyConfig, tmp_path: Path) -> None:
    """RA in [0,360], Dec in [-90,90]."""
    survey = HEALPixSurvey(small_config, state_dir=tmp_path)

    for ipix in survey._all_pixels_set:
        region = survey.pixel_to_region(ipix)
        assert 0.0 <= region.ra <= 360.0
        assert -90.0 <= region.dec <= 90.0


def test_small_nside(tmp_path: Path) -> None:
    """Works with NSIDE=1 (12 pixels)."""
    config = SurveyConfig(
        nside=1,
        min_galactic_lat=0.0,
        state_file="test_state.json",
    )
    survey = HEALPixSurvey(config, state_dir=tmp_path)

    assert len(survey._all_pixels_set) == 12
    assert len(survey._pending) == 12


def test_resume_from_state(small_config: SurveyConfig, tmp_path: Path) -> None:
    """Reloaded survey skips already-visited pixels."""
    survey = HEALPixSurvey(small_config, state_dir=tmp_path)

    # Visit 3 pixels
    visited_pixels = []
    for _ in range(3):
        region = survey.next_region()
        assert region is not None
        pixel = region._healpix_pixel  # type: ignore[attr-defined]
        survey.mark_visited(pixel)
        visited_pixels.append(pixel)
    survey.save_state()

    # Resume
    survey2 = HEALPixSurvey(small_config, state_dir=tmp_path)
    next_region = survey2.next_region()
    assert next_region is not None
    assert next_region._healpix_pixel not in visited_pixels  # type: ignore[attr-defined]


def test_findings_tracked(small_config: SurveyConfig, tmp_path: Path) -> None:
    """findings_per_pixel records counts."""
    survey = HEALPixSurvey(small_config, state_dir=tmp_path)

    region = survey.next_region()
    assert region is not None
    pixel = region._healpix_pixel  # type: ignore[attr-defined]
    survey.mark_visited(pixel, findings_count=5)

    assert survey._findings_per_pixel[pixel] == 5

    # Zero findings should not be recorded
    region2 = survey.next_region()
    assert region2 is not None
    pixel2 = region2._healpix_pixel  # type: ignore[attr-defined]
    survey.mark_visited(pixel2, findings_count=0)

    assert pixel2 not in survey._findings_per_pixel


def test_healpix_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raises ImportError with helpful message when astropy-healpix missing."""
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "astropy_healpix":
            raise ImportError("No module named 'astropy_healpix'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    config = SurveyConfig(nside=1)
    with pytest.raises(ImportError, match="astropy-healpix is required"):
        HEALPixSurvey(config)


def test_survey_config_defaults() -> None:
    """Default SurveyConfig has sane values."""
    config = SurveyConfig()
    assert config.nside == 64
    assert config.min_galactic_lat == 20.0
    assert config.radius_arcmin == 3.0
    assert config.order == "galactic_latitude"
    assert config.state_file == "survey_state.json"
