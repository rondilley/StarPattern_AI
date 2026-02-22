"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.sky_region import SkyRegion, RegionData
from star_pattern.core.catalog import CatalogEntry, StarCatalog
from star_pattern.core.config import PipelineConfig

# Project root where *.key.txt files live
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def project_root() -> Path:
    """Path to project root (where *.key.txt files are)."""
    return PROJECT_ROOT


@pytest.fixture
def llm_providers():
    """Discover real LLM providers from *.key.txt in project root."""
    from star_pattern.llm.providers.discovery import ProviderDiscovery

    discovery = ProviderDiscovery(key_dir=PROJECT_ROOT)
    providers = discovery.discover()
    if not providers:
        pytest.skip("No LLM providers available (no *.key.txt with valid keys)")
    return providers


@pytest.fixture
def first_provider(llm_providers):
    """First available LLM provider."""
    return llm_providers[0]


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    """Temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def sample_config(tmp_output: Path) -> PipelineConfig:
    """Default pipeline config with temp output dir."""
    config = PipelineConfig()
    config.output_dir = str(tmp_output / "runs")
    config.data.cache_dir = str(tmp_output / "cache")
    return config


@pytest.fixture
def synthetic_image() -> FITSImage:
    """A synthetic FITS image with some structure."""
    rng = np.random.default_rng(42)
    data = rng.normal(100, 10, (256, 256)).astype(np.float32)
    # Add some bright sources
    for _ in range(20):
        x, y = rng.integers(20, 236, size=2)
        sigma = rng.uniform(1.5, 4.0)
        yy, xx = np.mgrid[-10:11, -10:11]
        gaussian = 500 * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        data[y - 10 : y + 11, x - 10 : x + 11] += gaussian
    return FITSImage.from_array(data)


@pytest.fixture
def synthetic_image_with_arc() -> FITSImage:
    """Synthetic image with an arc-like feature (simulated gravitational lens)."""
    rng = np.random.default_rng(123)
    data = rng.normal(100, 10, (256, 256)).astype(np.float32)
    # Central bright source
    yy, xx = np.mgrid[-128:128, -128:128]
    data += 300 * np.exp(-(xx**2 + yy**2) / (2 * 5**2))
    # Arc at radius ~30 pixels
    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    arc_mask = (np.abs(r - 30) < 3) & (np.abs(theta) < 1.0)
    data[arc_mask] += 200
    return FITSImage.from_array(data)


@pytest.fixture
def sample_region() -> SkyRegion:
    """A sample sky region."""
    return SkyRegion(ra=180.0, dec=45.0, radius=3.0)


@pytest.fixture
def sample_catalog() -> StarCatalog:
    """A sample star catalog."""
    rng = np.random.default_rng(42)
    entries = []
    for i in range(100):
        entries.append(
            CatalogEntry(
                ra=180.0 + rng.normal(0, 0.01),
                dec=45.0 + rng.normal(0, 0.01),
                mag=rng.uniform(14, 22),
                mag_band="r",
                obj_type="star" if rng.random() > 0.3 else "galaxy",
                source="test",
                source_id=f"test_{i}",
            )
        )
    return StarCatalog(entries=entries, source="test")


@pytest.fixture
def sample_region_data(
    sample_region: SkyRegion,
    synthetic_image: FITSImage,
    sample_catalog: StarCatalog,
) -> RegionData:
    """A complete RegionData with image and catalog."""
    return RegionData(
        region=sample_region,
        images={"r": synthetic_image},
        catalogs={"test": sample_catalog},
    )


@pytest.fixture
def synthetic_image_with_wcs() -> FITSImage:
    """Synthetic FITS image with valid TAN WCS for mosaicking tests."""
    from astropy.wcs import WCS
    from astropy.io import fits

    rng = np.random.default_rng(99)
    data = rng.normal(100, 10, (128, 128)).astype(np.float32)
    # Add bright sources
    for _ in range(10):
        x, y = rng.integers(20, 108, size=2)
        sigma = rng.uniform(1.5, 3.0)
        yy, xx = np.mgrid[-10:11, -10:11]
        gaussian = 300 * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        data[y - 10 : y + 11, x - 10 : x + 11] += gaussian

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [64, 64]
    wcs.wcs.cdelt = [-0.4 / 3600, 0.4 / 3600]  # 0.4 arcsec/pixel
    wcs.wcs.crval = [180.0, 45.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    header = fits.Header(wcs.to_header())
    return FITSImage(data=data, header=header, wcs=wcs)


@pytest.fixture
def synthetic_fits_file(tmp_path: Path, synthetic_image: FITSImage) -> Path:
    """Save synthetic image to a FITS file."""
    path = tmp_path / "test.fits"
    synthetic_image.save(path)
    return path
