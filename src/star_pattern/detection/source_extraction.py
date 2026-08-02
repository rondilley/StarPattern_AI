"""Source extraction using SEP and photutils.

Detection uses SEP, including its deblender, because that is what finds
blended sources. Photometry and shape do NOT use SEP's segmentation
output, because it is not reproducible: given a byte-identical image,
sep 1.4.1 returns identical positions and counts but assigns blended
pixels to neighbours differently on every call. Measured on a synthetic
field, 16 of 38 sources had flux varying by up to 9.7% between runs, and
one source in 38 flipped between the star and galaxy classifications.
Every installable sep build and every deblend parameter was tested; the
only reproducible settings are the ones that switch deblending off.

So this module takes the stable half of SEP's answer -- the positions --
and measures flux and shape itself with fixed-aperture photometry and
flux-weighted second moments. Both are plain deterministic numpy, so the
same image now yields the same catalog every time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from star_pattern.utils.logging import get_logger

logger = get_logger("detection.source_extraction")


def _aperture_moments(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flux-weighted second moments in a circular aperture per source.

    Replaces SEP's a, b and theta, which are segmentation-derived and
    therefore not reproducible. This is the standard SExtractor moment
    definition evaluated on a fixed aperture around a fixed position, so
    it depends only on the pixel values and gives the same answer every
    run.

    Args:
        data: Background-subtracted 2D image.
        x: Source x positions in pixels.
        y: Source y positions in pixels.
        radius: Aperture radius in pixels.

    Returns:
        Tuple of (a, b, theta): semi-major axis, semi-minor axis and
        position angle in radians, one entry per source.
    """
    n = len(x)
    a = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    theta = np.zeros(n, dtype=np.float64)
    if n == 0:
        return a, b, theta

    pad = int(np.ceil(radius))
    # Zero-padding keeps every window the same size, so a source near the
    # edge needs no special case and cannot silently get a smaller aperture.
    padded = np.pad(data.astype(np.float64), pad, mode="constant", constant_values=0.0)

    grid = np.arange(-pad, pad + 1, dtype=np.float64)
    dy_grid, dx_grid = np.meshgrid(grid, grid, indexing="ij")
    inside = (dx_grid**2 + dy_grid**2) <= radius**2

    for i in range(n):
        ix = int(np.round(x[i]))
        iy = int(np.round(y[i]))
        window = padded[iy : iy + 2 * pad + 1, ix : ix + 2 * pad + 1]
        if window.shape != inside.shape:
            continue

        # Offset of the true centroid from the pixel the window centres on.
        dx = dx_grid - (x[i] - ix)
        dy = dy_grid - (y[i] - iy)

        weights = np.where(inside, np.maximum(window, 0.0), 0.0)
        total = weights.sum()
        if total <= 0:
            # No positive flux in the aperture: report a circular source
            # the size of the aperture rather than a divide-by-zero.
            a[i] = b[i] = radius / 2.0
            continue

        mx = (weights * dx).sum() / total
        my = (weights * dy).sum() / total
        cx = dx - mx
        cy = dy - my
        i_xx = (weights * cx * cx).sum() / total
        i_yy = (weights * cy * cy).sum() / total
        i_xy = (weights * cx * cy).sum() / total

        half_sum = 0.5 * (i_xx + i_yy)
        half_diff = 0.5 * (i_xx - i_yy)
        root = np.sqrt(max(half_diff * half_diff + i_xy * i_xy, 0.0))
        lam1 = max(half_sum + root, 0.0)
        lam2 = max(half_sum - root, 0.0)

        a[i] = np.sqrt(lam1)
        b[i] = np.sqrt(lam2)
        theta[i] = 0.5 * np.arctan2(2.0 * i_xy, i_xx - i_yy)

    return a, b, theta


class SourceExtractor:
    """Extract sources (stars, galaxies) from astronomical images."""

    # The moment window has to be wider than the photometric aperture.
    # A hard aperture caps the measurable extent, so measuring shape in the
    # same 5px circle used for flux makes every source look equally round
    # and the star/galaxy split degenerates to "everything is a star".
    # Measured on the synthetic field: at 1.0x the aperture the classifier
    # returned 38/38 stars, against 31/38 for the SEP shape parameters it
    # replaces. At 1.6x it returns 33/38, the closest non-degenerate match,
    # with the highest correlation to SEP's Kron radius (r = 0.77).
    MOMENT_RADIUS_FACTOR = 1.6

    def __init__(
        self,
        threshold: float = 3.0,
        min_area: int = 5,
        aperture_radius_px: float = 5.0,
        moment_radius_px: float | None = None,
    ):
        """
        Args:
            threshold: Detection threshold in units of background RMS.
            min_area: Minimum connected pixels for a detection.
            aperture_radius_px: Radius of the photometric aperture, in
                pixels. A fixed aperture is what makes the photometry
                reproducible; SEP's Kron radius cannot be used because it
                is computed from the unstable shape parameters.
            moment_radius_px: Radius of the window used for shape moments.
                Defaults to MOMENT_RADIUS_FACTOR times the photometric
                aperture. Must stay fixed to stay deterministic.
        """
        self.threshold = threshold
        self.min_area = min_area
        self.aperture_radius_px = aperture_radius_px
        self.moment_radius_px = (
            moment_radius_px
            if moment_radius_px is not None
            else aperture_radius_px * self.MOMENT_RADIUS_FACTOR
        )

    def extract(self, image: np.ndarray) -> dict[str, Any]:
        """Extract sources from an image.

        Args:
            image: 2D numpy array.

        Returns:
            Dict with 'sources' (structured array), 'n_sources', 'positions', 'fluxes'.
        """
        data = image.astype(np.float64)
        data = np.ascontiguousarray(data)

        try:
            return self._extract_sep(data)
        except (ImportError, Exception) as e:
            logger.debug(f"SEP unavailable ({e}), falling back to photutils")
            return self._extract_photutils(data)

    def _extract_sep(self, data: np.ndarray) -> dict[str, Any]:
        """Extract using SEP (Source Extractor Python)."""
        import sep

        # Estimate and subtract background
        bkg = sep.Background(data)
        data_sub = data - bkg.back()

        # Extract sources
        sources = sep.extract(
            data_sub,
            thresh=self.threshold,
            err=bkg.globalrms,
            minarea=self.min_area,
        )

        # Positions and counts are the reproducible part of SEP's answer.
        x = np.asarray(sources["x"], dtype=np.float64)
        y = np.asarray(sources["y"], dtype=np.float64)
        positions = np.column_stack([x, y])

        # Photometry: fixed circular aperture instead of sources["flux"].
        # sum_circle depends only on pixel values and positions, so it is
        # deterministic, whereas the segmentation flux is not.
        fluxes, _flux_err, _flag = sep.sum_circle(
            data_sub, x, y, self.aperture_radius_px, err=bkg.globalrms
        )
        fluxes = np.asarray(fluxes, dtype=np.float64)

        # Shape: our own moments instead of sources["a"/"b"/"theta"], in a
        # window wider than the photometric aperture so extent is visible.
        a, b, _theta = _aperture_moments(data_sub, x, y, self.moment_radius_px)

        ellipticity = 1 - b / np.maximum(a, 1e-10)
        fwhm = 2.0 * np.sqrt(np.log(2) * (a**2 + b**2))

        # Star/galaxy classification (simple: round=star, extended=galaxy)
        kronrad = np.sqrt(a * b)
        median_kronrad = np.median(kronrad) if len(kronrad) else 0.0
        star_mask = (ellipticity < 0.3) & (kronrad < median_kronrad * 1.5)

        logger.info(f"Extracted {len(sources)} sources ({star_mask.sum()} likely stars)")

        return {
            "sources": sources,
            "n_sources": len(sources),
            "positions": positions,
            "fluxes": fluxes,
            "ellipticity": ellipticity,
            "fwhm": fwhm,
            "star_mask": star_mask,
            "background_rms": float(bkg.globalrms),
        }

    def _extract_photutils(self, data: np.ndarray) -> dict[str, Any]:
        """Extract using photutils (fallback)."""
        from photutils.background import Background2D, MedianBackground
        from photutils.detection import DAOStarFinder

        # Background estimation
        try:
            bkg = Background2D(data, box_size=50, bkg_estimator=MedianBackground())
            data_sub = data - bkg.background
            rms = bkg.background_rms_median
        except Exception:
            median = np.median(data)
            data_sub = data - median
            rms = np.std(data)

        # Source detection
        finder = DAOStarFinder(fwhm=3.0, threshold=self.threshold * rms)
        table = finder(data_sub)

        if table is None or len(table) == 0:
            return {
                "sources": None,
                "n_sources": 0,
                "positions": np.empty((0, 2)),
                "fluxes": np.array([]),
                "background_rms": float(rms),
            }

        positions = np.column_stack([table["xcentroid"], table["ycentroid"]])
        fluxes = np.array(table["flux"])

        logger.info(f"Extracted {len(table)} sources (photutils)")

        return {
            "sources": table,
            "n_sources": len(table),
            "positions": positions,
            "fluxes": fluxes,
            "background_rms": float(rms),
        }

    def source_density(self, image: np.ndarray, grid_size: int = 8) -> np.ndarray:
        """Compute source density map on a grid.

        Returns:
            Grid of source counts.
        """
        result = self.extract(image)
        positions = result["positions"]

        h, w = image.shape[:2]
        density = np.zeros((grid_size, grid_size))

        if len(positions) == 0:
            return density

        x_bins = np.linspace(0, w, grid_size + 1)
        y_bins = np.linspace(0, h, grid_size + 1)

        # Vectorized: single histogram2d call replaces Python loop
        density, _, _ = np.histogram2d(
            positions[:, 1],
            positions[:, 0],
            bins=[y_bins, x_bins],
        )

        return density
