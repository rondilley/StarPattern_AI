"""Source extraction using SEP and photutils."""

from __future__ import annotations

from typing import Any

import numpy as np

from star_pattern.utils.logging import get_logger

logger = get_logger("detection.source_extraction")


class SourceExtractor:
    """Extract sources (stars, galaxies) from astronomical images."""

    def __init__(self, threshold: float = 3.0, min_area: int = 5):
        self.threshold = threshold
        self.min_area = min_area

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

        positions = np.column_stack([sources["x"], sources["y"]])
        fluxes = sources["flux"]

        # Compute ellipticity and FWHM
        a = sources["a"]
        b = sources["b"]
        ellipticity = 1 - b / np.maximum(a, 1e-10)
        fwhm = 2.0 * np.sqrt(np.log(2) * (a**2 + b**2))

        # Star/galaxy classification (simple: round=star, extended=galaxy)
        kronrad = np.sqrt(a * b)
        star_mask = (ellipticity < 0.3) & (kronrad < np.median(kronrad) * 1.5)

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
        from photutils.detection import DAOStarFinder
        from photutils.background import Background2D, MedianBackground

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
            positions[:, 1], positions[:, 0],
            bins=[y_bins, x_bins],
        )

        return density
