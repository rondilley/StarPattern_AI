"""Galaxy morphology analysis via embeddings and classical measures."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from star_pattern.utils.logging import get_logger

logger = get_logger("detection.morphology")


class MorphologyAnalyzer:
    """Analyze galaxy morphology using concentration, asymmetry, smoothness (CAS)
    and optional deep learning embeddings."""

    def __init__(self, use_backbone: bool = False):
        self.use_backbone = use_backbone
        self._backbone = None

    def analyze(self, image: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
        """Compute morphology metrics for an image or cutout.

        Args:
            image: 2D array (single object cutout ideally).
            mask: Optional binary mask for the object.

        Returns:
            Dict with CAS parameters, Gini, M20, and optional embedding.
        """
        data = image.astype(np.float64)
        data = np.nan_to_num(data, nan=0.0)

        if mask is None:
            threshold = np.percentile(data, 80)
            mask = data > threshold

        results: dict[str, Any] = {}

        # CAS parameters
        results["concentration"] = self._concentration(data, mask)
        results["asymmetry"] = self._asymmetry(data)
        results["smoothness"] = self._smoothness(data)

        # Gini coefficient
        results["gini"] = self._gini(data, mask)

        # M20 (second-order moment of brightest 20% of pixels)
        results["m20"] = self._m20(data, mask)

        # Ellipticity from moments
        results["ellipticity"] = self._ellipticity(data, mask)

        # Z-scores relative to typical galaxy CAS parameters
        # Reference values from Conselice (2003) for normal galaxies:
        # C ~ 3.0 +/- 0.8, A ~ 0.05 +/- 0.06, S ~ 0.05 +/- 0.04, Gini ~ 0.45 +/- 0.10
        results["C_zscore"] = float((results["concentration"] - 3.0) / 0.8)
        results["A_zscore"] = float((results["asymmetry"] - 0.05) / 0.06)
        results["S_zscore"] = float((results["smoothness"] - 0.05) / 0.04)
        results["gini_zscore"] = float((results["gini"] - 0.45) / 0.10)

        # Morphology score: higher = more unusual
        results["morphology_score"] = self._score(results)

        # Optional: deep embedding
        if self.use_backbone:
            results["embedding"] = self._get_embedding(data)

        return results

    @staticmethod
    def _concentration(data: np.ndarray, mask: np.ndarray) -> float:
        """Concentration index: C = 5 * log10(r80/r20)."""
        cy, cx = ndimage.center_of_mass(data * mask)
        y, x = np.mgrid[: data.shape[0], : data.shape[1]]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        flux_vals = data[mask]
        r_vals = r[mask]
        if len(flux_vals) == 0:
            return 0.0

        # Sort by radius once, compute cumulative flux in radial order
        r_order = np.argsort(r_vals)
        sorted_r = r_vals[r_order]
        cumflux = np.cumsum(flux_vals[r_order])
        total = cumflux[-1] if len(cumflux) > 0 else 1

        # Find radii containing 20% and 80% of flux
        r20_idx = np.searchsorted(cumflux / total, 0.2)
        r80_idx = np.searchsorted(cumflux / total, 0.8)

        r20 = sorted_r[min(r20_idx, len(sorted_r) - 1)]
        r80 = sorted_r[min(r80_idx, len(sorted_r) - 1)]

        if r20 <= 0:
            return 0.0
        return float(5 * np.log10(max(r80 / r20, 1.0)))

    @staticmethod
    def _asymmetry(data: np.ndarray) -> float:
        """Rotational asymmetry: A = min(|I - I_180| / |I|)."""
        rotated = np.rot90(np.rot90(data))
        # Ensure same size
        h, w = min(data.shape[0], rotated.shape[0]), min(data.shape[1], rotated.shape[1])
        d = data[:h, :w]
        r = rotated[:h, :w]
        total = np.sum(np.abs(d))
        if total == 0:
            return 0.0
        return float(np.sum(np.abs(d - r)) / (2 * total))

    @staticmethod
    def _smoothness(data: np.ndarray, sigma: float = 3.0) -> float:
        """Smoothness (clumpiness): S = |I - I_smooth| / |I|."""
        smoothed = ndimage.gaussian_filter(data, sigma=sigma)
        residual = data - smoothed
        total = np.sum(np.abs(data))
        if total == 0:
            return 0.0
        return float(np.sum(np.abs(residual)) / total)

    @staticmethod
    def _gini(data: np.ndarray, mask: np.ndarray) -> float:
        """Gini coefficient of pixel flux distribution."""
        values = np.sort(np.abs(data[mask]))
        n = len(values)
        if n < 2 or np.sum(values) == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n)

    @staticmethod
    def _m20(data: np.ndarray, mask: np.ndarray) -> float:
        """M20 statistic: normalized second-order moment of brightest 20% of pixels."""
        flux = data[mask]
        if len(flux) == 0:
            return 0.0

        cy, cx = ndimage.center_of_mass(data * mask)
        y, x = np.mgrid[: data.shape[0], : data.shape[1]]
        r2 = (x - cx) ** 2 + (y - cy) ** 2

        # Total second-order moment
        m_tot = np.sum(flux * r2[mask])
        if m_tot == 0:
            return 0.0

        # Brightest 20%
        sorted_idx = np.argsort(flux)[::-1]
        cumflux = np.cumsum(flux[sorted_idx])
        n20 = np.searchsorted(cumflux, 0.2 * cumflux[-1]) + 1

        r2_masked = r2[mask]
        m20 = np.sum(flux[sorted_idx[:n20]] * r2_masked[sorted_idx[:n20]])

        return float(np.log10(max(m20 / m_tot, 1e-20)))

    @staticmethod
    def _ellipticity(data: np.ndarray, mask: np.ndarray) -> float:
        """Ellipticity from second moments."""
        y, x = np.mgrid[: data.shape[0], : data.shape[1]]
        weights = data * mask
        total = np.sum(weights)
        if total == 0:
            return 0.0

        cx = np.sum(x * weights) / total
        cy = np.sum(y * weights) / total

        mxx = np.sum((x - cx) ** 2 * weights) / total
        myy = np.sum((y - cy) ** 2 * weights) / total
        mxy = np.sum((x - cx) * (y - cy) * weights) / total

        trace = mxx + myy
        if trace == 0:
            return 0.0

        det = mxx * myy - mxy**2
        discriminant = max(trace**2 - 4 * det, 0)
        lambda1 = (trace + np.sqrt(discriminant)) / 2
        lambda2 = (trace - np.sqrt(discriminant)) / 2

        if lambda1 <= 0:
            return 0.0
        return float(1 - np.sqrt(max(lambda2, 0) / lambda1))

    @staticmethod
    def _score(metrics: dict[str, float]) -> float:
        """Compute overall morphology unusualness score."""
        # High asymmetry, high Gini, or extreme concentration = unusual
        a = metrics.get("asymmetry", 0)
        g = metrics.get("gini", 0)
        c = metrics.get("concentration", 0)
        s = metrics.get("smoothness", 0)
        e = metrics.get("ellipticity", 0)

        # Unusual if: high asymmetry, extreme Gini, high clumpiness
        score = 0.3 * min(a * 3, 1.0) + 0.25 * g + 0.2 * s + 0.15 * abs(c - 3.0) / 3.0 + 0.1 * e
        return float(np.clip(score, 0, 1))

    def _get_embedding(self, data: np.ndarray) -> np.ndarray | None:
        """Get deep learning embedding (if backbone available)."""
        try:
            from star_pattern.ml.backbone import BackboneWrapper

            if self._backbone is None:
                self._backbone = BackboneWrapper()
            return self._backbone.embed_image(data)
        except Exception as e:
            logger.debug(f"Backbone unavailable: {e}")
            return None
