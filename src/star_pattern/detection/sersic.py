"""Sersic profile fitting for galaxy morphology analysis.

The Sersic profile I(r) = I_e * exp(-b_n * ((r/r_e)^(1/n) - 1)) is the standard
parametric model for galaxy surface brightness. The Sersic index n distinguishes:
  n=1: exponential disk (spiral disk component)
  n=4: de Vaucouleurs profile (elliptical galaxies, bulges)
  n<1: sub-exponential (disk-dominated, LSBs)
  n>4: super-concentrated (cD galaxies, compact ellipticals)

This module fits 1D azimuthally-averaged radial profiles and optionally 2D
elliptical models to extract r_e (effective/half-light radius), n, ellipticity,
position angle, and residual images for substructure detection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit

from star_pattern.utils.logging import get_logger

logger = get_logger("detection.sersic")


def _sersic_bn(n: float) -> float:
    """Compute b_n approximation for Sersic profile.

    Uses Ciotti & Bertin (1999) approximation, accurate to <0.1% for n>0.36.
    """
    if n <= 0:
        return 0.0
    return 1.9992 * n - 0.3271


def sersic_1d(r: np.ndarray, I_e: float, r_e: float, n: float) -> np.ndarray:
    """1D Sersic profile: I(r) = I_e * exp(-b_n * ((r/r_e)^(1/n) - 1)).

    Args:
        r: Radial distance array.
        I_e: Surface brightness at effective radius.
        r_e: Effective (half-light) radius.
        n: Sersic index.

    Returns:
        Surface brightness at each radius.
    """
    bn = _sersic_bn(n)
    r_safe = np.maximum(r, 0.1)
    r_e_safe = max(r_e, 0.1)
    exponent = -bn * ((r_safe / r_e_safe) ** (1.0 / max(n, 0.1)) - 1.0)
    # Clip exponent to avoid overflow
    exponent = np.clip(exponent, -50, 50)
    return I_e * np.exp(exponent)


class SersicAnalyzer:
    """Fit Sersic profiles to galaxy images and analyze residuals.

    Extracts fundamental galaxy structural parameters:
    - Effective radius (r_e): half-light radius in pixels
    - Sersic index (n): profile shape parameter
    - Ellipticity: from isophote fitting
    - Residual structure: deviations from smooth Sersic model

    Parameters:
        max_radius_frac: Maximum radius for fitting as fraction of image half-size.
        n_radial_bins: Number of radial bins for 1D profile.
        residual_sigma: Significance threshold for residual features.
    """

    def __init__(
        self,
        max_radius_frac: float = 0.8,
        n_radial_bins: int = 50,
        residual_sigma: float = 3.0,
    ):
        self.max_radius_frac = max_radius_frac
        self.n_radial_bins = n_radial_bins
        self.residual_sigma = residual_sigma

    def analyze(
        self,
        data: np.ndarray,
        pixel_scale_arcsec: float | None = None,
    ) -> dict[str, Any]:
        """Fit Sersic profile and analyze residuals.

        Args:
            data: 2D image array.
            pixel_scale_arcsec: Pixel scale in arcsec/pixel for physical sizes.

        Returns:
            Dict with Sersic parameters, residual features, and sersic_score.
        """
        results: dict[str, Any] = {}

        # Find center (brightest smoothed pixel)
        smoothed = ndimage.gaussian_filter(data.astype(np.float64), sigma=3.0)
        cy, cx = np.unravel_index(np.argmax(smoothed), smoothed.shape)

        # Background estimation from corners
        corner_size = max(5, min(data.shape) // 10)
        corners = np.concatenate([
            data[:corner_size, :corner_size].ravel(),
            data[:corner_size, -corner_size:].ravel(),
            data[-corner_size:, :corner_size].ravel(),
            data[-corner_size:, -corner_size:].ravel(),
        ])
        bkg = float(np.median(corners))
        bkg_rms = float(np.std(corners))
        data_sub = data.astype(np.float64) - bkg

        # Compute elliptical parameters from second moments
        ellip_params = self._compute_ellipticity(data_sub, cx, cy)
        results["ellipticity"] = ellip_params["ellipticity"]
        results["position_angle"] = ellip_params["position_angle"]

        # Extract azimuthally-averaged radial profile
        max_r = int(min(data.shape) * self.max_radius_frac / 2)
        radii, profile, profile_err = self._radial_profile(
            data_sub, cx, cy,
            ellipticity=ellip_params["ellipticity"],
            pa=ellip_params["position_angle"],
            max_r=max_r,
        )

        results["radial_profile"] = {
            "radii": radii.tolist(),
            "intensity": profile.tolist(),
        }

        # Fit 1D Sersic profile
        fit_result = self._fit_sersic_1d(radii, profile, profile_err)
        results["fit"] = fit_result

        if fit_result["success"]:
            r_e = fit_result["r_e"]
            n = fit_result["n"]
            I_e = fit_result["I_e"]

            # Convert to physical units if pixel scale known
            if pixel_scale_arcsec and pixel_scale_arcsec > 0:
                results["r_e_arcsec"] = r_e * pixel_scale_arcsec
            else:
                results["r_e_arcsec"] = None

            # Build 2D Sersic model and compute residuals
            model = self._build_2d_model(
                data.shape, cx, cy, I_e, r_e, n,
                ellip_params["ellipticity"],
                ellip_params["position_angle"],
            )
            residual = data_sub - model

            # Analyze residual for substructure
            residual_features = self._analyze_residuals(
                residual, bkg_rms, cx, cy, r_e,
            )
            results["residual_features"] = residual_features

            # Classify morphology
            morph_class = self._classify_morphology(n, ellip_params["ellipticity"])
            results["morphology_class"] = morph_class

        else:
            results["residual_features"] = []
            results["morphology_class"] = "unknown"

        # Compute composite score
        results["sersic_score"] = self._compute_score(results)

        logger.info(
            f"Sersic analysis: n={fit_result.get('n', 0):.2f}, "
            f"r_e={fit_result.get('r_e', 0):.1f}px, "
            f"class={results['morphology_class']}, "
            f"score={results['sersic_score']:.3f}"
        )
        return results

    def _compute_ellipticity(
        self, data: np.ndarray, cx: int, cy: int,
    ) -> dict[str, float]:
        """Compute ellipticity and position angle from intensity-weighted moments."""
        max_r = min(data.shape) // 4
        y, x = np.mgrid[:data.shape[0], :data.shape[1]]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = (r < max_r) & (data > 0)

        if mask.sum() < 10:
            return {"ellipticity": 0.0, "position_angle": 0.0}

        weights = data[mask]
        total_w = weights.sum()
        if total_w < 1e-10:
            return {"ellipticity": 0.0, "position_angle": 0.0}

        dx = (x[mask] - cx).astype(np.float64)
        dy = (y[mask] - cy).astype(np.float64)

        Ixx = np.sum(weights * dx * dx) / total_w
        Iyy = np.sum(weights * dy * dy) / total_w
        Ixy = np.sum(weights * dx * dy) / total_w

        # Eigenvalues of inertia tensor
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy * Ixy
        discriminant = max(trace * trace / 4 - det, 0)
        sqrt_disc = np.sqrt(discriminant)

        lambda1 = trace / 2 + sqrt_disc
        lambda2 = trace / 2 - sqrt_disc

        if lambda1 <= 0:
            return {"ellipticity": 0.0, "position_angle": 0.0}

        ellipticity = float(1.0 - np.sqrt(max(lambda2, 0) / lambda1))
        position_angle = float(0.5 * np.degrees(np.arctan2(2 * Ixy, Ixx - Iyy)))

        return {"ellipticity": ellipticity, "position_angle": position_angle}

    def _radial_profile(
        self,
        data: np.ndarray,
        cx: int,
        cy: int,
        ellipticity: float = 0.0,
        pa: float = 0.0,
        max_r: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract azimuthally-averaged radial profile with elliptical apertures."""
        y, x = np.mgrid[:data.shape[0], :data.shape[1]]
        dx = (x - cx).astype(np.float64)
        dy = (y - cy).astype(np.float64)

        # Rotate to align with major axis
        pa_rad = np.radians(pa)
        dx_rot = dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
        dy_rot = -dx * np.sin(pa_rad) + dy * np.cos(pa_rad)

        # Elliptical radius
        q = max(1.0 - ellipticity, 0.1)  # axis ratio
        r_ellip = np.sqrt(dx_rot ** 2 + (dy_rot / q) ** 2)

        # Bin into radial annuli
        n_bins = min(self.n_radial_bins, max_r)
        bin_edges = np.linspace(0, max_r, n_bins + 1)
        radii = (bin_edges[:-1] + bin_edges[1:]) / 2
        profile = np.zeros(n_bins)
        profile_err = np.zeros(n_bins)

        # Vectorized binning via digitize (avoids per-bin 2D boolean mask)
        r_flat = r_ellip.ravel()
        data_flat = data.ravel()

        # Exclude pixels outside [0, max_r) to match original behavior
        in_range = (r_flat >= 0) & (r_flat < max_r)
        r_in = r_flat[in_range]
        d_in = data_flat[in_range]

        bin_idx = np.digitize(r_in, bin_edges) - 1  # 0-based bin index
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        for i in range(n_bins):
            mask = bin_idx == i
            count = mask.sum()
            if count > 0:
                values = d_in[mask]
                profile[i] = np.median(values)
                profile_err[i] = np.std(values) / max(np.sqrt(count), 1)
            else:
                profile[i] = 0
                profile_err[i] = 1e10

        return radii, profile, profile_err

    def _fit_sersic_1d(
        self,
        radii: np.ndarray,
        profile: np.ndarray,
        profile_err: np.ndarray,
    ) -> dict[str, Any]:
        """Fit 1D Sersic profile using nonlinear least squares."""
        # Filter to positive profile values
        mask = (profile > 0) & (radii > 0) & np.isfinite(profile_err) & (profile_err > 0)
        if mask.sum() < 5:
            return {"success": False, "reason": "insufficient_data"}

        r_fit = radii[mask]
        I_fit = profile[mask]
        sigma_fit = profile_err[mask]
        sigma_fit = np.maximum(sigma_fit, 1e-10)

        # Initial guesses
        I_e_guess = float(I_fit[len(I_fit) // 2])  # intensity at mid-radius
        r_e_guess = float(r_fit[len(r_fit) // 2])   # half-light at mid-radius
        n_guess = 2.0  # moderate profile

        try:
            popt, pcov = curve_fit(
                sersic_1d,
                r_fit,
                I_fit,
                p0=[I_e_guess, r_e_guess, n_guess],
                sigma=sigma_fit,
                bounds=([0.0, 0.1, 0.1], [np.inf, r_fit[-1] * 2, 10.0]),
                maxfev=5000,
            )

            I_e, r_e, n = popt
            perr = np.sqrt(np.diag(pcov))

            # Compute fit quality (reduced chi-squared)
            model = sersic_1d(r_fit, *popt)
            chi2 = np.sum(((I_fit - model) / sigma_fit) ** 2)
            dof = max(len(r_fit) - 3, 1)
            reduced_chi2 = float(chi2 / dof)

            return {
                "success": True,
                "I_e": float(I_e),
                "r_e": float(r_e),
                "n": float(n),
                "I_e_err": float(perr[0]),
                "r_e_err": float(perr[1]),
                "n_err": float(perr[2]),
                "reduced_chi2": reduced_chi2,
            }

        except (RuntimeError, ValueError) as e:
            logger.debug(f"Sersic fit failed: {e}")
            return {"success": False, "reason": str(e)}

    def _build_2d_model(
        self,
        shape: tuple[int, ...],
        cx: int,
        cy: int,
        I_e: float,
        r_e: float,
        n: float,
        ellipticity: float,
        pa: float,
    ) -> np.ndarray:
        """Build a 2D Sersic model image."""
        # Work in a cutout for efficiency
        max_r = int(r_e * 5) + 10
        y0 = max(0, cy - max_r)
        y1 = min(shape[0], cy + max_r)
        x0 = max(0, cx - max_r)
        x1 = min(shape[1], cx + max_r)

        y, x = np.mgrid[y0:y1, x0:x1]
        dx = (x - cx).astype(np.float64)
        dy = (y - cy).astype(np.float64)

        pa_rad = np.radians(pa)
        dx_rot = dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
        dy_rot = -dx * np.sin(pa_rad) + dy * np.cos(pa_rad)

        q = max(1.0 - ellipticity, 0.1)
        r_ellip = np.sqrt(dx_rot ** 2 + (dy_rot / q) ** 2)

        model_full = np.zeros(shape, dtype=np.float64)
        cutout_model = sersic_1d(r_ellip, I_e, r_e, n)
        model_full[y0:y1, x0:x1] = cutout_model

        return model_full

    def _analyze_residuals(
        self,
        residual: np.ndarray,
        bkg_rms: float,
        cx: int,
        cy: int,
        r_e: float,
    ) -> list[dict[str, Any]]:
        """Find significant features in the Sersic-subtracted residual image."""
        features = []

        if bkg_rms < 1e-10:
            return features

        # Significance map
        snr_map = residual / max(bkg_rms, 1e-10)

        # Smooth to find extended features
        smoothed_snr = ndimage.gaussian_filter(snr_map, sigma=2.0)

        # Threshold for positive residuals (excess light)
        positive_mask = smoothed_snr > self.residual_sigma
        labeled_pos, n_pos = ndimage.label(positive_mask)

        # Collect all positive candidates (no iteration cap)
        pos_candidates = []
        for label_idx in range(1, n_pos + 1):
            region = labeled_pos == label_idx
            area = int(np.sum(region))
            if area < 5:
                continue
            ys, xs = np.where(region)
            peak_snr = float(np.max(smoothed_snr[region]))
            mean_x = float(np.mean(xs))
            mean_y = float(np.mean(ys))
            dist_from_center = float(np.sqrt((mean_x - cx) ** 2 + (mean_y - cy) ** 2))

            pos_candidates.append({
                "type": "excess_light",
                "x": mean_x,
                "y": mean_y,
                "area_px": area,
                "peak_snr": peak_snr,
                "dist_from_center": dist_from_center,
                "dist_in_re": dist_from_center / max(r_e, 1),
            })

        # Keep top 20 by SNR so the most significant features survive
        pos_candidates.sort(key=lambda f: abs(f["peak_snr"]), reverse=True)
        features.extend(pos_candidates[:20])

        # Negative residuals (deficit = absorption or tidal stripping)
        negative_mask = smoothed_snr < -self.residual_sigma
        labeled_neg, n_neg = ndimage.label(negative_mask)

        neg_candidates = []
        for label_idx in range(1, n_neg + 1):
            region = labeled_neg == label_idx
            area = int(np.sum(region))
            if area < 5:
                continue
            ys, xs = np.where(region)
            peak_snr = float(np.min(smoothed_snr[region]))
            mean_x = float(np.mean(xs))
            mean_y = float(np.mean(ys))

            neg_candidates.append({
                "type": "light_deficit",
                "x": mean_x,
                "y": mean_y,
                "area_px": area,
                "peak_snr": peak_snr,
                "dist_from_center": float(
                    np.sqrt((mean_x - cx) ** 2 + (mean_y - cy) ** 2)
                ),
            })

        # Keep top 10 by absolute SNR
        neg_candidates.sort(key=lambda f: abs(f["peak_snr"]), reverse=True)
        features.extend(neg_candidates[:10])

        return features

    def _classify_morphology(self, n: float, ellipticity: float) -> str:
        """Classify galaxy morphology from Sersic index and ellipticity."""
        if n < 0.5:
            return "irregular"
        elif n < 1.5:
            if ellipticity > 0.5:
                return "edge-on_disk"
            return "disk/spiral"
        elif n < 2.5:
            return "lenticular"
        elif n < 5.0:
            if ellipticity < 0.2:
                return "elliptical"
            return "elliptical_elongated"
        else:
            return "compact/cD"

    def _compute_score(self, results: dict[str, Any]) -> float:
        """Compute sersic_score [0, 1] based on fit quality and residual interest."""
        score = 0.0

        fit = results.get("fit", {})
        if not fit.get("success"):
            return 0.0

        n = fit.get("n", 2.0)
        chi2 = fit.get("reduced_chi2", 10.0)

        # Good fit quality contributes
        if chi2 < 3.0:
            score += 0.2
        elif chi2 < 10.0:
            score += 0.1

        # Unusual Sersic index (very high or very low)
        if n < 0.5 or n > 6.0:
            score += 0.2
        elif n > 4.0:  # Elliptical-like
            score += 0.1

        # High ellipticity
        ellip = results.get("ellipticity", 0)
        if ellip > 0.6:
            score += 0.1

        # Significant residual features (substructure beyond smooth model)
        n_features = len(results.get("residual_features", []))
        score += min(n_features / 5, 0.3)

        # Features at large radii (tidal features, extended emission)
        for feat in results.get("residual_features", []):
            if feat.get("dist_in_re", 0) > 2.0 and feat.get("peak_snr", 0) > 5:
                score += 0.1
                break

        return float(np.clip(score, 0, 1))
