"""Classical computer vision detection: Gabor filters, FFT, Hough transforms."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

from star_pattern.utils.gpu import get_array_module, to_device, to_numpy
from star_pattern.utils.logging import get_logger

logger = get_logger("detection.classical")


class GaborFilterBank:
    """Multi-scale, multi-orientation Gabor filter bank for pattern detection."""

    def __init__(
        self,
        frequencies: list[float] | None = None,
        n_orientations: int = 8,
        sigma_ratio: float = 0.56,
    ):
        self.frequencies = frequencies or [0.05, 0.1, 0.2, 0.4]
        self.n_orientations = n_orientations
        self.sigma_ratio = sigma_ratio
        self._kernels: list[np.ndarray] = []
        self._build_kernels()

    def _build_kernels(self) -> None:
        """Build all Gabor kernels."""
        thetas = np.linspace(0, np.pi, self.n_orientations, endpoint=False)
        for freq in self.frequencies:
            sigma = self.sigma_ratio / freq
            size = int(6 * sigma) | 1  # Ensure odd
            for theta in thetas:
                kernel = self._make_gabor(size, sigma, theta, freq)
                self._kernels.append(kernel)

    @staticmethod
    def _make_gabor(
        size: int, sigma: float, theta: float, frequency: float
    ) -> np.ndarray:
        """Create a single Gabor kernel."""
        half = size // 2
        y, x = np.mgrid[-half : half + 1, -half : half + 1]
        x_rot = x * np.cos(theta) + y * np.sin(theta)
        y_rot = -x * np.sin(theta) + y * np.cos(theta)
        envelope = np.exp(-0.5 * (x_rot**2 + y_rot**2) / sigma**2)
        carrier = np.cos(2 * np.pi * frequency * x_rot)
        kernel = envelope * carrier
        return kernel.astype(np.float32)

    def apply(self, image: np.ndarray) -> dict[str, Any]:
        """Apply all Gabor filters and return responses.

        Uses CuPy GPU acceleration when available: transfers image to GPU once,
        runs all 32 convolutions there, transfers results back once.

        Returns:
            Dict with 'responses' (list of filtered images),
            'max_response' (max across all filters),
            'mean_energy' (average filter energy),
            'dominant_orientation' (angle of strongest response).
        """
        xp, use_gpu = get_array_module()

        if use_gpu:
            try:
                import cupyx.scipy.signal

                gpu_image = to_device(image.astype(np.float32), xp)
                responses_gpu = []
                for kernel in self._kernels:
                    gpu_kernel = to_device(kernel, xp)
                    resp = cupyx.scipy.signal.fftconvolve(gpu_image, gpu_kernel, mode="same")
                    responses_gpu.append(xp.abs(resp))

                stack = xp.stack(responses_gpu, axis=0)
                max_response = to_numpy(xp.max(stack, axis=0))
                mean_energy = to_numpy(xp.mean(stack, axis=0))

                best_idx = to_numpy(xp.argmax(stack, axis=0))
                responses = [to_numpy(r) for r in responses_gpu]
            except Exception:
                # Fall back to CPU if GPU fails
                use_gpu = False

        if not use_gpu:
            responses = []
            for kernel in self._kernels:
                resp = fftconvolve(image, kernel, mode="same")
                responses.append(np.abs(resp))

            stack = np.stack(responses, axis=0)
            max_response = np.max(stack, axis=0)
            mean_energy = np.mean(stack, axis=0)
            best_idx = np.argmax(stack, axis=0)

        # Dominant orientation at each pixel
        n_per_freq = self.n_orientations
        orientation_idx = best_idx % n_per_freq
        thetas = np.linspace(0, np.pi, n_per_freq, endpoint=False)
        dominant_orientation = thetas[orientation_idx]

        return {
            "responses": responses,
            "max_response": max_response,
            "mean_energy": mean_energy,
            "dominant_orientation": dominant_orientation,
        }


class FFTAnalyzer:
    """FFT-based spatial frequency analysis."""

    @staticmethod
    def power_spectrum(image: np.ndarray) -> np.ndarray:
        """Compute 2D power spectrum. Uses CuPy GPU when available."""
        xp, use_gpu = get_array_module()
        if use_gpu:
            try:
                gpu_img = to_device(image, xp)
                f = xp.fft.fft2(gpu_img)
                f_shift = xp.fft.fftshift(f)
                return to_numpy(xp.abs(f_shift) ** 2)
            except Exception:
                pass
        f = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f)
        return np.abs(f_shift) ** 2

    @staticmethod
    def radial_profile(power_spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute radially-averaged power spectrum.

        Returns:
            Tuple of (frequencies, power) arrays.
        """
        cy, cx = np.array(power_spectrum.shape) // 2
        y, x = np.mgrid[: power_spectrum.shape[0], : power_spectrum.shape[1]]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

        max_r = min(cx, cy)
        radial = np.zeros(max_r)
        counts = np.zeros(max_r)

        mask = r < max_r
        np.add.at(radial, r[mask], power_spectrum[mask])
        np.add.at(counts, r[mask], 1)

        counts[counts == 0] = 1
        radial /= counts

        freqs = np.arange(max_r) / max_r
        return freqs, radial

    def analyze(self, image: np.ndarray) -> dict[str, Any]:
        """Full FFT analysis of an image."""
        ps = self.power_spectrum(image)
        freqs, radial = self.radial_profile(ps)

        # Find dominant non-DC frequency
        radial_nodc = radial.copy()
        radial_nodc[:3] = 0  # Mask DC and very low freq
        dominant_freq_idx = np.argmax(radial_nodc)
        dominant_freq = freqs[dominant_freq_idx] if dominant_freq_idx > 0 else 0

        return {
            "power_spectrum": ps,
            "radial_profile": (freqs, radial),
            "dominant_frequency": dominant_freq,
            "total_power": float(np.sum(ps)),
        }


class HoughArcDetector:
    """Detect arc-like features using Hough transform."""

    def __init__(self, min_radius: int = 10, max_radius: int = 100, threshold: float = 0.5):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.threshold = threshold

    def detect_arcs(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect circular arcs in an image.

        Returns:
            List of detected arcs with center, radius, and strength.
        """
        # Downsample large images to keep runtime reasonable
        # Hough is O(n_radii * n_edge_points * n_thetas)
        max_dim = 512
        scale_factor = 1.0
        work_image = image
        if max(image.shape) > max_dim:
            scale_factor = max_dim / max(image.shape)
            new_h = max(1, int(image.shape[0] * scale_factor))
            new_w = max(1, int(image.shape[1] * scale_factor))
            work_image = ndimage.zoom(
                image.astype(np.float64),
                (new_h / image.shape[0], new_w / image.shape[1]),
                order=1,
            )

        # Scale radii to match downsampled image
        min_r = max(3, int(self.min_radius * scale_factor))
        max_r = max(min_r + 1, int(self.max_radius * scale_factor))

        # Edge detection
        edges = self._canny_edges(work_image)
        edge_points = np.argwhere(edges)

        if len(edge_points) == 0:
            return []

        # Subsample edge points if still too many
        max_edge_points = 5000
        if len(edge_points) > max_edge_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(edge_points), max_edge_points, replace=False)
            edge_points = edge_points[idx]

        # Hough circle accumulator -- step through radii
        step = max(1, (max_r - min_r) // 30)
        radii = np.arange(min_r, max_r + 1, step)
        arcs = []

        n_thetas = 72  # 5-degree steps instead of 1-degree
        thetas = np.linspace(0, 2 * np.pi, n_thetas, endpoint=False)
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)

        h, w = work_image.shape
        for radius in radii:
            accum = np.zeros(work_image.shape, dtype=np.float64)

            # Fully vectorized: all thetas at once via broadcasting
            dx_all = np.round(radius * cos_thetas).astype(int)  # (n_thetas,)
            dy_all = np.round(radius * sin_thetas).astype(int)  # (n_thetas,)

            # (n_thetas, n_edges) shifted coordinates
            shifted_y = edge_points[:, 0][None, :] - dy_all[:, None]
            shifted_x = edge_points[:, 1][None, :] - dx_all[:, None]

            # Bounds check
            valid = (
                (shifted_y >= 0) & (shifted_y < h)
                & (shifted_x >= 0) & (shifted_x < w)
            )

            # Flatten valid votes and accumulate in one pass
            y_valid = shifted_y[valid]
            x_valid = shifted_x[valid]
            np.add.at(accum, (y_valid, x_valid), 1)

            # Find peaks
            accum_max = accum.max()
            if accum_max < 1:
                continue
            accum_norm = accum / accum_max
            peaks = accum_norm > self.threshold

            if peaks.any():
                peak_coords = np.argwhere(peaks)
                strengths = accum_norm[peaks]
                best = np.argmax(strengths)
                # Scale coordinates back to original image space
                arcs.append(
                    {
                        "center_y": int(peak_coords[best, 0] / scale_factor),
                        "center_x": int(peak_coords[best, 1] / scale_factor),
                        "radius": int(radius / scale_factor),
                        "strength": float(strengths[best]),
                    }
                )

        # Sort by strength
        arcs.sort(key=lambda a: a["strength"], reverse=True)
        return arcs[:20]  # Top 20

    @staticmethod
    def _canny_edges(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """Simple edge detection using gradient magnitude.

        Uses CuPy GPU acceleration when available.
        """
        xp, use_gpu = get_array_module()
        if use_gpu:
            try:
                import cupyx.scipy.ndimage as cu_ndimage

                gpu_img = to_device(image.astype(np.float64), xp)
                smoothed = cu_ndimage.gaussian_filter(gpu_img, sigma=sigma)
                gy = cu_ndimage.sobel(smoothed, axis=0)
                gx = cu_ndimage.sobel(smoothed, axis=1)
                magnitude = xp.hypot(gx, gy)
                threshold = float(xp.percentile(magnitude, 95))
                return to_numpy(magnitude > threshold)
            except Exception:
                pass
        smoothed = ndimage.gaussian_filter(image.astype(np.float64), sigma=sigma)
        gy = ndimage.sobel(smoothed, axis=0)
        gx = ndimage.sobel(smoothed, axis=1)
        magnitude = np.hypot(gx, gy)
        threshold = np.percentile(magnitude, 95)
        return magnitude > threshold


class ClassicalDetector:
    """Combined classical CV detection pipeline."""

    def __init__(
        self,
        gabor_frequencies: list[float] | None = None,
        gabor_orientations: int = 8,
    ):
        self.gabor = GaborFilterBank(
            frequencies=gabor_frequencies, n_orientations=gabor_orientations
        )
        self.fft = FFTAnalyzer()
        self.hough = HoughArcDetector()

    def detect(
        self,
        image: np.ndarray,
        pixel_scale_arcsec: float | None = None,
    ) -> dict[str, Any]:
        """Run all classical detectors on an image.

        Args:
            image: 2D image array.
            pixel_scale_arcsec: Pixel scale in arcsec/pixel. If provided,
                Hough radii are converted from physical to pixel units.
        """
        logger.debug(f"Running classical detection on {image.shape}")

        # Scale Hough radii by pixel scale if known
        if pixel_scale_arcsec and pixel_scale_arcsec > 0:
            min_r = max(3, int(3.0 / pixel_scale_arcsec))
            max_r = max(10, int(30.0 / pixel_scale_arcsec))
            self.hough = HoughArcDetector(
                min_radius=min_r, max_radius=max_r
            )

        gabor_result = self.gabor.apply(image)
        fft_result = self.fft.analyze(image)
        arcs = self.hough.detect_arcs(image)

        # Compute summary scores
        # Normalize gabor_score to [0,1] by dividing by image intensity range
        max_response = gabor_result["max_response"]
        img_range = float(np.percentile(image, 99) - np.percentile(image, 1))
        if img_range > 0:
            gabor_score = float(np.clip(np.mean(max_response) / img_range, 0, 1))
        else:
            gabor_score = 0.0

        fft_score = float(fft_result["dominant_frequency"])
        arc_score = float(arcs[0]["strength"]) if arcs else 0.0

        # Top Hough arc votes (raw accumulator strength)
        hough_votes = float(arcs[0]["strength"]) if arcs else 0.0
        # Mean Gabor filter energy across the image
        gabor_energy = float(np.mean(gabor_result["mean_energy"]))

        return {
            "gabor": gabor_result,
            "fft": fft_result,
            "arcs": arcs,
            "hough_arcs": arcs,
            "gabor_score": gabor_score,
            "fft_score": fft_score,
            "arc_score": arc_score,
            "hough_votes": hough_votes,
            "gabor_energy": gabor_energy,
            "fft_power": float(fft_result["total_power"]),
            "classical_score": (gabor_score + arc_score) / 2,
        }
