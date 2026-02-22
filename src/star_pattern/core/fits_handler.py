"""FITS image handling with WCS support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D
    import astropy.units as u
except ImportError as e:
    raise ImportError("astropy is required: pip install astropy") from e


class FITSImage:
    """Wrapper around FITS image data with WCS and utility methods."""

    def __init__(
        self,
        data: np.ndarray,
        header: fits.Header | None = None,
        wcs: WCS | None = None,
        path: Path | None = None,
    ):
        self.data = data.astype(np.float32)
        self.header = header or fits.Header()
        self.wcs = wcs
        self.path = path

        if self.wcs is None and header is not None:
            try:
                self.wcs = WCS(header)
            except Exception:
                self.wcs = None

    @classmethod
    def from_file(cls, path: str | Path, hdu_index: int = 0) -> FITSImage:
        """Load a FITS image from disk."""
        path = Path(path)
        with fits.open(path) as hdul:
            # Find first HDU with image data
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and hdu.data.ndim >= 2:
                    hdu_index = i
                    break
            hdu = hdul[hdu_index]
            data = hdu.data.copy()
            header = hdu.header.copy()
        return cls(data=data, header=header, path=path)

    @classmethod
    def from_array(cls, data: np.ndarray, wcs: WCS | None = None) -> FITSImage:
        """Create from a numpy array."""
        return cls(data=data, wcs=wcs)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def center_coord(self) -> SkyCoord | None:
        """Get the sky coordinate of the image center."""
        if self.wcs is None:
            return None
        cy, cx = self.data.shape[0] / 2, self.data.shape[1] / 2
        return self.wcs.pixel_to_world(cx, cy)

    def cutout(self, center: SkyCoord, size: float, unit: str = "arcmin") -> FITSImage:
        """Extract a cutout centered on a sky coordinate.

        Args:
            center: Sky coordinate for cutout center.
            size: Size of the cutout.
            unit: Unit for size ('arcmin', 'arcsec', 'deg').
        """
        if self.wcs is None:
            raise ValueError("WCS required for sky-coordinate cutouts")

        size_angle = size * getattr(u, unit)
        cut = Cutout2D(self.data, center, size_angle, wcs=self.wcs)
        header = cut.wcs.to_header()
        return FITSImage(data=cut.data, header=fits.Header(header), wcs=cut.wcs)

    def normalize(self, method: str = "arcsinh") -> FITSImage:
        """Return a normalized copy of the image.

        Args:
            method: 'arcsinh', 'log', 'linear', or 'zscale'.
        """
        data = self.data.copy()
        # Replace NaN/inf
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if method == "arcsinh":
            data = np.arcsinh(data / max(np.median(np.abs(data)), 1e-10))
        elif method == "log":
            data = np.log1p(np.clip(data - data.min(), 0, None))
        elif method == "linear":
            vmin, vmax = np.percentile(data, [1, 99])
            data = np.clip((data - vmin) / max(vmax - vmin, 1e-10), 0, 1)
        elif method == "zscale":
            median = np.median(data)
            std = np.std(data)
            vmin = median - 2 * std
            vmax = median + 2 * std
            data = np.clip((data - vmin) / max(vmax - vmin, 1e-10), 0, 1)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return FITSImage(data=data, header=self.header, wcs=self.wcs, path=self.path)

    def to_tensor(self) -> Any:
        """Convert to PyTorch tensor (C, H, W) format."""
        import torch

        data = self.data.copy()
        data = np.nan_to_num(data, nan=0.0)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]  # Add channel dim
        return torch.from_numpy(data).float()

    def to_rgb(self) -> np.ndarray:
        """Convert to RGB uint8 array for display."""
        norm = self.normalize("linear")
        gray = (norm.data * 255).astype(np.uint8)
        if gray.ndim == 2:
            return np.stack([gray, gray, gray], axis=-1)
        return gray

    def save(self, path: str | Path) -> None:
        """Save to a FITS file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = self.header.copy()
        if self.wcs is not None:
            header.update(self.wcs.to_header())
        hdu = fits.PrimaryHDU(data=self.data, header=header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(str(path), overwrite=True)

    def pixel_scale(self) -> float | None:
        """Return pixel scale in arcsec/pixel, or None if no WCS."""
        if self.wcs is None:
            return None
        try:
            scales = self.wcs.proj_plane_pixel_scales()
            # scales may be Quantity objects; extract .value if so
            val = scales[0]
            if hasattr(val, "value"):
                val = val.value
            return float(val * 3600)  # deg to arcsec
        except Exception:
            return None

    def __repr__(self) -> str:
        shape_str = f"{self.shape}"
        wcs_str = "WCS" if self.wcs else "no WCS"
        return f"FITSImage({shape_str}, {wcs_str})"
