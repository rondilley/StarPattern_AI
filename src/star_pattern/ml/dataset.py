"""FITS -> PyTorch Dataset with augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import torch
from torch.utils.data import Dataset

from star_pattern.core.fits_handler import FITSImage
from star_pattern.utils.logging import get_logger

logger = get_logger("ml.dataset")


class FITSDataset(Dataset):
    """PyTorch Dataset for FITS images."""

    def __init__(
        self,
        paths: list[str | Path],
        labels: list[int] | None = None,
        transform: Any = None,
        normalize: str = "arcsinh",
        target_size: int = 224,
    ):
        self.paths = [Path(p) for p in paths]
        self.labels = labels
        self.transform = transform
        self.normalize = normalize
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]

        try:
            img = FITSImage.from_file(path)
            img = img.normalize(self.normalize)
            data = img.data
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            data = np.zeros((self.target_size, self.target_size), dtype=np.float32)

        # Resize if needed
        if data.shape[0] != self.target_size or data.shape[1] != self.target_size:
            from scipy.ndimage import zoom

            zy = self.target_size / data.shape[0]
            zx = self.target_size / data.shape[1]
            data = zoom(data, (zy, zx), order=1)

        # Apply augmentation
        if self.transform:
            data = self.transform(data)

        # To tensor: (1, H, W)
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data[np.newaxis]).float()
        else:
            tensor = data.unsqueeze(0) if data.dim() == 2 else data

        result = {"image": tensor, "path": str(path)}

        if self.labels is not None:
            result["label"] = self.labels[idx]

        return result

    @classmethod
    def from_directory(
        cls,
        data_dir: str | Path,
        **kwargs: Any,
    ) -> FITSDataset:
        """Create dataset from a directory of FITS files.

        If subdirectories exist, use them as class labels.
        """
        data_dir = Path(data_dir)
        paths: list[Path] = []
        labels: list[int] = []

        subdirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]

        if subdirs:
            for label_idx, subdir in enumerate(subdirs):
                fits_files = sorted(subdir.glob("*.fits")) + sorted(subdir.glob("*.fits.gz"))
                paths.extend(fits_files)
                labels.extend([label_idx] * len(fits_files))
            logger.info(f"Found {len(paths)} FITS files in {len(subdirs)} classes")
        else:
            paths = sorted(data_dir.glob("*.fits")) + sorted(data_dir.glob("*.fits.gz"))
            labels = None  # type: ignore[assignment]
            logger.info(f"Found {len(paths)} FITS files (unlabeled)")

        return cls(paths=paths, labels=labels if subdirs else None, **kwargs)


class AstroAugmentation:
    """Astronomy-specific data augmentation."""

    def __init__(
        self,
        rotation: bool = True,
        flip: bool = True,
        noise: bool = True,
        noise_level: float = 0.02,
    ):
        self.rotation = rotation
        self.flip = flip
        self.noise = noise
        self.noise_level = noise_level
        self._rng = np.random.default_rng()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        data = image.copy()

        if self.rotation:
            k = self._rng.integers(0, 4)
            data = np.rot90(data, k)

        if self.flip:
            if self._rng.random() > 0.5:
                data = np.fliplr(data)
            if self._rng.random() > 0.5:
                data = np.flipud(data)

        if self.noise:
            noise = self._rng.normal(0, self.noise_level, data.shape).astype(data.dtype)
            data = data + noise

        return np.ascontiguousarray(data)
