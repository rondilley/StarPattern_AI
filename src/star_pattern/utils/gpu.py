"""GPU detection and device management."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_device_cache: dict[str, Any] = {}


def has_gpu() -> bool:
    """Check if a CUDA GPU is available via PyTorch."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device(prefer_gpu: bool = True) -> Any:
    """Get the best available PyTorch device."""
    key = f"torch_{prefer_gpu}"
    if key in _device_cache:
        return _device_cache[key]

    import torch

    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    _device_cache[key] = device
    return device


def get_array_module(prefer_gpu: bool = True) -> tuple[Any, bool]:
    """Get numpy or cupy array module.

    Returns:
        Tuple of (module, is_gpu) where module is numpy or cupy.
    """
    if prefer_gpu:
        try:
            import cupy as cp

            # Test that GPU is actually usable
            cp.array([1.0])
            logger.info("Using CuPy for GPU-accelerated array operations")
            return cp, True
        except (ImportError, Exception):
            pass

    return np, False


def to_device(arr: np.ndarray, xp: Any) -> Any:
    """Move numpy array to device (cupy or numpy passthrough).

    Args:
        arr: Input numpy array.
        xp: Array module (numpy or cupy).

    Returns:
        Array on the target device.
    """
    if xp is np:
        return arr
    return xp.asarray(arr)


def to_numpy(arr: Any) -> np.ndarray:
    """Move array back to numpy (cupy.asnumpy or passthrough).

    Args:
        arr: Input array (numpy or cupy).

    Returns:
        Numpy array.
    """
    if isinstance(arr, np.ndarray):
        return arr
    return arr.get()  # cupy -> numpy


def gpu_memory_info() -> dict[str, float] | None:
    """Return GPU memory info in MB, or None if no GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "total": torch.cuda.get_device_properties(0).total_mem / 1e6,
            "allocated": torch.cuda.memory_allocated(0) / 1e6,
            "cached": torch.cuda.memory_reserved(0) / 1e6,
        }
    except (ImportError, Exception):
        return None
