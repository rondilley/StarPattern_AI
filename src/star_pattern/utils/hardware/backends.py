"""GPU backend detection and device management.

Supports CUDA and ROCm through PyTorch, with CuPy as an optional array
module on CUDA only. Every probe degrades to CPU rather than raising, so
the detection pipeline runs unchanged on a machine with no accelerator.
"""

from __future__ import annotations

import functools
from enum import Enum
from types import ModuleType
from typing import Any

import numpy as np

from star_pattern.utils.hardware.npu import detect_npu_hardware, get_npu_backend
from star_pattern.utils.logging import get_logger

logger = get_logger("utils.hardware.backends")

_device_cache: dict[str, Any] = {}


class GPUBackend(Enum):
    """Available GPU compute backend."""

    NONE = "none"
    CUDA = "cuda"
    ROCM = "rocm"


def detect_gpu_backend() -> GPUBackend:
    """Detect the GPU backend PyTorch is built against.

    ROCm builds of PyTorch report through the same torch.cuda namespace,
    so the two are told apart by torch.version.hip.
    """
    try:
        import torch
    except ImportError:
        logger.debug("torch not installed; no GPU backend")
        return GPUBackend.NONE

    try:
        if not torch.cuda.is_available():
            return GPUBackend.NONE
        # is_available() only reports that the CUDA runtime loaded. With
        # CUDA_VISIBLE_DEVICES empty it stays True while device_count()
        # is 0, and every subsequent device call raises "Invalid device
        # id". Requiring a visible device makes has_gpu() mean what
        # callers assume it means.
        if torch.cuda.device_count() < 1:
            logger.debug("CUDA runtime present but no visible devices")
            return GPUBackend.NONE
    except (RuntimeError, OSError, AssertionError) as exc:
        # A broken or mismatched driver stack raises here.
        logger.warning("GPU availability check failed: %r", exc)
        return GPUBackend.NONE

    return GPUBackend.ROCM if getattr(torch.version, "hip", None) else GPUBackend.CUDA


@functools.lru_cache(maxsize=1)
def get_gpu_backend() -> GPUBackend:
    """Cached GPU backend. Hardware does not change during a run."""
    return detect_gpu_backend()


def has_gpu() -> bool:
    """True when any GPU backend (CUDA or ROCm) is usable."""
    return get_gpu_backend() != GPUBackend.NONE


def has_rocm() -> bool:
    """True when the GPU backend is ROCm."""
    return get_gpu_backend() == GPUBackend.ROCM


def get_device(prefer_gpu: bool = True) -> Any:
    """Get the best available PyTorch device."""
    key = f"torch_{prefer_gpu}"
    if key in _device_cache:
        return _device_cache[key]

    import torch

    if prefer_gpu and has_gpu():
        # ROCm builds also use the "cuda" device string.
        device = torch.device("cuda")
        try:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        except (RuntimeError, AssertionError) as exc:
            logger.debug("Cannot read GPU name: %r", exc)
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    _device_cache[key] = device
    return device


@functools.lru_cache(maxsize=1)
def _probe_cupy() -> ModuleType | None:
    """Import CuPy and confirm it can actually allocate on the device.

    Cached: the previous implementation re-imported CuPy, allocated a
    probe array and logged on every call, so a 5-scale wavelet
    decomposition paid for five identical probes per image.
    """
    try:
        import cupy as cp
    except ImportError:
        logger.debug("CuPy not installed; using numpy")
        return None

    try:
        cp.array([1.0])
    except Exception as exc:  # noqa: BLE001 - CuPy raises driver-specific types
        # CuPy present but the CUDA runtime is unusable (driver mismatch,
        # no device, out of memory). Report it: silent CPU fallback here
        # looks like unexplained slowness.
        logger.warning("CuPy present but unusable, falling back to numpy: %r", exc)
        return None

    logger.info("Using CuPy for GPU-accelerated array operations")
    return cp


def get_array_module(prefer_gpu: bool = True) -> tuple[Any, bool]:
    """Get the numpy or cupy array module.

    Returns:
        Tuple of (module, is_gpu). CuPy is offered on CUDA only; there is
        no CuPy build for ROCm in this project's dependency set, so a
        ROCm machine correctly gets numpy here and reaches the GPU
        through the torch-based operations instead.
    """
    if not prefer_gpu:
        return np, False
    if get_gpu_backend() != GPUBackend.CUDA:
        return np, False
    cp = _probe_cupy()
    if cp is None:
        return np, False
    return cp, True


def to_device(arr: np.ndarray, xp: Any) -> Any:
    """Move a numpy array to the device belonging to the array module."""
    if xp is np:
        return arr
    return xp.asarray(arr)


def to_numpy(arr: Any) -> np.ndarray:
    """Convert a numpy, torch or cupy array to numpy.

    Torch tensors may carry gradients and live on the GPU, so they need
    detach and host transfer before the numpy view is valid.
    """
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "detach"):  # torch.Tensor
        return arr.detach().cpu().numpy()
    if hasattr(arr, "get"):  # cupy.ndarray
        return arr.get()
    return np.asarray(arr)


def gpu_memory_info() -> dict[str, float] | None:
    """Return GPU memory figures in MB, or None when no GPU is present."""
    try:
        import torch
    except ImportError:
        return None

    if not has_gpu():
        return None

    try:
        # The property is total_memory. The previous code read total_mem,
        # which raised AttributeError into a bare except, so this function
        # returned None even on a fully working CUDA device.
        return {
            "total": torch.cuda.get_device_properties(0).total_memory / 1e6,
            "allocated": torch.cuda.memory_allocated(0) / 1e6,
            "cached": torch.cuda.memory_reserved(0) / 1e6,
        }
    except (RuntimeError, AttributeError, AssertionError) as exc:
        logger.warning("Cannot read GPU memory info: %r", exc)
        return None


def hardware_summary() -> dict[str, Any]:
    """Summarize every accelerator this machine exposes."""
    gpu_backend = get_gpu_backend()
    torch_version: str | None = None
    cuda_version: str | None = None
    hip_version: str | None = None
    gpu_name: str | None = None

    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        hip_version = getattr(torch.version, "hip", None)
        if gpu_backend != GPUBackend.NONE:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except (RuntimeError, AssertionError) as exc:
                logger.debug("Cannot read GPU name: %r", exc)
    except ImportError:
        logger.debug("torch not installed; hardware summary is CPU only")

    memory = gpu_memory_info()

    return {
        "gpu_backend": gpu_backend.value,
        "npu_backend": get_npu_backend().value,
        "npu_hardware": detect_npu_hardware(),
        "gpu_name": gpu_name,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "hip_version": hip_version,
        "memory_mb": memory["total"] if memory else None,
    }
