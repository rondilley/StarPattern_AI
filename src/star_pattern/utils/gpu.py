"""GPU and NPU detection, device management, and accelerated operations.

This module is the stable import path for accelerator support. The
implementation lives in star_pattern.utils.hardware, split by which
optional dependency each part needs: backends.py owns torch and CuPy,
npu.py owns ONNX Runtime, and ops.py owns the torch array operations.
Everything is re-exported here so existing imports keep working.
"""

from star_pattern.utils.hardware import (
    GPUBackend,
    NPUBackend,
    create_npu_session,
    detect_gpu_backend,
    detect_npu_backend,
    detect_npu_hardware,
    get_array_module,
    get_device,
    get_gpu_backend,
    get_npu_backend,
    get_npu_providers,
    get_npu_session_options,
    gpu_edge_magnitude,
    gpu_fft2_power,
    gpu_fftconvolve_batch,
    gpu_memory_info,
    gpu_separable_convolve,
    hardware_summary,
    has_gpu,
    has_npu,
    has_rocm,
    to_device,
    to_numpy,
)

__all__ = [
    "GPUBackend",
    "NPUBackend",
    "create_npu_session",
    "detect_gpu_backend",
    "detect_npu_backend",
    "detect_npu_hardware",
    "get_array_module",
    "get_device",
    "get_gpu_backend",
    "get_npu_backend",
    "get_npu_providers",
    "get_npu_session_options",
    "gpu_edge_magnitude",
    "gpu_fft2_power",
    "gpu_fftconvolve_batch",
    "gpu_memory_info",
    "gpu_separable_convolve",
    "hardware_summary",
    "has_gpu",
    "has_npu",
    "has_rocm",
    "to_device",
    "to_numpy",
]
