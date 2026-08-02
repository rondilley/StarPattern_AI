"""Hardware accelerator detection and GPU-accelerated operations.

Split by optional-dependency surface: backends.py owns torch and CuPy,
npu.py owns ONNX Runtime, ops.py owns the torch array operations. The
public names are re-exported here and again from star_pattern.utils.gpu,
which remains the stable import path for the rest of the package.
"""

from star_pattern.utils.hardware.backends import (
    GPUBackend,
    detect_gpu_backend,
    get_array_module,
    get_device,
    get_gpu_backend,
    gpu_memory_info,
    hardware_summary,
    has_gpu,
    has_rocm,
    to_device,
    to_numpy,
)
from star_pattern.utils.hardware.npu import (
    NPUBackend,
    create_npu_session,
    detect_npu_backend,
    detect_npu_hardware,
    get_npu_backend,
    get_npu_providers,
    get_npu_session_options,
    has_npu,
)
from star_pattern.utils.hardware.ops import (
    gpu_edge_magnitude,
    gpu_fft2_power,
    gpu_fftconvolve_batch,
    gpu_separable_convolve,
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
