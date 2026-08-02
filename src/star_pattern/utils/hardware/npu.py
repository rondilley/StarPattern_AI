"""NPU (neural processing unit) detection and ONNX Runtime session setup.

Targets AMD Ryzen AI parts, which expose the inference accelerator through
the Linux amdxdna driver at /dev/accel/accelN. Detection deliberately
requires BOTH the kernel device and a usable ONNX Runtime execution
provider: a machine with the silicon but no provider cannot run anything,
and a machine with a provider but no silicon would fall back to CPU while
claiming NPU support.
"""

from __future__ import annotations

import functools
import glob
import os
from enum import Enum
from pathlib import Path
from typing import Any

from star_pattern.utils.logging import get_logger

logger = get_logger("utils.hardware.npu")

# ONNX Runtime execution providers that map to an actual NPU, in
# preference order.
_NPU_PROVIDERS: tuple[str, ...] = (
    "VitisAIExecutionProvider",
    "MIGraphXExecutionProvider",
    "DmlExecutionProvider",
)

_CPU_PROVIDER = "CPUExecutionProvider"

# Kernel driver name reported by AMD Ryzen AI parts.
_AMD_NPU_DRIVER = "amdxdna"


class NPUBackend(Enum):
    """Available NPU backend."""

    NONE = "none"
    RYZEN_AI = "ryzen_ai"


def detect_npu_hardware() -> dict[str, Any]:
    """Probe the kernel for an NPU accelerator device.

    Returns:
        Mapping with 'present' (bool), 'device_path' (str or None) and
        'driver' (str or None). On platforms without /dev/accel, such as
        Windows, this reports absent rather than raising.
    """
    absent: dict[str, Any] = {
        "present": False,
        "device_path": None,
        "driver": None,
    }

    devices = sorted(glob.glob("/dev/accel/accel*"))
    if not devices:
        return absent

    for device_path in devices:
        name = os.path.basename(device_path)
        driver_link = f"/sys/class/accel/{name}/device/driver"
        try:
            driver = os.path.basename(os.path.realpath(driver_link))
        except OSError as exc:
            logger.debug("Cannot resolve driver for %s: %r", device_path, exc)
            continue
        if driver == _AMD_NPU_DRIVER:
            return {
                "present": True,
                "device_path": device_path,
                "driver": driver,
            }

    # Accelerator nodes exist but none of them is a supported NPU.
    logger.debug("Accel devices present but no %s driver", _AMD_NPU_DRIVER)
    return absent


def _available_onnx_providers() -> list[str]:
    """Return the execution providers ONNX Runtime reports, or an empty list."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.debug("onnxruntime not installed; NPU support unavailable")
        return []
    try:
        return list(ort.get_available_providers())
    except (RuntimeError, OSError) as exc:
        logger.warning("Cannot query ONNX Runtime providers: %r", exc)
        return []


def detect_npu_backend() -> NPUBackend:
    """Detect the usable NPU backend.

    Requires both the kernel device and a matching ONNX Runtime execution
    provider. Either one alone cannot run an inference session.
    """
    if not detect_npu_hardware()["present"]:
        return NPUBackend.NONE
    available = set(_available_onnx_providers())
    if available.intersection(_NPU_PROVIDERS):
        return NPUBackend.RYZEN_AI
    logger.debug("NPU hardware present but no NPU execution provider available")
    return NPUBackend.NONE


@functools.lru_cache(maxsize=1)
def get_npu_backend() -> NPUBackend:
    """Cached NPU backend. Hardware does not change during a run."""
    return detect_npu_backend()


def has_npu() -> bool:
    """True when a usable NPU backend is available."""
    return get_npu_backend() != NPUBackend.NONE


def get_npu_providers() -> list[str]:
    """Return the ONNX Runtime provider list to use, best first.

    CPU is always present as the final fallback. NPU providers appear
    only when an NPU is genuinely usable, so a session never silently
    claims acceleration it does not have.
    """
    if not has_npu():
        return [_CPU_PROVIDER]
    available = _available_onnx_providers()
    ordered = [ep for ep in _NPU_PROVIDERS if ep in available]
    return [*ordered, _CPU_PROVIDER]


def get_npu_session_options() -> Any | None:
    """Return tuned ONNX Runtime session options, or None without an NPU."""
    if not has_npu():
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def create_npu_session(model_path: str | Path) -> Any | None:
    """Create an ONNX Runtime inference session for a model.

    Falls back to CPU-only execution when the NPU provider rejects the
    model, which is common for unsupported operator sets.

    Returns:
        An onnxruntime.InferenceSession, or None when the model file is
        missing, ONNX Runtime is unavailable, or the session cannot be
        built at all.
    """
    path = Path(model_path)
    if not path.is_file():
        logger.debug("ONNX model not found: %s", path)
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed; cannot create session")
        return None

    providers = get_npu_providers()
    try:
        return ort.InferenceSession(
            str(path),
            sess_options=get_npu_session_options(),
            providers=providers,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        if providers == [_CPU_PROVIDER]:
            logger.error("Cannot create ONNX session for %s: %r", path, exc)
            return None
        logger.warning(
            "NPU providers %s rejected %s (%r); retrying on CPU",
            providers[:-1],
            path,
            exc,
        )

    try:
        return ort.InferenceSession(str(path), providers=[_CPU_PROVIDER])
    except (RuntimeError, ValueError, OSError) as exc:
        logger.error("Cannot create CPU ONNX session for %s: %r", path, exc)
        return None
