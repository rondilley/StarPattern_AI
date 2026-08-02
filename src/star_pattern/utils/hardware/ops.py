"""GPU-accelerated array operations for the detection hot paths.

Implemented against torch rather than CuPy so that one code path serves
both CUDA and ROCm machines. Every operation returns None when no GPU is
available, and every caller keeps its existing CPU implementation as the
fallback, so behaviour is identical on a CPU-only machine.

Precision is chosen per operation to match the CPU reference the callers
and tests compare against: float64 where the reference is numpy or scipy
in double precision, float32 where the reference is scipy's float32
fftconvolve.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from star_pattern.utils.hardware.backends import get_device, has_gpu
from star_pattern.utils.logging import get_logger

logger = get_logger("utils.hardware.ops")

# scipy.ndimage boundary modes and how to reproduce each one by padding.
# Note that scipy's "reflect" is half-sample symmetric (d c b a | a b c d)
# while torch's F.pad(mode="reflect") is whole-sample (d c b | a b c d),
# which scipy calls "mirror". Getting this backwards shifts every edge
# pixel, so the symmetric case is padded by hand.
_SUPPORTED_MODES = frozenset({"reflect", "mirror", "nearest", "wrap", "constant"})


def _gpu_unavailable(name: str) -> None:
    logger.debug("%s: no GPU available, caller uses CPU path", name)


def _pad_axis(tensor: Any, pad: int, axis: int, mode: str) -> Any:
    """Pad one axis of a 2-D tensor to reproduce a scipy boundary mode."""
    import torch

    if pad <= 0:
        return tensor

    if mode == "reflect":  # scipy: d c b a | a b c d
        left = tensor.flip(axis).narrow(axis, tensor.shape[axis] - pad, pad)
        right = tensor.flip(axis).narrow(axis, 0, pad)
        return torch.cat([left, tensor, right], dim=axis)
    if mode == "mirror":  # scipy: d c b | a b c d | c b a
        left = tensor.narrow(axis, 1, pad).flip(axis)
        right = tensor.narrow(axis, tensor.shape[axis] - pad - 1, pad).flip(axis)
        return torch.cat([left, tensor, right], dim=axis)
    if mode == "nearest":
        left = tensor.narrow(axis, 0, 1).repeat_interleave(pad, dim=axis)
        right = tensor.narrow(axis, tensor.shape[axis] - 1, 1).repeat_interleave(pad, dim=axis)
        return torch.cat([left, tensor, right], dim=axis)
    if mode == "wrap":
        left = tensor.narrow(axis, tensor.shape[axis] - pad, pad)
        right = tensor.narrow(axis, 0, pad)
        return torch.cat([left, tensor, right], dim=axis)
    # constant
    shape = list(tensor.shape)
    shape[axis] = pad
    zeros = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([zeros, tensor, zeros], dim=axis)


def _convolve1d(tensor: Any, kernel: Any, axis: int, mode: str) -> Any:
    """Convolve a 2-D tensor along one axis, matching ndimage.convolve1d."""
    import torch.nn.functional as F

    k = kernel.shape[0]
    pad = k // 2
    padded = _pad_axis(tensor, pad, axis, mode)

    # conv2d is cross-correlation; ndimage.convolve1d is true convolution,
    # so the weights are reversed.
    weight = kernel.flip(0)
    weight = weight.view(1, 1, k, 1) if axis == 0 else weight.view(1, 1, 1, k)

    out = F.conv2d(padded.unsqueeze(0).unsqueeze(0), weight)
    return out.squeeze(0).squeeze(0)


def gpu_separable_convolve(
    data: np.ndarray,
    kernel: np.ndarray,
    mode: str = "reflect",
) -> np.ndarray | None:
    """Apply a 1-D kernel along both axes on the GPU.

    Matches ndimage.convolve1d(axis=1) followed by ndimage.convolve1d(axis=0).

    Args:
        data: 2-D input array.
        kernel: 1-D kernel of odd length.
        mode: scipy.ndimage boundary mode.

    Returns:
        The convolved array, or None when no GPU is available.
    """
    if not has_gpu():
        _gpu_unavailable("gpu_separable_convolve")
        return None

    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"Unsupported boundary mode: {mode}")

    kernel_1d = np.asarray(kernel, dtype=np.float64).ravel()
    if kernel_1d.size % 2 == 0:
        raise ValueError(f"Kernel length must be odd, got {kernel_1d.size}")

    try:
        import torch

        device = get_device()
        tensor = torch.as_tensor(
            np.ascontiguousarray(data, dtype=np.float64),
            dtype=torch.float64,
            device=device,
        )
        weights = torch.as_tensor(kernel_1d, dtype=torch.float64, device=device)

        result = _convolve1d(tensor, weights, axis=1, mode=mode)
        result = _convolve1d(result, weights, axis=0, mode=mode)
        return result.cpu().numpy()
    except RuntimeError as exc:
        # torch.cuda.OutOfMemoryError subclasses RuntimeError, so this
        # covers device exhaustion as well as kernel launch failures.
        logger.warning("gpu_separable_convolve failed, using CPU path: %r", exc)
        return None


def gpu_fft2_power(image: np.ndarray) -> np.ndarray | None:
    """Compute the shifted 2-D FFT power spectrum on the GPU.

    Equivalent to abs(fftshift(fft2(image))) ** 2. Runs in complex128 so
    the result matches the numpy reference to within 1e-10 relative.

    Returns:
        The power spectrum, or None when no GPU is available.
    """
    if not has_gpu():
        _gpu_unavailable("gpu_fft2_power")
        return None

    try:
        import torch

        tensor = torch.as_tensor(
            np.ascontiguousarray(image, dtype=np.float64),
            dtype=torch.float64,
            device=get_device(),
        )
        spectrum = torch.fft.fftshift(torch.fft.fft2(tensor))
        return (torch.abs(spectrum) ** 2).cpu().numpy()
    except RuntimeError as exc:
        logger.warning("gpu_fft2_power failed, using CPU path: %r", exc)
        return None


def gpu_fftconvolve_batch(
    image: np.ndarray,
    kernels: Sequence[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Convolve one image against a bank of kernels in a single batch.

    The image FFT is computed once per kernel shape and reused across
    every kernel of that shape, which is the whole point of batching a
    Gabor bank.

    Returns:
        Tuple of (stack, responses) where stack has shape
        (len(kernels), H, W) and responses[i] equals
        abs(scipy.signal.fftconvolve(image, kernels[i], mode="same")).
        None when no GPU is available.
    """
    if not has_gpu():
        _gpu_unavailable("gpu_fftconvolve_batch")
        return None
    if not kernels:
        return None

    try:
        import torch

        device = get_device()
        img = torch.as_tensor(
            np.ascontiguousarray(image, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        height, width = img.shape

        by_shape: dict[tuple[int, int], list[int]] = {}
        for i, kernel in enumerate(kernels):
            by_shape.setdefault(np.shape(kernel), []).append(i)

        out = np.empty((len(kernels), height, width), dtype=np.float32)

        for (kh, kw), indices in by_shape.items():
            full = (height + kh - 1, width + kw - 1)
            img_f = torch.fft.rfft2(img, s=full)

            stacked = torch.as_tensor(
                np.ascontiguousarray(
                    np.stack([np.asarray(kernels[i]) for i in indices]),
                    dtype=np.float32,
                ),
                dtype=torch.float32,
                device=device,
            )
            ker_f = torch.fft.rfft2(stacked, s=full)
            conv = torch.fft.irfft2(img_f.unsqueeze(0) * ker_f, s=full)

            # scipy's "same" keeps the centre window of the full result.
            row0, col0 = (kh - 1) // 2, (kw - 1) // 2
            cropped = torch.abs(conv[:, row0 : row0 + height, col0 : col0 + width])
            out[indices] = cropped.cpu().numpy()

        return out, [out[i] for i in range(len(kernels))]
    except RuntimeError as exc:
        logger.warning("gpu_fftconvolve_batch failed, using CPU path: %r", exc)
        return None


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    """Build a normalized 1-D Gaussian, matching scipy's truncate=4.0."""
    radius = int(4.0 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def gpu_edge_magnitude(
    image: np.ndarray,
    sigma: float = 2.0,
) -> np.ndarray | None:
    """Find edge pixels by thresholded Sobel gradient magnitude.

    Smooths with a Gaussian, takes the Sobel gradient magnitude, and
    keeps pixels above the 95th percentile, which yields roughly a 5%
    edge fraction.

    Returns:
        Boolean edge mask, or None when no GPU is available.
    """
    if not has_gpu():
        _gpu_unavailable("gpu_edge_magnitude")
        return None

    try:
        import torch
        import torch.nn.functional as F

        device = get_device()
        tensor = torch.as_tensor(
            np.ascontiguousarray(image, dtype=np.float64),
            dtype=torch.float64,
            device=device,
        )

        if sigma > 0:
            gauss = torch.as_tensor(_gaussian_kernel1d(sigma), dtype=torch.float64, device=device)
            tensor = _convolve1d(tensor, gauss, axis=1, mode="reflect")
            tensor = _convolve1d(tensor, gauss, axis=0, mode="reflect")

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float64,
            device=device,
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)

        padded = _pad_axis(tensor, 1, 0, "reflect")
        padded = _pad_axis(padded, 1, 1, "reflect")
        batch = padded.unsqueeze(0).unsqueeze(0)

        gx = F.conv2d(batch, sobel_x).squeeze(0).squeeze(0)
        gy = F.conv2d(batch, sobel_y).squeeze(0).squeeze(0)
        magnitude = torch.hypot(gx, gy).cpu().numpy()

        # Threshold on the host so the percentile matches the CPU path
        # exactly rather than relying on a different quantile algorithm.
        threshold = float(np.percentile(magnitude, 95))
        return magnitude > threshold
    except RuntimeError as exc:
        logger.warning("gpu_edge_magnitude failed, using CPU path: %r", exc)
        return None
