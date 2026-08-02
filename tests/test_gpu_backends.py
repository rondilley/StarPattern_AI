"""Tests for multi-backend GPU/NPU detection and GPU-accelerated helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage
from scipy.signal import fftconvolve

from star_pattern.utils.gpu import (
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
    gpu_separable_convolve,
    hardware_summary,
    has_gpu,
    has_npu,
    has_rocm,
    to_numpy,
)


class TestBackendDetection:
    """Test backend enum types and detection consistency."""

    def test_gpu_backend_enum_values(self):
        assert GPUBackend.NONE.value == "none"
        assert GPUBackend.CUDA.value == "cuda"
        assert GPUBackend.ROCM.value == "rocm"

    def test_npu_backend_enum_values(self):
        assert NPUBackend.NONE.value == "none"
        assert NPUBackend.RYZEN_AI.value == "ryzen_ai"

    def test_detect_gpu_backend_returns_enum(self):
        result = detect_gpu_backend()
        assert isinstance(result, GPUBackend)

    def test_detect_npu_backend_returns_enum(self):
        result = detect_npu_backend()
        assert isinstance(result, NPUBackend)

    def test_cached_backend_matches_detect(self):
        assert get_gpu_backend() == detect_gpu_backend()

    def test_has_gpu_consistent_with_backend(self):
        backend = get_gpu_backend()
        if backend == GPUBackend.NONE:
            assert not has_gpu()
        else:
            assert has_gpu()

    def test_has_rocm_consistent_with_backend(self):
        backend = get_gpu_backend()
        if backend == GPUBackend.ROCM:
            assert has_rocm()
            assert has_gpu()  # ROCm is a GPU
        elif backend == GPUBackend.CUDA:
            assert not has_rocm()
            assert has_gpu()
        else:
            assert not has_rocm()
            assert not has_gpu()

    def test_has_npu_consistent_with_backend(self):
        backend = get_npu_backend()
        if backend == NPUBackend.NONE:
            assert not has_npu()
        else:
            assert has_npu()

    def test_mutual_exclusivity_cuda_rocm(self):
        """CUDA and ROCm cannot both be true."""
        backend = get_gpu_backend()
        if backend == GPUBackend.CUDA:
            assert not has_rocm()
        elif backend == GPUBackend.ROCM:
            # has_gpu() is True for ROCm, but not CUDA-specific
            assert has_rocm()


class TestGetDevice:
    """Test device selection."""

    def test_returns_torch_device(self):
        import torch

        device = get_device()
        assert isinstance(device, torch.device)

    def test_prefer_gpu_false_returns_cpu(self):
        import torch

        device = get_device(prefer_gpu=False)
        assert device == torch.device("cpu")

    def test_prefer_gpu_true_consistent(self):
        import torch

        device = get_device(prefer_gpu=True)
        if has_gpu():
            assert device == torch.device("cuda")
        else:
            assert device == torch.device("cpu")


class TestGetArrayModule:
    """Test array module selection."""

    def test_returns_tuple(self):
        result = get_array_module()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_prefer_cpu(self):
        xp, is_gpu = get_array_module(prefer_gpu=False)
        assert xp is np
        assert is_gpu is False

    def test_gpu_returns_cupy_only_on_cuda(self):
        xp, is_gpu = get_array_module(prefer_gpu=True)
        backend = get_gpu_backend()
        if backend == GPUBackend.CUDA:
            # May or may not have CuPy installed
            if is_gpu:
                assert xp is not np  # It's cupy
            else:
                assert xp is np
        elif backend == GPUBackend.ROCM:
            # CuPy not available on ROCm -- should return numpy
            assert xp is np
            assert is_gpu is False
        else:
            assert xp is np
            assert is_gpu is False


class TestToNumpy:
    """Test array conversion to numpy."""

    def test_numpy_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        assert result is arr

    def test_torch_tensor(self):
        import torch

        t = torch.tensor([1.0, 2.0, 3.0])
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_torch_tensor_requires_grad(self):
        import torch

        t = torch.tensor([1.0, 2.0], requires_grad=True)
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)


class TestHardwareSummary:
    """Test hardware summary report."""

    def test_returns_dict(self):
        result = hardware_summary()
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = hardware_summary()
        expected_keys = {
            "gpu_backend",
            "npu_backend",
            "npu_hardware",
            "gpu_name",
            "torch_version",
            "cuda_version",
            "hip_version",
            "memory_mb",
        }
        assert expected_keys == set(result.keys())

    def test_gpu_backend_value_matches_enum(self):
        result = hardware_summary()
        backend = get_gpu_backend()
        assert result["gpu_backend"] == backend.value

    def test_npu_backend_value_matches_enum(self):
        result = hardware_summary()
        backend = get_npu_backend()
        assert result["npu_backend"] == backend.value

    def test_torch_version_populated(self):
        result = hardware_summary()
        assert result["torch_version"] is not None

    def test_gpu_name_when_gpu_available(self):
        result = hardware_summary()
        if has_gpu():
            assert result["gpu_name"] is not None
            assert isinstance(result["gpu_name"], str)
        else:
            assert result["gpu_name"] is None


class TestNPUHooks:
    """Test NPU integration hooks."""

    def test_get_npu_providers_always_has_cpu(self):
        providers = get_npu_providers()
        assert "CPUExecutionProvider" in providers

    def test_get_npu_providers_npu_when_available(self):
        providers = get_npu_providers()
        npu_providers = {
            "VitisAIExecutionProvider",
            "MIGraphXExecutionProvider",
            "DmlExecutionProvider",
        }
        if has_npu():
            assert providers[0] in npu_providers
        else:
            assert not npu_providers.intersection(providers)

    def test_get_npu_session_options_none_without_npu(self):
        if not has_npu():
            assert get_npu_session_options() is None


class TestNPUHardwareDetection:
    """Test kernel-level NPU hardware detection."""

    def test_detect_npu_hardware_returns_dict(self):
        result = detect_npu_hardware()
        assert isinstance(result, dict)
        assert "present" in result
        assert "device_path" in result
        assert "driver" in result
        assert isinstance(result["present"], bool)

    def test_npu_hardware_consistent_with_dev_accel(self):
        """Cross-check detect_npu_hardware() against /dev/accel/ existence."""
        import glob

        result = detect_npu_hardware()
        accel_devices = glob.glob("/dev/accel/accel*")
        if not accel_devices:
            assert not result["present"]
            assert result["device_path"] is None
            assert result["driver"] is None
        # If devices exist but no amdxdna driver, present may still be False
        if result["present"]:
            assert result["device_path"] is not None
            assert result["driver"] == "amdxdna"

    def test_npu_hardware_in_summary(self):
        summary = hardware_summary()
        npu_hw = summary["npu_hardware"]
        assert isinstance(npu_hw, dict)
        assert "present" in npu_hw


# ---------------------------------------------------------------------------
# GPU helper tests -- marked @pytest.mark.gpu so they only run on GPU machines
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestGPUFftconvolveBatch:
    """Test batch FFT convolution GPU helper."""

    def test_returns_valid_result(self):
        rng = np.random.default_rng(42)
        image = rng.standard_normal((64, 64)).astype(np.float32)
        kernels = [rng.standard_normal((5, 5)).astype(np.float32) for _ in range(4)]

        result = gpu_fftconvolve_batch(image, kernels)
        if not has_gpu():
            assert result is None
            return

        assert result is not None
        stack, responses = result
        assert stack.shape == (4, 64, 64)
        assert len(responses) == 4
        for r in responses:
            assert r.shape == (64, 64)

    def test_numerical_accuracy_vs_scipy(self):
        if not has_gpu():
            pytest.skip("No GPU available")

        rng = np.random.default_rng(42)
        image = rng.standard_normal((32, 32)).astype(np.float32)
        kernels = [rng.standard_normal((5, 5)).astype(np.float32) for _ in range(3)]

        result = gpu_fftconvolve_batch(image, kernels)
        assert result is not None
        stack, responses = result

        for i, kernel in enumerate(kernels):
            cpu_resp = np.abs(fftconvolve(image, kernel, mode="same"))
            np.testing.assert_allclose(responses[i], cpu_resp, rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
class TestGPUFft2Power:
    """Test 2D FFT power spectrum GPU helper."""

    def test_returns_valid_result(self):
        rng = np.random.default_rng(42)
        image = rng.standard_normal((32, 32))

        result = gpu_fft2_power(image)
        if not has_gpu():
            assert result is None
            return

        assert result is not None
        assert result.shape == (32, 32)
        assert np.all(result >= 0)

    def test_numerical_accuracy_vs_numpy(self):
        if not has_gpu():
            pytest.skip("No GPU available")

        rng = np.random.default_rng(42)
        image = rng.standard_normal((32, 32))

        gpu_result = gpu_fft2_power(image)
        assert gpu_result is not None

        cpu_result = np.abs(np.fft.fftshift(np.fft.fft2(image))) ** 2
        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-10)


@pytest.mark.gpu
class TestGPUEdgeMagnitude:
    """Test GPU edge magnitude detection."""

    def test_returns_valid_result(self):
        rng = np.random.default_rng(42)
        image = rng.standard_normal((64, 64))

        result = gpu_edge_magnitude(image, sigma=2.0)
        if not has_gpu():
            assert result is None
            return

        assert result is not None
        assert result.shape == (64, 64)
        assert result.dtype == bool

    def test_edge_fraction_near_five_percent(self):
        """95th percentile threshold should yield roughly 5% edges."""
        if not has_gpu():
            pytest.skip("No GPU available")

        rng = np.random.default_rng(42)
        image = rng.standard_normal((128, 128))

        result = gpu_edge_magnitude(image, sigma=2.0)
        assert result is not None
        edge_fraction = np.mean(result)
        # Should be close to 5% (95th percentile threshold)
        assert 0.01 < edge_fraction < 0.15


@pytest.mark.gpu
class TestGPUSeparableConvolve:
    """Test GPU separable convolution."""

    def test_returns_valid_result(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((32, 32))
        kernel = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0

        result = gpu_separable_convolve(data, kernel, mode="reflect")
        if not has_gpu():
            assert result is None
            return

        assert result is not None
        assert result.shape == (32, 32)

    def test_numerical_accuracy_vs_scipy(self):
        if not has_gpu():
            pytest.skip("No GPU available")

        rng = np.random.default_rng(42)
        data = rng.standard_normal((32, 32))
        kernel = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0

        gpu_result = gpu_separable_convolve(data, kernel, mode="reflect")
        assert gpu_result is not None

        # CPU reference
        cpu_result = ndimage.convolve1d(data, kernel, axis=1, mode="reflect")
        cpu_result = ndimage.convolve1d(cpu_result, kernel, axis=0, mode="reflect")

        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-10)


class TestGPUCheckCLI:
    """Test the gpu-check CLI command."""

    def test_gpu_check_exits_cleanly(self):
        from click.testing import CliRunner

        from star_pattern.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["gpu-check"])
        assert result.exit_code == 0
        assert "Hardware Accelerator Status" in result.output

    def test_gpu_check_shows_pytorch_version(self):
        from click.testing import CliRunner

        from star_pattern.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["gpu-check"])
        assert result.exit_code == 0
        assert "PyTorch version" in result.output


class TestGPUHelpersWithoutGPU:
    """Test that all GPU helpers return None gracefully when no GPU is present."""

    def test_fftconvolve_batch_no_gpu(self):
        """On CPU-only systems, should return None."""
        if has_gpu():
            pytest.skip("GPU is available -- test is for CPU-only")

        rng = np.random.default_rng(42)
        image = rng.standard_normal((16, 16)).astype(np.float32)
        kernels = [rng.standard_normal((3, 3)).astype(np.float32)]
        assert gpu_fftconvolve_batch(image, kernels) is None

    def test_fft2_power_no_gpu(self):
        if has_gpu():
            pytest.skip("GPU is available -- test is for CPU-only")

        image = np.ones((8, 8))
        assert gpu_fft2_power(image) is None

    def test_edge_magnitude_no_gpu(self):
        if has_gpu():
            pytest.skip("GPU is available -- test is for CPU-only")

        image = np.ones((8, 8))
        assert gpu_edge_magnitude(image) is None

    def test_separable_convolve_no_gpu(self):
        if has_gpu():
            pytest.skip("GPU is available -- test is for CPU-only")

        data = np.ones((8, 8))
        kernel = np.array([1, 2, 1], dtype=np.float64) / 4.0
        assert gpu_separable_convolve(data, kernel) is None


class TestCreateNPUSession:
    """Test ONNX NPU session creation."""

    def test_returns_none_for_nonexistent_file(self):
        """create_npu_session returns None for a path that does not exist."""
        result = create_npu_session("/tmp/does_not_exist_12345.onnx")
        assert result is None

    def test_creates_session_for_valid_onnx(self):
        """Export a trivial model to ONNX, verify session creation."""
        import tempfile

        torch = pytest.importorskip("torch")
        onnxruntime = pytest.importorskip("onnxruntime")

        # Build a trivial model
        model = torch.nn.Linear(4, 2)
        model.eval()
        dummy = torch.randn(1, 4)

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "trivial.onnx"
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
            )

            session = create_npu_session(onnx_path)
            assert session is not None
            # Must have at least CPUExecutionProvider
            assert "CPUExecutionProvider" in session.get_providers()
