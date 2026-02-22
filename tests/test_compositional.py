"""Tests for compositional detection operations and pipelines."""

import numpy as np
import pytest

from star_pattern.detection.compositional import (
    OperationSpec,
    OperationRegistry,
    PipelineSpec,
    ComposedPipeline,
    ComposedPipelineScorer,
    ALL_OPERATIONS,
)


def _make_synthetic_image(size=128, rng=None):
    """Create a synthetic test image with a bright central source."""
    rng = rng or np.random.default_rng(42)
    image = rng.normal(100, 10, (size, size)).astype(np.float64)
    # Add a bright central Gaussian source
    y, x = np.mgrid[0:size, 0:size]
    cy, cx = size // 2, size // 2
    source = 500 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * 10 ** 2))
    image += source
    return image


class TestOperationRegistry:
    """Tests for individual operations."""

    def test_all_operations_registered(self):
        """All 10 operations are in the registry."""
        registry = OperationRegistry()
        assert len(registry.names) == 10
        for name in ALL_OPERATIONS:
            assert name in registry.names

    def test_mask_sources(self):
        """mask_sources zeros out source positions."""
        registry = OperationRegistry()
        image = np.ones((64, 64), dtype=np.float64)
        context = {"positions": [[32, 32]]}
        op = OperationSpec("mask_sources", {"radius_px": 5})

        result, ctx = registry.execute(op, image, context)
        # Center should be zeroed
        assert result[32, 32] == 0
        # Corners should be untouched
        assert result[0, 0] == 1.0

    def test_subtract_model(self):
        """subtract_model produces residuals with near-zero mean for smooth images."""
        registry = OperationRegistry()
        image = _make_synthetic_image()
        context = {}
        op = OperationSpec("subtract_model", {"smooth_sigma": 1.0})

        result, ctx = registry.execute(op, image, context)
        assert ctx.get("model_subtracted") is True
        # Residual mean should be near zero
        assert abs(np.mean(result)) < 1.0

    def test_wavelet_residual(self):
        """wavelet_residual extracts detail at specified scale."""
        registry = OperationRegistry()
        image = _make_synthetic_image()
        context = {}
        op = OperationSpec("wavelet_residual", {"scale": 3})

        result, ctx = registry.execute(op, image, context)
        assert ctx["wavelet_scale"] == 3
        assert result.shape == image.shape

    def test_threshold_image(self):
        """threshold_image produces binary output."""
        registry = OperationRegistry()
        image = np.random.default_rng(42).random((64, 64))
        context = {}
        op = OperationSpec("threshold_image", {"percentile": 90})

        result, ctx = registry.execute(op, image, context)
        # Should be binary (0 or 1)
        unique = np.unique(result)
        assert set(unique).issubset({0.0, 1.0})
        # About 10% should be above threshold
        fraction = np.mean(result > 0)
        assert 0.05 < fraction < 0.2

    def test_convolve_kernel_gaussian(self):
        """convolve_kernel with gaussian smooths the image."""
        registry = OperationRegistry()
        image = np.zeros((64, 64))
        image[32, 32] = 1000.0  # Point source
        context = {}
        op = OperationSpec("convolve_kernel", {"kernel_size": 5, "type": "gaussian"})

        result, ctx = registry.execute(op, image, context)
        # Point source should be spread out
        assert result[32, 32] < 1000.0
        assert result[33, 33] > 0

    def test_convolve_kernel_laplacian(self):
        """convolve_kernel with laplacian detects edges."""
        registry = OperationRegistry()
        image = _make_synthetic_image()
        context = {}
        op = OperationSpec("convolve_kernel", {"kernel_size": 5, "type": "laplacian"})

        result, ctx = registry.execute(op, image, context)
        assert result.shape == image.shape

    def test_cross_correlate(self):
        """cross_correlate with source positions modulates the image."""
        registry = OperationRegistry()
        image = np.ones((64, 64))
        context = {"positions": [[32, 32], [10, 10]]}
        op = OperationSpec("cross_correlate", {"smooth_sigma": 5.0})

        result, ctx = registry.execute(op, image, context)
        assert ctx.get("density_correlated") is True
        # Near sources should have higher values
        assert result[32, 32] > result[0, 0]

    def test_combine_masks(self):
        """combine_masks performs logical operations."""
        registry = OperationRegistry()
        mask1 = np.zeros((64, 64))
        mask1[20:40, 20:40] = 1.0  # Square in center
        context = {"prev_mask": np.ones((64, 64))}  # Full mask

        # AND: result should be the center square
        op = OperationSpec("combine_masks", {"mode": "and"})
        result, ctx = registry.execute(op, mask1, context)
        assert result[30, 30] == 1.0
        assert result[0, 0] == 0.0

    def test_region_statistics_count(self):
        """region_statistics counts connected components."""
        registry = OperationRegistry()
        image = np.zeros((64, 64))
        image[10:15, 10:15] = 1.0
        image[40:45, 40:45] = 1.0
        context = {}
        op = OperationSpec("region_statistics", {"stat_type": "count"})

        result, ctx = registry.execute(op, image, context)
        assert ctx["n_components"] == 2

    def test_edge_detect_sobel(self):
        """edge_detect produces gradient magnitude."""
        registry = OperationRegistry()
        image = _make_synthetic_image()
        context = {}
        op = OperationSpec("edge_detect", {"method": "sobel", "threshold": 3.0})

        result, ctx = registry.execute(op, image, context)
        assert result.shape == image.shape
        assert ctx["edge_method"] == "sobel"

    def test_radial_profile_residual(self):
        """radial_profile_residual subtracts azimuthal average."""
        registry = OperationRegistry()
        image = _make_synthetic_image()
        context = {}
        op = OperationSpec("radial_profile_residual", {"center_method": "peak"})

        result, ctx = registry.execute(op, image, context)
        assert result.shape == image.shape
        assert "radial_center" in ctx
        # Residual should have smaller dynamic range than original
        assert np.std(result) < np.std(image)

    def test_unknown_operation(self):
        """Unknown operation returns input unchanged."""
        registry = OperationRegistry()
        image = np.ones((32, 32))
        context = {}
        op = OperationSpec("nonexistent_op", {})

        result, ctx = registry.execute(op, image, context)
        np.testing.assert_array_equal(result, image)


class TestComposedPipeline:
    """Tests for ComposedPipeline."""

    def test_pipeline_chaining(self):
        """Operations chain correctly: output of one feeds into next."""
        spec = PipelineSpec(
            operations=[
                OperationSpec("subtract_model", {"smooth_sigma": 5.0}),
                OperationSpec("threshold_image", {"percentile": 95}),
                OperationSpec("region_statistics", {"stat_type": "count"}),
            ],
            score_method="component_count",
        )
        pipeline = ComposedPipeline(spec)
        image = _make_synthetic_image()

        result = pipeline.run(image)
        assert "composed_score" in result
        assert 0.0 <= result["composed_score"] <= 1.0
        assert "pipeline_description" in result
        assert len(result["intermediate_results"]) == 3

    def test_pipeline_with_detection_context(self):
        """Pipeline uses source positions from detection results."""
        spec = PipelineSpec(
            operations=[
                OperationSpec("mask_sources", {"radius_px": 5}),
                OperationSpec("wavelet_residual", {"scale": 2}),
            ],
            score_method="max_residual",
        )
        pipeline = ComposedPipeline(spec)
        image = _make_synthetic_image()
        detection = {"sources": {"positions": [[64, 64], [32, 32]]}}

        result = pipeline.run(image, detection_results=detection)
        assert "composed_score" in result

    def test_empty_pipeline(self):
        """Pipeline with no operations still returns a valid result."""
        spec = PipelineSpec(operations=[], score_method="max_residual")
        pipeline = ComposedPipeline(spec)
        image = _make_synthetic_image()

        result = pipeline.run(image)
        assert "composed_score" in result
        assert len(result["intermediate_results"]) == 0


class TestComposedPipelineScorer:
    """Tests for scoring methods."""

    def test_component_count(self):
        """component_count scores based on number of regions."""
        # Image with 3 components
        image = np.zeros((64, 64))
        image[5:10, 5:10] = 1.0
        image[25:30, 25:30] = 1.0
        image[45:50, 45:50] = 1.0
        context = {"n_components": 3}

        score = ComposedPipelineScorer.component_count(image, context)
        assert 0.0 < score <= 1.0

    def test_max_residual(self):
        """max_residual scores based on peak signal."""
        image = np.random.default_rng(42).normal(0, 1, (64, 64))
        image[32, 32] = 20.0  # Strong signal
        context = {}

        score = ComposedPipelineScorer.max_residual(image, context)
        assert 0.0 < score <= 1.0

    def test_area_fraction(self):
        """area_fraction scores based on signal area."""
        image = np.zeros((100, 100))
        image[45:55, 45:55] = 1.0  # 1% of image
        context = {}

        score = ComposedPipelineScorer.area_fraction(image, context)
        assert 0.0 < score <= 1.0
