"""Curated preset pipelines for seeding the compositional evolution population."""

from __future__ import annotations

import numpy as np

from star_pattern.detection.compositional import OperationSpec
from star_pattern.discovery.pipeline_genome import PipelineGenome


def get_preset_pipelines(
    rng: np.random.Generator | None = None,
) -> list[PipelineGenome]:
    """Return 8 curated starting pipelines for compositional evolution."""
    rng = rng or np.random.default_rng()
    presets: list[PipelineGenome] = []

    # 1. Sersic residual analysis
    presets.append(
        PipelineGenome(
            operations=[
                OperationSpec("subtract_model", {"smooth_sigma": 5.0}),
                OperationSpec("threshold_image", {"percentile": 95}),
                OperationSpec("region_statistics", {"stat_type": "count"}),
            ],
            score_method="component_count",
            rng=rng,
        )
    )

    # 2. Source-subtracted wavelet detection
    presets.append(
        PipelineGenome(
            operations=[
                OperationSpec("mask_sources", {"radius_px": 5}),
                OperationSpec("wavelet_residual", {"scale": 3}),
                OperationSpec("threshold_image", {"percentile": 97}),
            ],
            score_method="component_count",
            rng=rng,
        )
    )

    # 3. Edge detection on smooth residual
    presets.append(
        PipelineGenome(
            operations=[
                OperationSpec("subtract_model", {"smooth_sigma": 3.0}),
                OperationSpec("edge_detect", {"method": "canny", "threshold": 5.0}),
                OperationSpec("threshold_image", {"percentile": 90}),
            ],
            score_method="area_fraction",
            rng=rng,
        )
    )

    # 4. Multi-scale radial profile residual
    presets.append(
        PipelineGenome(
            operations=[
                OperationSpec("radial_profile_residual", {"center_method": "peak"}),
                OperationSpec("wavelet_residual", {"scale": 2}),
                OperationSpec("threshold_image", {"percentile": 95}),
            ],
            score_method="max_residual",
            rng=rng,
        )
    )

    # 5. Cross-correlation of source density map
    presets.append(
        PipelineGenome(
            operations=[
                OperationSpec("subtract_model", {"smooth_sigma": 8.0}),
                OperationSpec("cross_correlate", {"smooth_sigma": 5.0}),
                OperationSpec("threshold_image", {"percentile": 92}),
            ],
            score_method="area_fraction",
            rng=rng,
        )
    )

    # 6. Threshold of combined wavelet scales
    presets.append(
        PipelineGenome(
            operations=[
                OperationSpec("wavelet_residual", {"scale": 2}),
                OperationSpec("convolve_kernel", {"kernel_size": 5, "type": "gaussian"}),
                OperationSpec("threshold_image", {"percentile": 93}),
                OperationSpec("region_statistics", {"stat_type": "count"}),
            ],
            score_method="component_count",
            rng=rng,
        )
    )

    # 7. Source mask -> radial profile -> residual
    presets.append(
        PipelineGenome(
            operations=[
                OperationSpec("mask_sources", {"radius_px": 8}),
                OperationSpec("radial_profile_residual", {"center_method": "centroid"}),
                OperationSpec("threshold_image", {"percentile": 96}),
            ],
            score_method="max_residual",
            rng=rng,
        )
    )

    # 8. Convolve kernel -> threshold -> count
    presets.append(
        PipelineGenome(
            operations=[
                OperationSpec("convolve_kernel", {"kernel_size": 7, "type": "laplacian"}),
                OperationSpec("threshold_image", {"percentile": 90}),
                OperationSpec("region_statistics", {"stat_type": "count"}),
            ],
            score_method="component_count",
            rng=rng,
        )
    )

    return presets
