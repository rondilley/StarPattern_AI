"""Known-good parameter presets for seeding the initial population."""

from __future__ import annotations

import numpy as np

from star_pattern.discovery.genome import DetectionGenome, GENE_DEFINITIONS


def _make_genome(values: dict[str, float], rng: np.random.Generator) -> DetectionGenome:
    """Create a genome from named values, random for unspecified genes.

    All enable_* genes default to 1.0 (all detectors on) unless
    explicitly overridden in values.
    """
    genes = np.array([g.random(rng) for g in GENE_DEFINITIONS])
    # Default all enable gates to 1.0 (on) so presets don't randomly
    # disable detectors
    for i, gdef in enumerate(GENE_DEFINITIONS):
        if gdef.name.startswith("enable_") and gdef.name not in values:
            genes[i] = 1.0
    for i, gdef in enumerate(GENE_DEFINITIONS):
        if gdef.name in values:
            genes[i] = gdef.clip(values[gdef.name])
    return DetectionGenome(genes=genes, rng=rng)


def get_preset_genomes(rng: np.random.Generator | None = None) -> list[DetectionGenome]:
    """Return preset genomes tuned for different science cases."""
    rng = rng or np.random.default_rng()

    presets = []

    # Preset 1: Optimized for gravitational lens detection
    presets.append(
        _make_genome(
            {
                "source_threshold": 3.0,
                "lens_arc_min_length": 15,
                "lens_arc_max_width": 6,
                "lens_ring_min_r": 10,
                "lens_ring_max_r": 60,
                "lens_snr_threshold": 2.5,
                "weight_classical": 0.3,
                "weight_morphology": 0.1,
                "weight_anomaly": 0.5,
                "weight_distribution": 0.05,
                "weight_galaxy": 0.03,
                "weight_kinematic": 0.01,
                "weight_transient": 0.01,
            },
            rng,
        )
    )

    # Preset 2: Optimized for unusual galaxy morphology
    presets.append(
        _make_genome(
            {
                "source_threshold": 2.5,
                "morph_smoothing": 2.0,
                "morph_threshold_pctile": 75,
                "galaxy_tidal_threshold": 0.25,
                "galaxy_asymmetry_threshold": 0.2,
                "weight_classical": 0.1,
                "weight_morphology": 0.3,
                "weight_anomaly": 0.15,
                "weight_distribution": 0.05,
                "weight_galaxy": 0.35,
                "weight_kinematic": 0.03,
                "weight_transient": 0.02,
            },
            rng,
        )
    )

    # Preset 3: Optimized for stellar distribution anomalies
    presets.append(
        _make_genome(
            {
                "source_threshold": 2.0,
                "source_min_area": 3,
                "dist_overdensity_sigma": 3.0,
                "weight_classical": 0.05,
                "weight_morphology": 0.05,
                "weight_anomaly": 0.1,
                "weight_distribution": 0.5,
                "weight_galaxy": 0.05,
                "weight_kinematic": 0.2,
                "weight_transient": 0.05,
            },
            rng,
        )
    )

    # Preset 4: Balanced all-purpose
    presets.append(
        _make_genome(
            {
                "source_threshold": 3.0,
                "gabor_freq_min": 0.05,
                "gabor_freq_max": 0.4,
                "gabor_n_freqs": 4,
                "gabor_n_orient": 8,
                "anomaly_contamination": 0.05,
                "galaxy_tidal_threshold": 0.3,
                "galaxy_color_sigma": 2.5,
                "kinematic_pm_min": 5.0,
                "kinematic_cluster_eps": 2.0,
                "transient_noise_threshold": 3.0,
                "weight_classical": 0.15,
                "weight_morphology": 0.15,
                "weight_anomaly": 0.15,
                "weight_distribution": 0.15,
                "weight_galaxy": 0.15,
                "weight_kinematic": 0.15,
                "weight_transient": 0.1,
            },
            rng,
        )
    )

    # Preset 5: Sensitive / low threshold
    presets.append(
        _make_genome(
            {
                "source_threshold": 1.8,
                "lens_snr_threshold": 1.8,
                "anomaly_contamination": 0.1,
                "dist_overdensity_sigma": 2.5,
                "galaxy_tidal_threshold": 0.2,
                "transient_noise_threshold": 2.0,
                "weight_classical": 0.15,
                "weight_morphology": 0.15,
                "weight_anomaly": 0.2,
                "weight_distribution": 0.15,
                "weight_galaxy": 0.15,
                "weight_kinematic": 0.1,
                "weight_transient": 0.1,
            },
            rng,
        )
    )

    # Preset 6: Optimized for kinematic detection (proper motion)
    presets.append(
        _make_genome(
            {
                "source_threshold": 2.5,
                "kinematic_pm_min": 3.0,
                "kinematic_cluster_eps": 1.5,
                "kinematic_cluster_min": 4,
                "kinematic_stream_min_length": 6,
                "weight_classical": 0.05,
                "weight_morphology": 0.05,
                "weight_anomaly": 0.05,
                "weight_distribution": 0.15,
                "weight_galaxy": 0.05,
                "weight_kinematic": 0.55,
                "weight_transient": 0.1,
            },
            rng,
        )
    )

    # Preset 7: Optimized for transient / variability detection
    presets.append(
        _make_genome(
            {
                "source_threshold": 2.0,
                "transient_noise_threshold": 2.0,
                "transient_parallax_snr": 3.0,
                "weight_classical": 0.05,
                "weight_morphology": 0.05,
                "weight_anomaly": 0.1,
                "weight_distribution": 0.05,
                "weight_galaxy": 0.05,
                "weight_kinematic": 0.15,
                "weight_transient": 0.55,
            },
            rng,
        )
    )

    # Preset 8: Optimized for Sersic profile / galaxy structure
    presets.append(
        _make_genome(
            {
                "source_threshold": 2.5,
                "sersic_max_radius_frac": 0.85,
                "sersic_residual_sigma": 2.5,
                "galaxy_tidal_threshold": 0.25,
                "weight_classical": 0.05,
                "weight_morphology": 0.15,
                "weight_anomaly": 0.1,
                "weight_galaxy": 0.2,
                "weight_sersic": 0.35,
                "weight_wavelet": 0.1,
                "weight_population": 0.05,
            },
            rng,
        )
    )

    # Preset 9: Optimized for multi-scale / wavelet detection
    presets.append(
        _make_genome(
            {
                "source_threshold": 2.0,
                "wavelet_n_scales": 6,
                "wavelet_significance": 2.5,
                "weight_classical": 0.1,
                "weight_morphology": 0.1,
                "weight_anomaly": 0.1,
                "weight_distribution": 0.1,
                "weight_sersic": 0.1,
                "weight_wavelet": 0.4,
                "weight_population": 0.1,
            },
            rng,
        )
    )

    # Preset 10: Optimized for stellar population / CMD analysis
    presets.append(
        _make_genome(
            {
                "source_threshold": 2.0,
                "population_ms_width": 0.25,
                "population_bs_offset": 0.3,
                "kinematic_pm_min": 3.0,
                "kinematic_cluster_eps": 1.5,
                "weight_distribution": 0.15,
                "weight_kinematic": 0.2,
                "weight_transient": 0.1,
                "weight_population": 0.45,
                "weight_wavelet": 0.1,
            },
            rng,
        )
    )

    # Preset 11: Optimized for time-domain variability detection
    presets.append(
        _make_genome(
            {
                "source_threshold": 2.0,
                "variability_min_epochs": 8,
                "variability_significance": 2.0,
                "variability_period_min": 0.05,
                "variability_period_max": 800,
                "weight_variability": 0.45,
                "weight_transient": 0.15,
                "weight_kinematic": 0.1,
                "weight_population": 0.1,
                "weight_anomaly": 0.1,
                "weight_distribution": 0.1,
            },
            rng,
        )
    )

    # Preset 12: Optimized for temporal change detection (multi-epoch differencing)
    presets.append(
        _make_genome(
            {
                "source_threshold": 2.5,
                "temporal_snr_threshold": 4.0,
                "temporal_min_epochs": 3,
                "temporal_max_baseline": 1000,
                "temporal_min_baseline": 5,
                "temporal_dipole_max_sep": 5.0,
                "variability_min_epochs": 8,
                "variability_significance": 2.0,
                "weight_temporal": 0.40,
                "weight_variability": 0.20,
                "weight_transient": 0.15,
                "weight_anomaly": 0.10,
                "weight_distribution": 0.05,
                "weight_kinematic": 0.05,
                "weight_population": 0.05,
            },
            rng,
        )
    )

    return presets
