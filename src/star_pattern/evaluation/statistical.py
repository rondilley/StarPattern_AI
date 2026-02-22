"""Statistical significance testing and validation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from star_pattern.utils.logging import get_logger

logger = get_logger("evaluation.statistical")


def bootstrap_confidence(
    data: np.ndarray,
    statistic_fn: callable = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations.
        statistic_fn: Function to compute the statistic.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        rng: Random number generator.

    Returns:
        Dict with estimate, lower, upper bounds.
    """
    rng = rng or np.random.default_rng()
    n = len(data)
    if n == 0:
        return {"estimate": 0.0, "lower": 0.0, "upper": 0.0}

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(float(statistic_fn(sample)))

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = (1 - confidence) / 2

    return {
        "estimate": float(statistic_fn(data)),
        "lower": float(np.percentile(bootstrap_stats, alpha * 100)),
        "upper": float(np.percentile(bootstrap_stats, (1 - alpha) * 100)),
        "std": float(np.std(bootstrap_stats)),
    }


def ks_test_uniformity(values: np.ndarray) -> dict[str, float]:
    """Kolmogorov-Smirnov test for uniformity."""
    if len(values) < 5:
        return {"statistic": 0.0, "p_value": 1.0}

    # Normalize to [0, 1]
    v_min, v_max = values.min(), values.max()
    if v_max - v_min < 1e-10:
        return {"statistic": 0.0, "p_value": 1.0}

    normalized = (values - v_min) / (v_max - v_min)
    stat, p = stats.kstest(normalized, "uniform")
    return {"statistic": float(stat), "p_value": float(p)}


def anderson_darling_normality(values: np.ndarray) -> dict[str, Any]:
    """Anderson-Darling test for normality."""
    if len(values) < 8:
        return {"statistic": 0.0, "is_normal": True, "significance_level": None}

    result = stats.anderson(values, dist="norm")
    # Use 5% significance level
    sig_idx = 2  # 5% is at index 2
    is_normal = result.statistic < result.critical_values[sig_idx]

    return {
        "statistic": float(result.statistic),
        "is_normal": is_normal,
        "significance_level": float(result.significance_level[sig_idx]),
        "critical_value": float(result.critical_values[sig_idx]),
    }


def multiple_comparison_correction(
    p_values: list[float], method: str = "bonferroni"
) -> list[float]:
    """Apply multiple comparison correction to p-values.

    Args:
        p_values: List of uncorrected p-values.
        method: 'bonferroni' or 'fdr' (Benjamini-Hochberg).
    """
    n = len(p_values)
    if n == 0:
        return []

    if method == "bonferroni":
        return [min(p * n, 1.0) for p in p_values]

    elif method == "fdr":
        # Benjamini-Hochberg
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        corrected = [0.0] * n
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            corrected[orig_idx] = min(p * n / rank, 1.0)
        # Enforce monotonicity
        for i in range(n - 2, -1, -1):
            corrected[i] = min(corrected[i], corrected[i + 1])
        return corrected

    raise ValueError(f"Unknown method: {method}")


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic_fn: callable = lambda a, b: np.mean(a) - np.mean(b),
    n_permutations: int = 10000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Permutation test comparing two groups.

    Returns:
        Dict with observed statistic and p-value.
    """
    rng = rng or np.random.default_rng()
    observed = float(statistic_fn(group1, group2))

    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_stat = statistic_fn(combined[:n1], combined[n1:])
        if abs(perm_stat) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return {
        "observed_statistic": observed,
        "p_value": float(p_value),
        "n_permutations": n_permutations,
    }
