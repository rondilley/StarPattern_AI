"""Time-domain variability analysis for light curves.

Analyzes light curves (primarily from ZTF) for variability, periodicity,
and transient events. Uses variability indices, Lomb-Scargle periodograms,
and outburst detection to classify variable sources.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from star_pattern.core.catalog import StarCatalog
from star_pattern.core.config import DetectionConfig
from star_pattern.utils.logging import get_logger

logger = get_logger("detection.variability")


class VariabilityAnalyzer:
    """Analyze light curves for variability, periodicity, and transients."""

    def __init__(self, config: DetectionConfig | None = None):
        config = config or DetectionConfig()
        self.min_epochs = config.variability_min_epochs
        self.significance_threshold = config.variability_significance
        self.period_min_days = config.variability_period_min
        self.period_max_days = config.variability_period_max

    def analyze(self, catalog: StarCatalog) -> dict[str, Any]:
        """Run variability analysis on all sources with light curves.

        Extracts sources that have ztf_lightcurve in their properties,
        computes variability metrics, and classifies variable types.

        Returns:
            Dict with variability_score, variable_candidates,
            periodic_candidates, transient_candidates, and per-source details.
        """
        sources_with_lc = []
        for entry in catalog.entries:
            lc = entry.properties.get("ztf_lightcurve")
            if lc and isinstance(lc, dict):
                # Check if any band has enough epochs
                has_enough = any(
                    len(pts) >= self.min_epochs
                    for pts in lc.values()
                    if isinstance(pts, list)
                )
                if has_enough:
                    sources_with_lc.append(entry)

        if not sources_with_lc:
            return {
                "variability_score": 0.0,
                "n_analyzed": 0,
                "variable_candidates": [],
                "periodic_candidates": [],
                "transient_candidates": [],
            }

        variable_candidates = []
        periodic_candidates = []
        transient_candidates = []
        all_scores = []

        for entry in sources_with_lc:
            lc_data = entry.properties["ztf_lightcurve"]
            best_var_index = None
            best_periodogram = None
            best_outbursts = []
            best_band = None

            for band, points in lc_data.items():
                if not isinstance(points, list) or len(points) < self.min_epochs:
                    continue

                times = np.array([p[0] for p in points])
                mags = np.array([p[1] for p in points])
                errs = np.array([p[2] for p in points])

                # Compute variability indices
                var_index = self._compute_variability_index(times, mags, errs)

                # Run Lomb-Scargle periodogram
                periodogram = self._lomb_scargle(times, mags, errs)

                # Detect outbursts
                outbursts = self._detect_outbursts(times, mags, errs)

                # Keep the band with highest chi2_reduced
                if best_var_index is None or var_index.get("chi2_reduced", 0) > best_var_index.get("chi2_reduced", 0):
                    best_var_index = var_index
                    best_periodogram = periodogram
                    best_outbursts = outbursts
                    best_band = band

            if best_var_index is None:
                continue

            # Classify the variable type
            classification = self._classify_variable(
                best_var_index, best_periodogram, best_outbursts
            )

            # Compute a per-source score (0-1)
            source_score = self._source_variability_score(
                best_var_index, best_periodogram, best_outbursts
            )
            all_scores.append(source_score)

            source_result = {
                "source_id": entry.source_id,
                "ra": entry.ra,
                "dec": entry.dec,
                "band": best_band,
                "variability_index": best_var_index,
                "periodogram": {
                    "best_period": best_periodogram.get("best_period"),
                    "best_power": best_periodogram.get("best_power"),
                    "fap": best_periodogram.get("fap"),
                    "is_periodic": best_periodogram.get("is_periodic", False),
                },
                "n_outbursts": len(best_outbursts),
                "classification": classification,
                "score": source_score,
            }

            if best_var_index.get("is_variable", False):
                variable_candidates.append(source_result)

            if best_periodogram and best_periodogram.get("is_periodic", False):
                periodic_candidates.append(source_result)

            if classification == "transient" or best_outbursts:
                transient_candidates.append(source_result)

        # Overall variability score: combination of fraction of variables
        # and mean score of top variables
        if all_scores:
            n_variable = len(variable_candidates)
            frac_variable = n_variable / len(sources_with_lc)
            top_scores = sorted(all_scores, reverse=True)[:10]
            mean_top = float(np.mean(top_scores))
            variability_score = float(np.clip(
                0.4 * frac_variable + 0.6 * mean_top, 0, 1
            ))
        else:
            variability_score = 0.0

        logger.info(
            f"Variability analysis: {len(sources_with_lc)} sources analyzed, "
            f"{len(variable_candidates)} variable, "
            f"{len(periodic_candidates)} periodic, "
            f"{len(transient_candidates)} transient candidates"
        )

        return {
            "variability_score": variability_score,
            "n_analyzed": len(sources_with_lc),
            "variable_candidates": variable_candidates,
            "periodic_candidates": periodic_candidates,
            "transient_candidates": transient_candidates,
        }

    def _compute_variability_index(
        self,
        times: np.ndarray,
        mags: np.ndarray,
        errs: np.ndarray,
    ) -> dict[str, Any]:
        """Compute multiple variability indices for a single light curve.

        Returns dict with:
        - weighted_stdev: flux-error-weighted standard deviation
        - chi2_reduced: reduced chi-squared vs constant model
        - iqr: interquartile range of magnitudes
        - eta: von Neumann ratio (time-ordered variability)
        - amplitude: max - min magnitude
        - median_abs_dev: MAD of magnitudes
        - is_variable: bool (chi2_reduced > significance_threshold)
        """
        n = len(mags)
        if n < 2:
            return {
                "weighted_stdev": 0.0,
                "chi2_reduced": 0.0,
                "iqr": 0.0,
                "eta": 0.0,
                "amplitude": 0.0,
                "median_abs_dev": 0.0,
                "is_variable": False,
            }

        # Weighted mean magnitude
        weights = 1.0 / (errs ** 2 + 1e-10)
        wmean = float(np.average(mags, weights=weights))

        # Weighted standard deviation
        wvar = float(np.average((mags - wmean) ** 2, weights=weights))
        weighted_stdev = float(np.sqrt(wvar))

        # Reduced chi-squared vs constant model
        chi2 = float(np.sum(((mags - wmean) / (errs + 1e-10)) ** 2))
        chi2_reduced = chi2 / (n - 1) if n > 1 else 0.0

        # Interquartile range
        q75, q25 = np.percentile(mags, [75, 25])
        iqr = float(q75 - q25)

        # Von Neumann ratio (eta): measures time-correlated variability
        # eta = (1/(n-1)) * sum((m_{i+1} - m_i)^2) / var(m)
        # Low eta -> correlated variability, high eta -> uncorrelated noise
        var_m = float(np.var(mags, ddof=1)) if n > 1 else 1e-10
        if var_m > 1e-10:
            sorted_idx = np.argsort(times)
            sorted_mags = mags[sorted_idx]
            delta_sq = np.sum(np.diff(sorted_mags) ** 2)
            eta = float(delta_sq / ((n - 1) * var_m))
        else:
            eta = 2.0  # neutral value (Gaussian noise -> eta ~ 2)

        # Amplitude
        amplitude = float(np.max(mags) - np.min(mags))

        # Median absolute deviation
        median_mag = float(np.median(mags))
        mad = float(np.median(np.abs(mags - median_mag)))

        # Is variable: chi2_reduced significantly above 1.0
        is_variable = chi2_reduced > self.significance_threshold

        return {
            "weighted_stdev": weighted_stdev,
            "chi2_reduced": chi2_reduced,
            "iqr": iqr,
            "eta": eta,
            "amplitude": amplitude,
            "median_abs_dev": mad,
            "is_variable": is_variable,
        }

    def _lomb_scargle(
        self,
        times: np.ndarray,
        mags: np.ndarray,
        errs: np.ndarray,
    ) -> dict[str, Any]:
        """Lomb-Scargle periodogram for unevenly sampled data.

        Uses astropy.timeseries.LombScargle.

        Returns dict with:
        - best_period: float (days)
        - best_power: float
        - fap: false alarm probability
        - is_periodic: bool (fap < 0.01)
        - periods: list of trial periods (for plotting)
        - powers: list of periodogram powers (for plotting)
        """
        try:
            from astropy.timeseries import LombScargle
        except ImportError:
            logger.warning("astropy.timeseries.LombScargle not available")
            return {
                "best_period": None,
                "best_power": 0.0,
                "fap": 1.0,
                "is_periodic": False,
                "periods": [],
                "powers": [],
            }

        baseline = float(np.max(times) - np.min(times))
        if baseline < self.period_min_days * 2:
            return {
                "best_period": None,
                "best_power": 0.0,
                "fap": 1.0,
                "is_periodic": False,
                "periods": [],
                "powers": [],
            }

        ls = LombScargle(times, mags, errs)

        # Compute frequency grid
        min_freq = 1.0 / min(self.period_max_days, baseline / 2)
        max_freq = 1.0 / self.period_min_days
        # Cap number of frequency points to avoid excessive computation
        n_freq = min(10000, int((max_freq - min_freq) * baseline * 5))
        n_freq = max(100, n_freq)

        frequency = np.linspace(min_freq, max_freq, n_freq)

        try:
            power = ls.power(frequency)
        except Exception as e:
            logger.warning(f"Lomb-Scargle computation failed: {e}")
            return {
                "best_period": None,
                "best_power": 0.0,
                "fap": 1.0,
                "is_periodic": False,
                "periods": [],
                "powers": [],
            }

        periods = 1.0 / frequency
        best_idx = int(np.argmax(power))
        best_period = float(periods[best_idx])
        best_power = float(power[best_idx])

        # False alarm probability
        try:
            fap = float(ls.false_alarm_probability(best_power))
        except Exception:
            # Fallback: use analytical approximation
            fap = 1.0

        is_periodic = fap < 0.01

        return {
            "best_period": best_period,
            "best_power": best_power,
            "fap": fap,
            "is_periodic": is_periodic,
            "periods": periods.tolist(),
            "powers": power.tolist(),
        }

    def _detect_outbursts(
        self,
        times: np.ndarray,
        mags: np.ndarray,
        errs: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Detect significant brightening/fading events.

        Uses a rolling median baseline so long-term trends (eclipsing
        binaries, secular variability) do not bias outburst detection.
        Points deviating more than N sigma from the local baseline are
        flagged.

        Returns list of outburst dicts with: mjd, mag, deviation_sigma,
        type (brightening/fading).
        """
        n = len(mags)
        if n < 5:
            return []

        # Sort by time for rolling window
        sort_idx = np.argsort(times)
        sorted_times = times[sort_idx]
        sorted_mags = mags[sort_idx]

        # Rolling window size: ~30 points or half the data, whichever is smaller
        half_window = min(15, n // 4)
        half_window = max(half_window, 2)  # at least 2 neighbors each side

        # Compute rolling median and MAD for each point
        baseline = np.empty(n)
        local_sigma = np.empty(n)

        for i in range(n):
            lo = max(0, i - half_window)
            hi = min(n, i + half_window + 1)
            window = sorted_mags[lo:hi]
            med = np.median(window)
            baseline[i] = med
            mad = np.median(np.abs(window - med))
            local_sigma[i] = mad * 1.4826  # MAD to sigma

        # Fall back to global sigma where local estimate is degenerate
        global_mad = float(np.median(np.abs(sorted_mags - np.median(sorted_mags))))
        global_sigma = global_mad * 1.4826
        if global_sigma < 1e-6:
            return []
        local_sigma = np.where(local_sigma < 1e-6, global_sigma, local_sigma)

        outbursts = []
        threshold = self.significance_threshold

        for i in range(n):
            deviation = (baseline[i] - sorted_mags[i]) / local_sigma[i]
            if abs(deviation) > threshold:
                outburst_type = "brightening" if deviation > 0 else "fading"
                outbursts.append({
                    "mjd": float(sorted_times[i]),
                    "mag": float(sorted_mags[i]),
                    "deviation_sigma": float(deviation),
                    "type": outburst_type,
                })

        return outbursts

    def _classify_variable(
        self,
        var_index: dict[str, Any],
        periodogram: dict[str, Any] | None,
        outbursts: list[dict[str, Any]],
    ) -> str:
        """Heuristic variable star classification.

        Categories:
        - periodic_pulsator: periodic with period < 100 days
        - eclipsing_binary: periodic with short period and large amplitude
        - eruptive: has outbursts, not periodic
        - long_period_variable: periodic with period > 100 days
        - agn_like: aperiodic, high amplitude, low eta
        - transient: monotonic fading or single outburst
        - unclassified_variable: variable but doesn't fit other categories
        """
        is_periodic = periodogram.get("is_periodic", False) if periodogram else False
        best_period = periodogram.get("best_period") if periodogram else None
        amplitude = var_index.get("amplitude", 0)
        eta = var_index.get("eta", 2.0)
        is_variable = var_index.get("is_variable", False)

        if not is_variable:
            return "non_variable"

        if is_periodic and best_period is not None:
            if best_period < 1.0 and amplitude > 0.3:
                return "eclipsing_binary"
            elif best_period > 100:
                return "long_period_variable"
            else:
                return "periodic_pulsator"

        # Non-periodic variable
        if outbursts:
            n_brightening = sum(1 for o in outbursts if o["type"] == "brightening")
            n_fading = sum(1 for o in outbursts if o["type"] == "fading")

            # Predominantly fading -> transient
            if n_fading > 0 and n_brightening == 0:
                return "transient"

            # Single bright outburst -> eruptive
            if n_brightening <= 3:
                return "eruptive"

        # High amplitude, aperiodic, low eta -> AGN-like
        if amplitude > 0.5 and eta < 1.5:
            return "agn_like"

        return "unclassified_variable"

    def _source_variability_score(
        self,
        var_index: dict[str, Any],
        periodogram: dict[str, Any] | None,
        outbursts: list[dict[str, Any]],
    ) -> float:
        """Compute a single variability score (0-1) for a source.

        Combines chi2_reduced, periodogram power, and outburst count
        into a single score.
        """
        chi2 = var_index.get("chi2_reduced", 0.0)
        # Map chi2 to 0-1: chi2=1 -> 0, chi2=10 -> 0.5, chi2=100 -> 1.0
        chi2_score = float(np.clip(np.log10(max(chi2, 1.0)) / 2.0, 0, 1))

        # Periodogram power contribution
        best_power = periodogram.get("best_power", 0.0) if periodogram else 0.0
        fap = periodogram.get("fap", 1.0) if periodogram else 1.0
        period_score = float(np.clip(best_power, 0, 1)) * (1.0 if fap < 0.01 else 0.3)

        # Outburst contribution
        n_out = len(outbursts)
        outburst_score = float(np.clip(n_out / 5.0, 0, 1))

        return float(np.clip(
            0.5 * chi2_score + 0.3 * period_score + 0.2 * outburst_score,
            0, 1,
        ))
