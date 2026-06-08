"""Spectrum comparison metrics for bssunfold.

Each function follows single-responsibility principle and operates on
1-D numpy arrays. All metrics are implemented with numpy/scipy only.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

__all__ = [
    "compare_spectra",
    "compare_multiple",
    "kl_divergence",
    "cross_entropy",
    "entropy",
    "entropy_difference_percent",
    "wasserstein_dist",
    "energy_dist",
    "kolmogorov_smirnov_stat",
    "pearson_r",
    "spearman_r",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "mape",
    "r2_score",
    "max_error",
    "median_absolute_error",
    "cosine_similarity",
    "mmd_rbf",
    "chi_squared",
    "g_test",
    "freeman_tukey",
    "cressie_read",
    "anderson_darling",
    "standardized_mean_difference",
    "wilcoxon_test",
    "mannwhitneyu_test",
]

EPS = 1e-15


def _normalize(p: np.ndarray) -> np.ndarray:
    """Normalize array to a probability distribution."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, EPS, None)
    return p / np.sum(p)


def _check_same_length(s1: np.ndarray, s2: np.ndarray) -> None:
    if len(s1) != len(s2):
        raise ValueError(
            f"Spectra must have same length, got {len(s1)} and {len(s2)}"
        )


# ─── Entropy-based ────────────────────────────────────────────────

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback–Leibler divergence D_KL(p || q).

    Both inputs are normalized to probability distributions internally.
    """
    pn = _normalize(p)
    qn = _normalize(q)
    return float(np.sum(pn * np.log(pn / qn)))


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Cross-entropy H(p, q) = -sum(p * log(q))."""
    pn = _normalize(p)
    qn = _normalize(q)
    return float(-np.sum(pn * np.log(qn)))


def entropy(p: np.ndarray) -> float:
    """Shannon entropy H(p) = -sum(p * log(p))."""
    pn = _normalize(p)
    return float(-np.sum(pn * np.log(pn)))


def entropy_difference_percent(p: np.ndarray, q: np.ndarray) -> float:
    """Relative difference between cross-entropy and entropy in percent.

    100 * (H(p,q) - H(p)) / H(p)
    """
    pn = _normalize(p)
    qn = _normalize(q)
    h_p = -np.sum(pn * np.log(pn))
    h_pq = -np.sum(pn * np.log(qn))
    if h_p == 0:
        return 0.0
    return float(100.0 * (h_pq - h_p) / h_p)


# ─── Distribution distances ──────────────────────────────────────


def wasserstein_dist(p: np.ndarray, q: np.ndarray) -> float:
    """Wasserstein (earth mover's) distance between two distributions.

    Uses scipy.stats.wasserstein_distance.
    """
    from scipy.stats import wasserstein_distance as _wd
    return float(_wd(p, q))


def energy_dist(p: np.ndarray, q: np.ndarray) -> float:
    """Energy distance between two distributions.

    Uses scipy.stats.energy_distance.
    """
    from scipy.stats import energy_distance as _ed
    return float(_ed(p, q))


def kolmogorov_smirnov_stat(p: np.ndarray, q: np.ndarray) -> float:
    """Kolmogorov-Smirnov test statistic (D statistic)."""
    from scipy.stats import ks_2samp
    return float(ks_2samp(p, q)[0])


# ─── Correlation ─────────────────────────────────────────────────


def pearson_r(p: np.ndarray, q: np.ndarray) -> float:
    """Pearson correlation coefficient.

    Returns 0.0 if either input is constant (variance = 0).
    """
    from scipy.stats import pearsonr
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    if np.std(p_arr) == 0 or np.std(q_arr) == 0:
        return 0.0
    return float(pearsonr(p_arr, q_arr)[0])


def spearman_r(p: np.ndarray, q: np.ndarray) -> float:
    """Spearman rank correlation coefficient.

    Returns 0.0 if either input is constant (variance = 0).
    """
    from scipy.stats import spearmanr
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    if np.std(p_arr) == 0 or np.std(q_arr) == 0:
        return 0.0
    return float(spearmanr(p_arr, q_arr)[0])


# ─── Error metrics ────────────────────────────────────────────────


def mean_squared_error(p: np.ndarray, q: np.ndarray) -> float:
    """Mean squared error."""
    _check_same_length(p, q)
    return float(np.mean((np.asarray(p) - np.asarray(q)) ** 2))


def root_mean_squared_error(p: np.ndarray, q: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(p, q)))


def mean_absolute_error(p: np.ndarray, q: np.ndarray) -> float:
    """Mean absolute error."""
    _check_same_length(p, q)
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(q))))


def mape(p: np.ndarray, q: np.ndarray) -> float:
    """Mean absolute percentage error.

    Returns percentage (0–100). Skips elements where p is near zero.
    """
    _check_same_length(p, q)
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    mask = np.abs(p_arr) > EPS
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((p_arr[mask] - q_arr[mask]) / p_arr[mask])) * 100.0)


def r2_score(p: np.ndarray, q: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    _check_same_length(p, q)
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    ss_res = np.sum((p_arr - q_arr) ** 2)
    ss_tot = np.sum((p_arr - np.mean(p_arr)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def max_error(p: np.ndarray, q: np.ndarray) -> float:
    """Maximum residual error."""
    _check_same_length(p, q)
    return float(np.max(np.abs(np.asarray(p) - np.asarray(q))))


def median_absolute_error(p: np.ndarray, q: np.ndarray) -> float:
    """Median absolute error."""
    _check_same_length(p, q)
    return float(np.median(np.abs(np.asarray(p) - np.asarray(q))))


# ─── Kernel / similarity ──────────────────────────────────────────


def cosine_similarity(p: np.ndarray, q: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    Returns 0 if either vector is zero-norm.
    """
    _check_same_length(p, q)
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    norm_p = np.linalg.norm(p_arr)
    norm_q = np.linalg.norm(q_arr)
    if norm_p == 0 or norm_q == 0:
        return 0.0
    return float(np.dot(p_arr, q_arr) / (norm_p * norm_q))


def mmd_rbf(p: np.ndarray, q: np.ndarray, gamma: Optional[float] = None) -> float:
    """Maximum Mean Discrepancy with RBF kernel.

    Parameters
    ----------
    p, q : np.ndarray
        1-D arrays.
    gamma : float, optional
        RBF kernel width. If None, uses 1 / (2 * median_distance**2).
    """
    _check_same_length(p, q)
    from scipy.spatial.distance import cdist
    X = np.asarray(p, dtype=float).reshape(-1, 1)
    Y = np.asarray(q, dtype=float).reshape(-1, 1)
    if gamma is None:
        all_pts = np.vstack([X, Y])
        med = np.median(cdist(all_pts, all_pts))
        gamma = 1.0 / (2.0 * max(med, EPS) ** 2)
    XX = np.exp(-gamma * cdist(X, X, metric="sqeuclidean"))
    YY = np.exp(-gamma * cdist(Y, Y, metric="sqeuclidean"))
    XY = np.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))
    return float(np.mean(XX) + np.mean(YY) - 2.0 * np.mean(XY))


# ─── Chi-squared family (power divergence) ────────────────────────


def chi_squared(p: np.ndarray, q: np.ndarray) -> float:
    """Pearson's chi-squared test statistic.

    Internally normalizes both inputs as probability distributions.
    """
    from scipy.stats import power_divergence
    pn = _normalize(p)
    qn = _normalize(q)
    return float(power_divergence(pn, qn, lambda_="pearson")[0])


def g_test(p: np.ndarray, q: np.ndarray) -> float:
    """G-test (log-likelihood ratio) statistic."""
    from scipy.stats import power_divergence
    pn = _normalize(p)
    qn = _normalize(q)
    return float(power_divergence(pn, qn, lambda_="log-likelihood")[0])


def freeman_tukey(p: np.ndarray, q: np.ndarray) -> float:
    """Freeman-Tukey statistic."""
    from scipy.stats import power_divergence
    pn = _normalize(p)
    qn = _normalize(q)
    return float(power_divergence(pn, qn, lambda_="freeman-tukey")[0])


def cressie_read(p: np.ndarray, q: np.ndarray) -> float:
    """Cressie-Read statistic."""
    from scipy.stats import power_divergence
    pn = _normalize(p)
    qn = _normalize(q)
    return float(power_divergence(pn, qn, lambda_="cressie-read")[0])


# ─── Statistical tests ────────────────────────────────────────────


def anderson_darling(p: np.ndarray, q: np.ndarray) -> float:
    """Anderson-Darling test statistic for k-samples.

    Returns 0.0 if either input is constant (all identical values).
    """
    from scipy.stats import anderson_ksamp, PermutationMethod
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    if len(np.unique(p_arr)) < 2 or len(np.unique(q_arr)) < 2:
        return 0.0
    return float(anderson_ksamp(
        [p_arr, q_arr],
        method=PermutationMethod(n_resamples=999),
        variant="right",
    )[0])


def wilcoxon_test(p: np.ndarray, q: np.ndarray) -> float:
    """Wilcoxon signed-rank test statistic.

    Returns 0.0 if both inputs are identical (all differences zero).
    """
    from scipy.stats import wilcoxon
    _check_same_length(p, q)
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    if np.allclose(p_arr, q_arr):
        return 0.0
    return float(wilcoxon(p_arr, q_arr, alternative="two-sided")[0])


def mannwhitneyu_test(p: np.ndarray, q: np.ndarray) -> float:
    """Mann-Whitney U test statistic."""
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(p, q, alternative="two-sided")[0])


def standardized_mean_difference(p: np.ndarray, q: np.ndarray) -> float:
    """Standardized mean difference (Cohen's d).

    SMD = (mean(p) - mean(q)) / sqrt((var(p) + var(q)) / 2)
    """
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    diff = np.mean(p_arr) - np.mean(q_arr)
    denom = np.sqrt((np.var(p_arr, ddof=1) + np.var(q_arr, ddof=1)) / 2.0)
    if denom == 0:
        return 0.0
    return float(diff / denom)


# ─── High-level comparison functions ──────────────────────────────

_ALL_METRICS: Dict[str, str] = {
    "kl_divergence": "KL divergence",
    "cross_entropy": "Cross entropy",
    "entropy_difference_percent": "Entropy difference (%)",
    "wasserstein_dist": "Wasserstein distance",
    "energy_dist": "Energy distance",
    "kolmogorov_smirnov_stat": "Kolmogorov-Smirnov statistic",
    "pearson_r": "Pearson correlation",
    "spearman_r": "Spearman correlation",
    "mean_squared_error": "Mean squared error",
    "root_mean_squared_error": "Root mean squared error",
    "mean_absolute_error": "Mean absolute error",
    "mape": "MAPE (%)",
    "r2_score": "R² score",
    "max_error": "Max error",
    "median_absolute_error": "Median absolute error",
    "cosine_similarity": "Cosine similarity",
    "mmd_rbf": "MMD (RBF)",
    "chi_squared": "Chi-squared (Pearson)",
    "g_test": "G-test (log-likelihood)",
    "freeman_tukey": "Freeman-Tukey",
    "cressie_read": "Cressie-Read",
    "anderson_darling": "Anderson-Darling",
    "standardized_mean_difference": "Standardized mean difference",
    "wilcoxon_test": "Wilcoxon test",
    "mannwhitneyu_test": "Mann-Whitney U test",
}

_METRIC_FUNCTIONS: Dict[str, callable] = {
    "kl_divergence": kl_divergence,
    "cross_entropy": cross_entropy,
    "entropy_difference_percent": entropy_difference_percent,
    "wasserstein_dist": wasserstein_dist,
    "energy_dist": energy_dist,
    "kolmogorov_smirnov_stat": kolmogorov_smirnov_stat,
    "pearson_r": pearson_r,
    "spearman_r": spearman_r,
    "mean_squared_error": mean_squared_error,
    "root_mean_squared_error": root_mean_squared_error,
    "mean_absolute_error": mean_absolute_error,
    "mape": mape,
    "r2_score": r2_score,
    "max_error": max_error,
    "median_absolute_error": median_absolute_error,
    "cosine_similarity": cosine_similarity,
    "mmd_rbf": mmd_rbf,
    "chi_squared": chi_squared,
    "g_test": g_test,
    "freeman_tukey": freeman_tukey,
    "cressie_read": cressie_read,
    "anderson_darling": anderson_darling,
    "standardized_mean_difference": standardized_mean_difference,
    "wilcoxon_test": wilcoxon_test,
    "mannwhitneyu_test": mannwhitneyu_test,
}


def compare_spectra(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    metrics: Optional[Union[str, List[str]]] = None,
    bins: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compare two spectra using selected metrics.

    Parameters
    ----------
    spectrum1, spectrum2 : np.ndarray
        1-D arrays of the same length.
    metrics : str, list of str, or None
        Metric name(s). If None, all available metrics are computed.
    bins : np.ndarray, optional
        Energy bins (unused, reserved for future use).

    Returns
    -------
    Dict[str, float]
        Mapping from metric name (short key) to computed value.
    """
    _check_same_length(spectrum1, spectrum2)

    if metrics is None:
        keys = list(_METRIC_FUNCTIONS.keys())
    elif isinstance(metrics, str):
        keys = [metrics]
    else:
        keys = metrics

    results: Dict[str, float] = {}
    for key in keys:
        if key not in _METRIC_FUNCTIONS:
            raise ValueError(
                f"Unknown metric '{key}'. Available: {list(_METRIC_FUNCTIONS.keys())}"
            )
        try:
            results[key] = _METRIC_FUNCTIONS[key](spectrum1, spectrum2)
        except Exception as e:
            results[key] = float("nan")

    return results


def compare_multiple(
    spectra: List[np.ndarray],
    metrics: Optional[Union[str, List[str]]] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare multiple spectra pairwise against the first one.

    Parameters
    ----------
    spectra : list of np.ndarray
        List of spectra. First entry is treated as reference.
    metrics : str, list of str, or None
        Metric name(s). If None, all metrics are computed.
    labels : list of str, optional
        Labels for each spectrum.

    Returns
    -------
    Dict[str, Dict[str, float]]
        {label: {metric: value}} for each non-reference spectrum.
    """
    n = len(spectra)
    if n < 2:
        raise ValueError("At least two spectra required for comparison")
    if labels is None:
        labels = [f"Spectrum {i}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("Number of labels must match number of spectra")

    ref = spectra[0]
    results: Dict[str, Dict[str, float]] = {}
    for i in range(1, n):
        key = f"{labels[0]} vs {labels[i]}"
        results[key] = compare_spectra(ref, spectra[i], metrics=metrics)
    return results
