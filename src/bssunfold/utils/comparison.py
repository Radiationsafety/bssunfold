"""Spectrum comparison metrics for bssunfold.

Each function follows single-responsibility principle and operates on
1-D numpy arrays. All metrics are implemented with numpy/scipy only.
"""

import numpy as np
from typing import Dict, List, Optional, Union

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
    "total_flux",
    "total_flux_ratio",
    "mmd_rbf",
    "chi_squared",
    "g_test",
    "freeman_tukey",
    "cressie_read",
    "anderson_darling",
    "standardized_mean_difference",
    "wilcoxon_test",
    "mannwhitneyu_test",
    "fluence_difference_percent",
    "energy_group_fluence_diff",
    "dose_difference_percent",
    "fluence_averaged_energy_diff",
    "dose_averaged_energy_diff",
    "spectral_shape_similarity",
    "log_lethargy_correlation",
    "peak_location_error",
    "peak_width_error",
    "dose_weighted_error",
    "response_matrix_consistency",
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


# ─── Flux utility ─────────────────────────────────────────────────


def total_flux(s: np.ndarray) -> float:
    """Total flux (integral under spectrum curve).

    For discrete spectra equally spaced in lethargy, this is simply
    the sum of the bin values.
    """
    return float(np.sum(np.asarray(s, dtype=float)))


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


# ─── Flux comparison ──────────────────────────────────────────────


def total_flux_ratio(p: np.ndarray, q: np.ndarray) -> float:
    """Ratio of total fluxes: sum(unfolded) / sum(reference).

    1.0 = perfect conservation of total flux.
    > 1.0 = unfolded overestimates total flux.
    < 1.0 = unfolded underestimates total flux.
    Returns 0.0 if reference flux is zero.
    """
    _check_same_length(p, q)
    p_sum = np.sum(np.asarray(p, dtype=float))
    q_sum = np.sum(np.asarray(q, dtype=float))
    if p_sum == 0:
        return 0.0
    return float(q_sum / p_sum)


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


# ─── EURADOS-style integral quantity metrics ──────────────────────


def fluence_difference_percent(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    energy_bins: Optional[np.ndarray] = None,
) -> float:
    """Relative difference in total fluence between two spectra (%).

    As used in EURADOS comparison: Δ(%) = 100 * (Q_participant - Q_reference) / Q_reference

    Parameters
    ----------
    spectrum1, spectrum2 : np.ndarray
        Spectra to compare (fluence per energy bin).
    energy_bins : np.ndarray, optional
        Energy bin widths. If None, assumes uniform weighting.
    """
    _check_same_length(spectrum1, spectrum2)
    s1 = np.asarray(spectrum1, dtype=float)
    s2 = np.asarray(spectrum2, dtype=float)
    if energy_bins is not None:
        bins = np.asarray(energy_bins, dtype=float)
        total1 = np.sum(s1 * bins)
        total2 = np.sum(s2 * bins)
    else:
        total1 = np.sum(s1)
        total2 = np.sum(s2)
    if abs(total1) < EPS:
        return 0.0
    return float(100.0 * (total2 - total1) / total1)


def energy_group_fluence_diff(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    energy: np.ndarray,
    thermal_max: float = 0.4e-6,
    epithermal_max: float = 0.1,
) -> Dict[str, float]:
    """Relative difference in fluence for three energy groups (%).

    Groups: thermal (E < 0.4 eV), epithermal (0.4 eV <= E < 0.1 MeV),
    fast (E >= 0.1 MeV).

    Parameters
    ----------
    spectrum1, spectrum2 : np.ndarray
        Spectra to compare.
    energy : np.ndarray
        Energy grid in MeV.
    thermal_max : float
        Upper bound of thermal group in MeV (default: 0.4e-6 = 0.4 eV).
    epithermal_max : float
        Upper bound of epithermal group in MeV (default: 0.1).
    """
    _check_same_length(spectrum1, spectrum2)
    if len(energy) != len(spectrum1):
        raise ValueError("Energy array must match spectrum length")
    s1 = np.asarray(spectrum1, dtype=float)
    s2 = np.asarray(spectrum2, dtype=float)
    e = np.asarray(energy, dtype=float)

    thermal_mask = e < thermal_max
    epithermal_mask = (e >= thermal_max) & (e < epithermal_max)
    fast_mask = e >= epithermal_max

    result: Dict[str, float] = {}
    for name, mask in [("thermal", thermal_mask), ("epithermal", epithermal_mask), ("fast", fast_mask)]:
        if not np.any(mask):
            result[name] = 0.0
            continue
        t1 = np.sum(s1[mask])
        t2 = np.sum(s2[mask])
        if abs(t1) < EPS:
            result[name] = 0.0
        else:
            result[name] = float(100.0 * (t2 - t1) / t1)
    return result


def dose_difference_percent(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    energy: np.ndarray,
    cc_icrp116: Optional[np.ndarray] = None,
) -> float:
    """Relative difference in ambient dose equivalent H*(10) (%).

    Parameters
    ----------
    spectrum1, spectrum2 : np.ndarray
        Spectra to compare.
    energy : np.ndarray
        Energy grid in MeV.
    cc_icrp116 : np.ndarray, optional
        ICRP-116 conversion coefficients. If None, uses a simple approximation.
    """
    _check_same_length(spectrum1, spectrum2)
    if len(energy) != len(spectrum1):
        raise ValueError("Energy array must match spectrum length")
    s1 = np.asarray(spectrum1, dtype=float)
    s2 = np.asarray(spectrum2, dtype=float)
    e = np.asarray(energy, dtype=float)

    if cc_icrp116 is None:
        cc_icrp116 = np.ones_like(e)
    cc = np.asarray(cc_icrp116, dtype=float)

    log_steps = np.zeros_like(e)
    log_e = np.log10(e + 1e-15)
    log_steps[0] = log_e[1] - log_e[0] if len(e) > 1 else 1.0
    log_steps[-1] = log_e[-1] - log_e[-2] if len(e) > 1 else 1.0
    log_steps[1:-1] = (log_e[2:] - log_e[:-2]) / 2.0
    ln_steps = log_steps * np.log(10)

    dose1 = np.sum(s1 * cc * ln_steps)
    dose2 = np.sum(s2 * cc * ln_steps)

    if abs(dose1) < EPS:
        return 0.0
    return float(100.0 * (dose2 - dose1) / dose1)


def fluence_averaged_energy_diff(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    energy: np.ndarray,
) -> float:
    """Relative difference in fluence-averaged energy (%).

    <E> = sum(E_i * Phi_i) / sum(Phi_i)
    """
    _check_same_length(spectrum1, spectrum2)
    if len(energy) != len(spectrum1):
        raise ValueError("Energy array must match spectrum length")
    s1 = np.asarray(spectrum1, dtype=float)
    s2 = np.asarray(spectrum2, dtype=float)
    e = np.asarray(energy, dtype=float)

    e1_avg = np.sum(e * s1) / np.sum(s1) if np.sum(s1) > 0 else 0.0
    e2_avg = np.sum(e * s2) / np.sum(s2) if np.sum(s2) > 0 else 0.0

    if abs(e1_avg) < EPS:
        return 0.0
    return float(100.0 * (e2_avg - e1_avg) / e1_avg)


def dose_averaged_energy_diff(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    energy: np.ndarray,
    cc_icrp116: Optional[np.ndarray] = None,
) -> float:
    """Relative difference in H*(10)-averaged energy (%).

    <E>_H = sum(E_i * H_i * Phi_i) / sum(H_i * Phi_i)
    """
    _check_same_length(spectrum1, spectrum2)
    if len(energy) != len(spectrum1):
        raise ValueError("Energy array must match spectrum length")
    s1 = np.asarray(spectrum1, dtype=float)
    s2 = np.asarray(spectrum2, dtype=float)
    e = np.asarray(energy, dtype=float)

    if cc_icrp116 is None:
        cc_icrp116 = np.ones_like(e)
    cc = np.asarray(cc_icrp116, dtype=float)

    log_steps = np.zeros_like(e)
    log_e = np.log10(e + 1e-15)
    log_steps[0] = log_e[1] - log_e[0] if len(e) > 1 else 1.0
    log_steps[-1] = log_e[-1] - log_e[-2] if len(e) > 1 else 1.0
    log_steps[1:-1] = (log_e[2:] - log_e[:-2]) / 2.0
    ln_steps = log_steps * np.log(10)

    h1 = np.sum(e * s1 * cc * ln_steps) / np.sum(s1 * cc * ln_steps) if np.sum(s1 * cc * ln_steps) > 0 else 0.0
    h2 = np.sum(e * s2 * cc * ln_steps) / np.sum(s2 * cc * ln_steps) if np.sum(s2 * cc * ln_steps) > 0 else 0.0

    if abs(h1) < EPS:
        return 0.0
    return float(100.0 * (h2 - h1) / h1)


# ─── Spectral shape metrics ──────────────────────────────────────


def spectral_shape_similarity(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
) -> float:
    """Similarity of normalized spectral shapes (0-1).

    Computes cosine similarity of unit-normalized spectra.
    """
    _check_same_length(spectrum1, spectrum2)
    s1 = np.asarray(spectrum1, dtype=float)
    s2 = np.asarray(spectrum2, dtype=float)
    sum1 = np.sum(s1)
    sum2 = np.sum(s2)
    if sum1 < EPS or sum2 < EPS:
        return 0.0
    n1 = s1 / sum1
    n2 = s2 / sum2
    norm1 = np.linalg.norm(n1)
    norm2 = np.linalg.norm(n2)
    if norm1 < EPS or norm2 < EPS:
        return 0.0
    return float(np.dot(n1, n2) / (norm1 * norm2))


def log_lethargy_correlation(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    energy: np.ndarray,
) -> float:
    """Pearson correlation in log(E)*Phi(E) coordinates.

    As used in EURADOS figures showing spectra in lethargy representation.
    """
    _check_same_length(spectrum1, spectrum2)
    if len(energy) != len(spectrum1):
        raise ValueError("Energy array must match spectrum length")
    s1 = np.asarray(spectrum1, dtype=float)
    s2 = np.asarray(spectrum2, dtype=float)
    e = np.asarray(energy, dtype=float)

    log_e = np.log10(e + 1e-15)
    leth1 = log_e * s1
    leth2 = log_e * s2

    if np.std(leth1) == 0 or np.std(leth2) == 0:
        return 0.0
    from scipy.stats import pearsonr
    return float(pearsonr(leth1, leth2)[0])


def peak_location_error(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    energy: np.ndarray,
) -> float:
    """Relative error in peak location (%).

    Finds the energy of maximum flux in each spectrum and computes
    relative difference.
    """
    _check_same_length(spectrum1, spectrum2)
    if len(energy) != len(spectrum1):
        raise ValueError("Energy array must match spectrum length")
    s1 = np.asarray(spectrum1, dtype=float)
    s2 = np.asarray(spectrum2, dtype=float)
    e = np.asarray(energy, dtype=float)

    idx1 = np.argmax(s1)
    idx2 = np.argmax(s2)
    e_peak1 = e[idx1]
    e_peak2 = e[idx2]

    if abs(e_peak1) < EPS:
        return 0.0
    return float(100.0 * (e_peak2 - e_peak1) / e_peak1)


def peak_width_error(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    energy: np.ndarray,
) -> float:
    """Relative error in peak width at half maximum (%).

    Computes FWHM for each spectrum and returns relative difference.
    """
    _check_same_length(spectrum1, spectrum2)
    if len(energy) != len(spectrum1):
        raise ValueError("Energy array must match spectrum length")
    s1 = np.asarray(spectrum1, dtype=float)
    s2 = np.asarray(spectrum2, dtype=float)
    e = np.asarray(energy, dtype=float)

    def _fwhm(spec, e_arr):
        max_val = np.max(spec)
        if max_val < EPS:
            return 0.0
        half_max = max_val / 2.0
        above = spec >= half_max
        if not np.any(above):
            return 0.0
        indices = np.where(above)[0]
        return float(e_arr[indices[-1]] - e_arr[indices[0]])

    fwhm1 = _fwhm(s1, e)
    fwhm2 = _fwhm(s2, e)

    if abs(fwhm1) < EPS:
        return 0.0
    return float(100.0 * (fwhm2 - fwhm1) / fwhm1)


def dose_weighted_error(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    energy: np.ndarray,
    cc_icrp116: Optional[np.ndarray] = None,
) -> float:
    """Dose-weighted mean squared error.

    MSE weighted by dose contribution: sum(H_i * (s1_i - s2_i)^2) / sum(H_i)
    """
    _check_same_length(spectrum1, spectrum2)
    if len(energy) != len(spectrum1):
        raise ValueError("Energy array must match spectrum length")
    s1 = np.asarray(spectrum1, dtype=float)
    s2 = np.asarray(spectrum2, dtype=float)
    e = np.asarray(energy, dtype=float)

    if cc_icrp116 is None:
        cc_icrp116 = np.ones_like(e)
    cc = np.asarray(cc_icrp116, dtype=float)

    log_steps = np.zeros_like(e)
    log_e = np.log10(e + 1e-15)
    log_steps[0] = log_e[1] - log_e[0] if len(e) > 1 else 1.0
    log_steps[-1] = log_e[-1] - log_e[-2] if len(e) > 1 else 1.0
    log_steps[1:-1] = (log_e[2:] - log_e[:-2]) / 2.0
    ln_steps = log_steps * np.log(10)

    weights = cc * ln_steps
    total_weight = np.sum(weights)
    if total_weight < EPS:
        return 0.0
    weighted_mse = np.sum(weights * (s1 - s2) ** 2) / total_weight
    return float(np.sqrt(weighted_mse))


def response_matrix_consistency(
    spectrum: np.ndarray,
    readings: np.ndarray,
    response_matrix: np.ndarray,
) -> float:
    """Consistency between unfolded spectrum and measured readings.

    Computes chi-squared: sum((R_measured - R_computed)^2 / R_measured)
    where R_computed = response_matrix @ spectrum.
    """
    _check_same_length(readings, np.zeros(response_matrix.shape[0]))
    s = np.asarray(spectrum, dtype=float)
    r = np.asarray(readings, dtype=float)
    A = np.asarray(response_matrix, dtype=float)

    r_computed = A @ s
    mask = r > EPS
    if not np.any(mask):
        return 0.0
    chi2 = np.sum((r[mask] - r_computed[mask]) ** 2 / r[mask])
    return float(chi2)


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
    "total_flux_ratio": "Total flux ratio",
    "mmd_rbf": "MMD (RBF)",
    "chi_squared": "Chi-squared (Pearson)",
    "g_test": "G-test (log-likelihood)",
    "freeman_tukey": "Freeman-Tukey",
    "cressie_read": "Cressie-Read",
    "anderson_darling": "Anderson-Darling",
    "standardized_mean_difference": "Standardized mean difference",
    "wilcoxon_test": "Wilcoxon test",
    "mannwhitneyu_test": "Mann-Whitney U test",
    "fluence_difference_percent": "Fluence difference (%)",
    "dose_difference_percent": "Dose difference (%)",
    "fluence_averaged_energy_diff": "Fluence-averaged energy diff (%)",
    "dose_averaged_energy_diff": "Dose-averaged energy diff (%)",
    "spectral_shape_similarity": "Spectral shape similarity",
    "log_lethargy_correlation": "Log lethargy correlation",
    "peak_location_error": "Peak location error (%)",
    "peak_width_error": "Peak width error (%)",
    "dose_weighted_error": "Dose-weighted error",
    "response_matrix_consistency": "Response matrix consistency (χ²)",
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
    "total_flux_ratio": total_flux_ratio,
    "mmd_rbf": mmd_rbf,
    "chi_squared": chi_squared,
    "g_test": g_test,
    "freeman_tukey": freeman_tukey,
    "cressie_read": cressie_read,
    "anderson_darling": anderson_darling,
    "standardized_mean_difference": standardized_mean_difference,
    "wilcoxon_test": wilcoxon_test,
    "mannwhitneyu_test": mannwhitneyu_test,
    "spectral_shape_similarity": spectral_shape_similarity,
}

# Metrics requiring additional parameters (energy, response matrix, etc.)
_METRIC_FUNCTIONS_WITH_PARAMS: Dict[str, callable] = {
    "fluence_difference_percent": fluence_difference_percent,
    "energy_group_fluence_diff": energy_group_fluence_diff,
    "dose_difference_percent": dose_difference_percent,
    "fluence_averaged_energy_diff": fluence_averaged_energy_diff,
    "dose_averaged_energy_diff": dose_averaged_energy_diff,
    "log_lethargy_correlation": log_lethargy_correlation,
    "peak_location_error": peak_location_error,
    "peak_width_error": peak_width_error,
    "dose_weighted_error": dose_weighted_error,
    "response_matrix_consistency": response_matrix_consistency,
}


def compare_spectra(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    metrics: Optional[Union[str, List[str]]] = None,
    bins: Optional[np.ndarray] = None,
    energy: Optional[np.ndarray] = None,
    cc_icrp116: Optional[np.ndarray] = None,
    readings1: Optional[np.ndarray] = None,
    readings2: Optional[np.ndarray] = None,
    response_matrix: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compare two spectra using selected metrics.

    Parameters
    ----------
    spectrum1, spectrum2 : np.ndarray
        1-D arrays of the same length.
    metrics : str, list of str, or None
        Metric name(s). If None, all available metrics are computed
        (simple metrics only; pass ``energy`` to include EURADOS metrics).
    bins : np.ndarray, optional
        Energy bins (unused, reserved for future use).
    energy : np.ndarray, optional
        Energy grid in MeV. When provided, EURADOS-style metrics
        (dose differences, peak errors, etc.) are included automatically.
    cc_icrp116 : np.ndarray, optional
        ICRP-116 conversion coefficients for dose calculations.
    readings1, readings2 : np.ndarray, optional
        Measured readings for response-matrix consistency check.
    response_matrix : np.ndarray, optional
        Response matrix for consistency check.

    Returns
    -------
    Dict[str, float]
        Mapping from metric name (short key) to computed value.
    """
    _check_same_length(spectrum1, spectrum2)

    all_simple = list(_METRIC_FUNCTIONS.keys())
    all_eurados = list(_METRIC_FUNCTIONS_WITH_PARAMS.keys())

    if metrics is None:
        simple_keys = list(all_simple)
        eurados_keys = list(all_eurados) if energy is not None else []
    elif isinstance(metrics, str):
        if metrics in _METRIC_FUNCTIONS:
            simple_keys = [metrics]
            eurados_keys = []
        elif metrics in _METRIC_FUNCTIONS_WITH_PARAMS:
            simple_keys = []
            eurados_keys = [metrics]
        else:
            avail = all_simple + all_eurados
            raise ValueError(
                f"Unknown metric '{metrics}'. Available: {avail}"
            )
    else:
        simple_keys = [k for k in metrics if k in _METRIC_FUNCTIONS]
        eurados_keys = [k for k in metrics if k in _METRIC_FUNCTIONS_WITH_PARAMS]
        unknown = [k for k in metrics if k not in _METRIC_FUNCTIONS and k not in _METRIC_FUNCTIONS_WITH_PARAMS]
        if unknown:
            avail = all_simple + all_eurados
            raise ValueError(
                f"Unknown metric(s) {unknown}. Available: {avail}"
            )

    results: Dict[str, float] = {}
    for key in simple_keys:
        try:
            results[key] = _METRIC_FUNCTIONS[key](spectrum1, spectrum2)
        except Exception:
            results[key] = float("nan")

    for key in eurados_keys:
        if energy is None:
            results[key] = float("nan")
            continue
        try:
            func = _METRIC_FUNCTIONS_WITH_PARAMS[key]
            if key == "fluence_difference_percent":
                results[key] = func(spectrum1, spectrum2)
            elif key == "energy_group_fluence_diff":
                groups = func(spectrum1, spectrum2, energy)
                for group_name, group_val in groups.items():
                    results[f"energy_group_fluence_diff_{group_name}"] = group_val
            elif key == "fluence_averaged_energy_diff":
                results[key] = func(spectrum1, spectrum2, energy)
            elif key in ("log_lethargy_correlation", "peak_location_error", "peak_width_error"):
                results[key] = func(spectrum1, spectrum2, energy)
            elif key == "response_matrix_consistency":
                if readings1 is not None and response_matrix is not None:
                    results["response_matrix_consistency_ref"] = func(spectrum1, readings1, response_matrix)
                    results["response_matrix_consistency_test"] = func(spectrum2, readings2 if readings2 is not None else readings1, response_matrix)
            else:
                results[key] = func(spectrum1, spectrum2, energy, cc_icrp116)
        except Exception:
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
        Metric name(s). If None, all available metrics are computed.
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



