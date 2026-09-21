"""OSEM-ANLM (ordered-subset EM with asymptotic non-local means) method.

Port of the OSEM-ANLM algorithm of Jamaati et al. (2026), "Enhanced sparse
view CT reconstruction using ordered subset expectation maximization and
asymptotic non-local means algorithms", Scientific Reports, Article in Press
(https://doi.org/10.1038/s41598-026-70607-1), adapted to neutron spectrum
unfolding from Bonner sphere readings.

The algorithm alternates ordered-subset expectation maximisation (OSEM)
updates with the asymptotic non-local means (ANLM) filter applied to the
intermediate spectrum after every subset update (article pseudo-code steps
4-5).  The ANLM filter is a two-stage non-local means (NLM) filter:

* stage 1 applies the uniform parameter ``h1(i) = 0.5 * sigma`` ("the first
  filter applies h1(i) = 0.5 sigma uniformly across all points");
* stage 2 applies the point-wise parameter ``h2(i) = sigma_2(i) =
  sqrt(sum_{j in N_i} w(i,j)^2 * sigma^2)`` (article eq. 6), i.e. the noise
  standard deviation smoothed by the initial NLM weights ``w(i, j)`` of the
  first stage.

For one-dimensional spectra the 2D NLM windows of the article become index
windows on the energy grid: the search window ``N`` locates similar
neighbouring bins, and the similarity (patch) window ``nu`` with a Gaussian
kernel of spread ``alpha`` weights the squared-L2 patch distance

    d(i, j) = sum_k G_alpha(k) * (x(i + k) - x(j + k))^2,

so the NLM weights are ``w(i, j) = exp(-d(i, j) / h^2) / z(i)`` (article
eq. 5 with the normalisation factor ``z(i)``).  With the article's optimal
settings ``N = 11``, ``nu = 3`` the filter preserves spectral structure
(peaks and edges) while suppressing the streak-like oscillations produced
by sparse (few-detector) data, exactly as it removes sparse-view streak
artefacts in CT.

The filter parameter ``h`` corresponds to the noise level ``sigma`` of the
reconstructed spectrum (article: "the h amount is chosen based on the
image's noise level").  When ``h=None`` the noise level is estimated
automatically at every ANLM application from the current iterate with a
robust median-absolute-deviation estimator on the second differences, which
makes the method scale-free for spectra spanning several orders of
magnitude.  Because Bonner-sphere spectra span several orders of magnitude
(unlike CT images in Hounsfield units) and the EM noise amplitude scales
with the local fluence, the filter by default operates on the logarithm of
the spectrum (``log_space=True``); pass ``log_space=False`` to reproduce
the raw-unit filtering of the original CT formulation.
"""

from typing import Any

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = [
    "anlm_filter_1d",
    "estimate_noise_1d",
    "solve_osem_anlm",
    "unfold_osem_anlm",
]

_ANLM_MODES = ("subset", "post")


def estimate_noise_1d(x: np.ndarray) -> float:
    """Estimate the noise standard deviation of a 1D signal.

    Uses the robust median-absolute-deviation (MAD) estimator on the
    second-difference residual.  For i.i.d. noise of standard deviation
    ``sigma`` the second difference ``d2[k] = x[k+1] - 2 x[k] + x[k-1]``
    has variance ``6 * sigma^2`` and ``E|d2| = 0.6745 * sqrt(6) * sigma``,
    hence

        sigma_hat = MAD(|d2|) / (0.6745 * sqrt(6)),

    the 1D analogue of Immerkaer's fast noise-variance estimator.  The
    second difference annihilates linear trends, so slowly varying spectral
    structure does not inflate the estimate.

    Parameters
    ----------
    x : np.ndarray
        Input signal (n,).

    Returns
    -------
    float
        Estimated noise standard deviation (strictly positive; a tiny
        relative floor is enforced so the value can safely be used as an
        NLM filter parameter).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return 0.0
    d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    mad = float(np.median(np.abs(d2)))
    sigma = mad / (0.6745 * np.sqrt(6.0))
    # Strictly positive floor relative to the signal scale: guarantees the
    # value is usable as an NLM h parameter even for perfectly smooth input.
    floor = 1e-12 * float(np.max(np.abs(x))) + 1e-300
    return float(max(sigma, floor))


def _reflect_indices(offsets: np.ndarray, n: int) -> np.ndarray:
    """Map shifted indices back into ``[0, n)`` by symmetric reflection."""
    idx = np.arange(n)[:, None] + offsets[None, :]
    if n == 1:
        return np.zeros_like(idx)
    period = 2 * (n - 1)
    idx = np.abs(idx) % period
    return np.where(idx >= n, period - idx, idx)


def anlm_filter_1d(
    x: np.ndarray,
    h: float | None = None,
    search_window: int = 11,
    similarity_window: int = 3,
    alpha: float = 1.0,
    log_space: bool = True,
) -> np.ndarray:
    """Two-stage asymptotic non-local means filter for 1D spectra.

    Implements the ANLM regularisation of Jamaati et al. (2026) (eqs. 4-6
    and the ANLM filter section) adapted to a one-dimensional energy grid:

    1. Patch (similarity-window) distances ``d(i, j)`` are computed with a
       Gaussian kernel of spread ``alpha`` over the similarity window
       ``nu``; indices outside the signal are reflected at the borders.
    2. Stage 1 applies NLM with the uniform parameter ``h1 = 0.5 * sigma``,
       producing a lightly denoised intermediate spectrum and the "initial"
       normalised weights ``w1(i, j)``.
    3. Stage 2 applies NLM with the point-wise parameter of article eq. 6,
       ``h2(i) = sqrt(sum_j w1(i, j)^2 * sigma^2)`` — the noise standard
       deviation smoothed by the initial weights — to the stage-1 output,
       incrementally reducing the noise while preserving structure.

    By default (``log_space=True``) the filter operates on the logarithm
    of the spectrum: Bonner-sphere spectra span several orders of
    magnitude and the EM noise amplitude scales with the local fluence,
    so a single absolute filter parameter cannot match every bin.  In log
    space the relative noise level is uniform across the grid, making the
    automatic ``h`` estimate scale-free (the filtered value becomes a
    weighted geometric mean, which also preserves non-negativity).  Set
    ``log_space=False`` to filter in raw units exactly as the original CT
    formulation of the article.

    Parameters
    ----------
    x : np.ndarray
        Input spectrum (n,), typically non-negative.
    h : float, optional
        Noise level ``sigma`` used by both stages (in log units when
        ``log_space=True``).  If None (default) the noise level is
        estimated automatically with :func:`estimate_noise_1d` (robust
        MAD on second differences).
    search_window : int, optional
        Size ``N`` of the search window around each bin (article optimum:
        11).  Must be a positive integer; even values are rounded down to
        the preceding odd size.
    similarity_window : int, optional
        Size ``nu`` of the Gaussian-weighted similarity (patch) window
        (article optimum: 3).  Must be a positive integer; even values are
        rounded down to the preceding odd size.
    alpha : float, optional
        Spread of the Gaussian kernel over the similarity window
        (default: 1.0).  Must be positive.
    log_space : bool, optional
        Filter the logarithm of the spectrum instead of the raw values
        (default: True, scale-free for spectra spanning orders of
        magnitude).

    Returns
    -------
    np.ndarray
        Filtered spectrum (n,).  Non-negative when ``log_space=True``;
        reduces to the identity for ``search_window == 1``.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        raise ValueError("input spectrum must be non-empty")
    if search_window < 1:
        raise ValueError(f"search_window must be >= 1, got {search_window}")
    if similarity_window < 1:
        raise ValueError(
            f"similarity_window must be >= 1, got {similarity_window}"
        )
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if h is not None and not (float(h) > 0):
        raise ValueError(f"h must be a positive noise level, got {h!r}")

    if n == 1:
        return x.copy()
    if search_window // 2 == 0:
        # No neighbours in the search window: the NLM weights collapse to
        # the self bin, so the filter is the exact identity.
        return x.copy()

    sigma = float(h) if h is not None else estimate_noise_1d(x)
    # Strictly positive floor: estimate_noise_1d returns 0.0 for fewer than
    # 3 samples, and a zero filter parameter would produce 0/0 weights.
    sigma = max(sigma, 1e-12 * float(np.max(np.abs(x))) + 1e-300)

    if log_space:
        # Log domain: add a tiny relative floor so zero bins stay finite.
        work = np.log(x + 1e-12 * float(np.max(x)) + 1e-300)
    else:
        work = x.copy()

    search_r = search_window // 2
    half_v = similarity_window // 2
    offsets = np.arange(-half_v, half_v + 1)
    gauss = np.exp(-0.5 * (offsets / alpha) ** 2)
    gauss = gauss / gauss.sum()

    # Reflected patch columns: P[i, t] = x[reflect(i + offsets[t])]
    refl = _reflect_indices(offsets, n)
    patches = x[refl]

    def _window_bounds(i: int) -> tuple[int, int]:
        return max(0, i - search_r), min(n, i + search_r + 1)

    def _nlm_pass(signal: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        """One NLM pass with per-bin filter parameters ``sigmas``."""
        out = np.empty(n)
        sig_patches = signal[refl]
        for i in range(n):
            lo, hi = _window_bounds(i)
            diff = sig_patches[i] - sig_patches[lo:hi]
            dist = diff**2 @ gauss  # (W_i,) Gaussian-weighted patch distances
            # sqrt form avoids h^2 underflow for extremely small h; the
            # clip keeps the squared ratio finite when ``sigmas[i]`` is a
            # denormal (the weight is exp(-inf) = 0 either way).
            ratio = np.minimum(np.sqrt(dist) / sigmas[i], 1e150)
            weights = np.exp(-(ratio**2))
            z = weights.sum()  # >= 1: the self-bin weight is exp(0) = 1
            out[i] = weights @ signal[lo:hi] / z
        return out

    # Stage 1: uniform h1 = 0.5 * sigma (article, ANLM filter section).
    h1 = 0.5 * sigma
    intermediate = _nlm_pass(work, np.full(n, h1))
    w2_sq = np.empty(n)
    for i in range(n):
        lo, hi = _window_bounds(i)
        diff = patches[i] - patches[lo:hi]
        dist = diff**2 @ gauss
        ratio = np.minimum(np.sqrt(dist) / h1, 1e150)
        weights = np.exp(-(ratio**2))
        w1 = weights / weights.sum()
        # Article eq. (6): h2(i) = sigma_2(i) = sqrt(sum_j w(i, j)^2 sigma^2)
        w2_sq[i] = (w1**2).sum()
    h2 = sigma * np.sqrt(w2_sq)

    # Stage 2: point-wise h2(i) applied to the stage-1 output.
    filtered = _nlm_pass(intermediate, h2)
    return np.exp(filtered) if log_space else filtered


def solve_osem_anlm(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    max_iterations: int = 50,
    n_subsets: int = 1,
    tolerance: float = 1e-6,
    h: float | None = None,
    search_window: int = 11,
    similarity_window: int = 3,
    alpha: float = 1.0,
    anlm_mode: str = "subset",
    log_space: bool = True,
) -> tuple[np.ndarray, int, bool]:
    """Solve the unfolding problem with OSEM-ANLM.

    OSEM update (article eq. 3 / pseudo-code step 4):

        x^{n+1} = x^n * A_m^T ( b_m / (A_m x^n + eps) ) / ( A_m^T 1 + eps )

    followed by the ANLM filter (pseudo-code step 5,
    ``f^{*(n+1,b)} = ANLMFilter(mu^{*(n+1,b)})``) applied after every
    subset update when ``anlm_mode="subset"`` (default, per the article
    pseudo-code).  With ``anlm_mode="post"`` the plain OSEM solution is
    produced first and the ANLM filter is applied once at the end
    ("OSEM reconstruction followed by ANLM regularization" in the article
    abstract).  With ``n_subsets=1`` the OSEM update reduces to standard
    MLEM.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray
        Initial spectrum guess (n,).
    max_iterations : int, optional
        Maximum number of full iterations (sweeps over all subsets)
        (default: 50).
    n_subsets : int, optional
        Number of ordered subsets over the detector readings (default: 1,
        i.e. standard MLEM with per-iteration ANLM).
    tolerance : float, optional
        Relative change tolerance for early stopping (default: 1e-6).
    h : float, optional
        Noise level for the ANLM filter.  If None (default), it is
        estimated automatically from each intermediate spectrum.
    search_window : int, optional
        ANLM search window ``N`` (default: 11, article optimum).
    similarity_window : int, optional
        ANLM similarity (patch) window ``nu`` (default: 3, article optimum).
    alpha : float, optional
        Spread of the Gaussian kernel over the similarity window
        (default: 1.0).
    anlm_mode : str, optional
        ``'subset'`` — ANLM after every subset update (default, article
        pseudo-code); ``'post'`` — single ANLM application to the OSEM
        result.
    log_space : bool, optional
        Apply the ANLM filter to the logarithm of the spectrum (default:
        True, scale-free for spectra spanning orders of magnitude).  See
        :func:`anlm_filter_1d`.

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (solution spectrum, iterations used, converged flag).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    m, _ = A.shape

    if n_subsets < 1:
        raise ValueError("n_subsets must be >= 1")
    if n_subsets > m:
        raise ValueError(
            f"n_subsets ({n_subsets}) must not exceed the number of "
            f"detectors ({m})"
        )
    if anlm_mode not in _ANLM_MODES:
        raise ValueError(
            f"anlm_mode must be one of {_ANLM_MODES}, got {anlm_mode!r}"
        )
    if h is not None and not (float(h) > 0):
        raise ValueError(f"h must be a positive noise level, got {h!r}")
    if search_window < 1:
        raise ValueError(f"search_window must be >= 1, got {search_window}")
    if similarity_window < 1:
        raise ValueError(
            f"similarity_window must be >= 1, got {similarity_window}"
        )
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    eps = 1e-11
    subset_indices = np.array_split(np.arange(m), n_subsets)
    x = np.maximum(x0, 0).copy()
    converged = False
    iterations = 0

    def _anlm(current: np.ndarray) -> np.ndarray:
        return anlm_filter_1d(
            current,
            h=h,
            search_window=search_window,
            similarity_window=similarity_window,
            alpha=alpha,
            log_space=log_space,
        )

    for it in range(1, max_iterations + 1):
        iterations = it
        x_old = x.copy()

        for idx in subset_indices:
            A_sub = A[idx]
            b_sub = b[idx]
            norm = A_sub.sum(axis=0)
            ratio = b_sub / (A_sub @ x + eps)
            correction = A_sub.T @ ratio
            x = np.maximum(x * correction / (norm + eps), 0.0)
            if anlm_mode == "subset":
                # Article pseudo-code step 5: ANLM after every subset update.
                x = _anlm(x)

        rel = np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + eps)
        if rel < tolerance:
            converged = True
            break

    if anlm_mode == "post":
        # Article abstract: "OSEM reconstruction followed by ANLM
        # regularization".
        x = _anlm(x)

    return x, iterations, converged


def unfold_osem_anlm(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    initial_spectrum: np.ndarray | None = None,
    max_iterations: int = 50,
    n_subsets: int = 1,
    tolerance: float = 1e-6,
    h: float | None = None,
    search_window: int = 11,
    similarity_window: int = 3,
    alpha: float = 1.0,
    anlm_mode: str = "subset",
    log_space: bool = True,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Unfold neutron spectrum using the OSEM-ANLM algorithm.

    Ordered-subset expectation maximisation with asymptotic non-local means
    regularization (Jamaati et al. 2026,
    https://doi.org/10.1038/s41598-026-70607-1), adapted to Bonner sphere
    spectra: the ANLM filter is applied to the intermediate spectrum after
    every OSEM subset update (``anlm_mode="subset"``) or once to the OSEM
    result (``anlm_mode="post"``).

    Parameters
    ----------
    detector_names : List[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid.
    sensitivities : Dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : Dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback to save result to history.
    readings : Dict[str, float]
        Detector readings.
    initial_spectrum : Optional[np.ndarray], optional
        Initial spectrum guess. If None, a flat spectrum is used.
    max_iterations : int, optional
        Maximum number of iterations (default: 50).
    n_subsets : int, optional
        Number of ordered subsets over the detector readings (default: 1,
        i.e. standard MLEM with per-iteration ANLM).
    tolerance : float, optional
        Relative change tolerance for early stopping (default: 1e-6).
    h : float, optional
        Noise level for the ANLM filter.  If None (default), it is
        estimated automatically from the intermediate spectra.
    search_window : int, optional
        ANLM search window ``N`` (default: 11, article optimum).
    similarity_window : int, optional
        ANLM similarity (patch) window ``nu`` (default: 3, article optimum).
    alpha : float, optional
        Spread of the Gaussian kernel over the similarity window
        (default: 1.0).
    anlm_mode : str, optional
        ``'subset'`` — ANLM after every subset update (default, article
        pseudo-code); ``'post'`` — single ANLM application to the OSEM
        result.
    log_space : bool, optional
        Apply the ANLM filter to the logarithm of the spectrum (default:
        True, scale-free for spectra spanning orders of magnitude).  See
        :func:`anlm_filter_1d`.
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Unfolding results dictionary.
    """
    x0_default = np.ones(n_energy_bins)
    x0_default[0] = 0.0

    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=make_solve_wrapper(
            solve_osem_anlm,
            max_iterations=max_iterations,
            n_subsets=n_subsets,
            tolerance=tolerance,
            h=h,
            search_window=search_window,
            similarity_window=similarity_window,
            alpha=alpha,
            anlm_mode=anlm_mode,
            log_space=log_space,
        ),
        solve_kwargs={},
        method_name="OSEM-ANLM",
        extra_output={
            "n_subsets": int(n_subsets),
            "h": None if h is None else float(h),
            "search_window": int(search_window),
            "similarity_window": int(similarity_window),
            "alpha": float(alpha),
            "anlm_mode": str(anlm_mode),
            "log_space": bool(log_space),
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
