"""Bayesian unfolding methods powered by CUQIpy (https://github.com/CUQI-DTU/CUQIpy).

This module integrates the samplers and distributions of the CUQIpy library
(Computational Uncertainty Quantification for Inverse Problems, DTU) into the
bssunfold Bonner-sphere spectrum unfolding workflow.

Statistical model
-----------------
The spectrum is modelled on the log scale (``f = exp(theta)``) with a
smoothness prior anchored on a data-driven center (the non-negative
least-squares solution or a user-supplied ``initial_spectrum``), following
the same well-tested formulation as :mod:`bssunfold.core.unfold_mcmc`.  The
likelihood is Gaussian with a relative noise scale ``sigma = noise_level *
|b|``:

- Prior ``'gmrf'`` (default): ``theta ~ GMRF(mu, prec)`` using the CUQIpy
  finite-difference precision operator; ``gmrf_order`` (1 or 2) controls the
  smoothness of the implied random walk.
- Prior ``'ou'``: ``theta ~ Gaussian(mu, C_ou / prec)`` with the dense
  Ornstein-Uhlenbeck correlation ``C_ou[i, j] = exp(-|i - j| /
  lengthscale)``.

Samplers (all from :mod:`cuqi.sampler`)
---------------------------------------
- ``'pcn'``   : Preconditioned Crank-Nicolson.
- ``'cwmh'``  : Component-wise random-walk Metropolis-Hastings.
- ``'nuts'``  : No-U-Turn Sampler (gradient based).
- ``'mala'``  : Metropolis-adjusted Langevin algorithm (gradient based).
- ``'ula'``   : Unadjusted Langevin algorithm (gradient based, experimental).
- ``'gibbs'``     : Hierarchical Gibbs (CUQIpy ``HybridGibbs``): the GMRF
  smoothness precision ``delta`` is inferred from the data through a
  conjugate Gamma-GMRF update while the spectrum block uses PCN.
- ``'gibbs_nuts'``: Same hierarchical Gibbs scheme with NUTS for the
  spectral block.

Sampler parameterizations (statistically equivalent, chosen for robustness)
---------------------------------------------------------------------------
Bonner-sphere unfolding posteriors span many decades of stiffness per energy
bin, which defeats naive random-walk samplers.  Each sampler therefore uses
the formulation in which CUQIpy's own adaptation works best:

- ``'pcn'`` and ``'cwmh'`` sample the *centered* log-spectrum
  ``t = theta - mu`` (pCN contracts towards the prior mean, so centering is
  essential for the correct scaling of its proposals).
- ``'nuts'`` samples ``theta`` directly with the native CUQIpy posterior.
- ``'mala'`` and ``'ula'`` automatically sample a *Laplace-whitened*
  coordinate ``z`` with ``theta = mu + t_map + L z`` where ``t_map`` is a
  Gauss-Newton MAP estimate and ``L`` is the Cholesky factor of the inverse
  Gauss-Newton curvature.  The whitened posterior is approximately
  isotropic, which is what scalar-step Langevin samplers require.
- ``'gibbs'`` / ``'gibbs_nuts'`` use the native hierarchical GMRF model so
  that the conjugate Gamma update of the precision is available.

CUQIpy is an *optional* dependency: it is imported lazily on first use and
:mod:`bssunfold` keeps working without it (``CUQI_AVAILABLE`` flag in
:mod:`bssunfold.platform_check`).
"""

from typing import Any

import numpy as np

from ._base_unfolder import run_unfolding
from .unfold_mcmc import _hpd_interval, _prior_center

__all__ = ["solve_cuqi_bayesian", "unfold_cuqi", "check_cuqi_available"]

# ---------------------------------------------------------------------------
# Lazy CUQIpy loading (PEP 562 module __getattr__)
#
# ``cuqi`` pulls in scipy/xarray/arviz and friends; importing it eagerly at
# module scope would slow down ``import bssunfold`` for users who never touch
# the CUQIpy methods.  Instead it is imported on first attribute access, and
# the loader caches results in the module namespace so repeated lookups are
# free.  The pattern mirrors ``unfold_mcmc`` (PyMC lazy loader).
# ---------------------------------------------------------------------------

_cuqi = None
_cuqi_checked = False


def _load_cuqi() -> Any:
    """Import cuqi on first use; cache result in the module globals.

    Returns
    -------
    Optional[Any]
        The ``cuqi`` module, or ``None`` when unavailable.
    """
    global _cuqi, _cuqi_checked
    import sys as _sys

    if not _cuqi_checked:
        # Purge cached entries so blocked-import test fixtures (which patch
        # builtins.__import__) can actually intercept the import.
        _sys.modules.pop("cuqi", None)
        try:
            import cuqi as _cuqi_mod

            _cuqi = _cuqi_mod
        except Exception:
            _cuqi = None
        _cuqi_checked = True
    return _cuqi


def check_cuqi_available() -> bool:
    """Report CUQIpy availability, honoring an externally patched flag."""
    if not _cuqi_checked:
        _load_cuqi()
    available = _cuqi is not None
    g = globals()
    if "CUQI_AVAILABLE" in g and available:
        return bool(g["CUQI_AVAILABLE"])
    return available


def __getattr__(name: str) -> Any:
    if name == "cuqi":
        mod = _load_cuqi()
        globals()["cuqi"] = mod
        return mod
    if name == "CUQI_AVAILABLE":
        available = check_cuqi_available()
        globals()["CUQI_AVAILABLE"] = available
        return available
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Supported sampler identifiers (public API)
_VALID_SAMPLERS = (
    "pcn",
    "cwmh",
    "ula",
    "mala",
    "nuts",
    "gibbs",
    "gibbs_nuts",
)
_LANGEVIN_SAMPLERS = ("ula", "mala")  # auto Laplace-whitened
_GIBBS_SAMPLERS = ("gibbs", "gibbs_nuts")

# Default proposal scales per sampler (linear log-spectrum units, except the
# whitened Langevin samplers where the scale is in whitened units).
_DEFAULT_SCALES = {
    "pcn": 0.05,
    "cwmh": 0.05,
    "nuts": 0.05,
    "mala": 0.3,
    "ula": 0.01,
    "gibbs": 0.05,
    "gibbs_nuts": 0.05,
}


def _ou_correlation(n_bins: int, lengthscale: float) -> np.ndarray:
    """Dense Ornstein-Uhlenbeck correlation matrix.

    ``C[i, j] = exp(-|i - j| / lengthscale)`` — the same smoothness structure
    used by the PyMC-based :func:`bssunfold.core.unfold_mcmc.solve_bayesian_mcmc`.
    """
    idx = np.arange(n_bins)
    corr = np.exp(-np.abs(idx[:, None] - idx[None, :]) / max(float(lengthscale), 1e-9))
    return corr + 1e-9 * np.eye(n_bins)


def _ou_precision(n_bins: int, lengthscale: float) -> np.ndarray:
    """Dense precision (inverse correlation) of the OU prior."""
    corr = _ou_correlation(n_bins, lengthscale)
    return np.linalg.inv(corr) + 1e-9 * np.eye(n_bins)


def _prior_center_nnls(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    initial_spectrum: np.ndarray | None,
    n_energy: int,
) -> np.ndarray:
    """Data-driven log-space prior center for the spectrum.

    Uses the user-supplied ``initial_spectrum`` when available.  Otherwise a
    true non-negative least-squares solution (``scipy.optimize.nnls``) is
    computed — plain ``lstsq`` clipped at zero can return an almost-zero
    center for the severely underdetermined Bonner-sphere system, which
    would collapse the log-scale prior towards zero flux.
    """
    if initial_spectrum is not None:
        center = np.maximum(np.asarray(initial_spectrum, dtype=float), 0.0)
        if center.ndim != 1 or len(center) != n_energy:
            center = np.zeros(n_energy)
    else:
        try:
            from scipy.optimize import nnls

            center, _ = nnls(A_matrix, b_readings)
        except Exception:
            center = np.maximum(
                np.linalg.lstsq(A_matrix, b_readings, rcond=None)[0], 0.0
            )
        center = np.maximum(np.asarray(center, dtype=float), 0.0)
        if not np.any(center > 0):
            center = np.ones(n_energy)
    return np.log(np.maximum(center, 1e-6))


def _gmrf_precision(n_bins: int, order: int) -> np.ndarray:
    """Dense first/second-order difference precision with zero BCs.

    Mirrors the structure of the CUQIpy ``GMRF`` finite-difference precision
    operator (order 1: first differences, order 2: second differences) and is
    used for the Laplace whitening of the Langevin samplers.
    """
    if order not in (1, 2):
        raise ValueError(f"gmrf_order must be 1 or 2, got {order!r}")
    diff = np.zeros((max(n_bins - order, 1), n_bins))
    for i in range(diff.shape[0]):
        diff[i, i] = 1.0
        diff[i, i + 1] = -1.0
        if order == 2:
            diff[i, i + 2] = 1.0
            diff[i, i + 1] = -2.0
    precision = diff.T @ diff
    # Anchor the first bin(s) like the zero boundary condition does, so the
    # precision matrix is positive definite.
    precision[0, 0] += 1.0
    if order == 2:
        precision[1, 1] += 1.0
    return precision + 1e-9 * np.eye(n_bins)


def _build_forward_model(A_matrix: np.ndarray, cuqi_mod: Any, center: np.ndarray | None = None) -> Any:
    """Build the CUQIpy forward model ``A @ exp(theta)``.

    When ``center`` is given, the model becomes ``A @ exp(center + theta)``
    (the *centered* parameterization used by the PCN/CWMH samplers).

    The analytic vector-Jacobian product is provided so gradient-based
    samplers (ULA, MALA, NUTS) work out of the box: for ``f(t) = A exp(t)``
    the Jacobian is ``J = diag(exp(t)) A^T``, hence ``J^T d = exp(t) (A^T d)``.
    """
    n_detectors, n_energy = A_matrix.shape

    if center is None:

        def _forward(theta):
            return A_matrix @ np.exp(theta)

        def _gradient(direction, theta):
            return np.exp(theta) * (A_matrix.T @ direction)

    else:

        def _forward(theta):
            return A_matrix @ np.exp(center + theta)

        def _gradient(direction, theta):
            return np.exp(center + theta) * (A_matrix.T @ direction)

    return cuqi_mod.model.Model(
        _forward,
        range_geometry=n_detectors,
        domain_geometry=n_energy,
        gradient=_gradient,
    )


def _gauss_newton_map(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    sigma2: np.ndarray,
    mu: np.ndarray,
    prior_precision: np.ndarray,
    n_iter: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Newton MAP estimate of the log-spectrum and its curvature.

    Maximizes ``log p(theta | b)`` for the model ``b = A exp(theta)`` with a
    Gaussian smoothness prior (precision ``prior_precision``) using damped
    Gauss-Newton iterations with backtracking line search.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(theta_map, whitening_cholesky)`` where ``whitening_cholesky`` is
        the lower-triangular Cholesky factor ``L`` of the inverse curvature
        (the Laplace posterior covariance approximation), such that the
        whitened coordinate ``z`` solves ``theta = theta_map + L @ z``.
    """
    n_energy = A_matrix.shape[1]
    theta = mu.copy()
    current_logp = -np.inf

    def _logp(t):
        resid = b_readings - A_matrix @ np.exp(t)
        prior_dev = t - mu
        return (
            -0.5 * np.sum(resid**2 / sigma2)
            - 0.5 * prior_dev @ prior_precision @ prior_dev
        )

    current_logp = _logp(theta)
    for _ in range(int(n_iter)):
        resid = b_readings - A_matrix @ np.exp(theta)
        jacobian = A_matrix * np.exp(theta)[None, :]
        curvature = prior_precision + jacobian.T @ (jacobian / sigma2[:, None])
        grad = jacobian.T @ (resid / sigma2) - prior_precision @ (theta - mu)
        try:
            step = np.linalg.solve(curvature + 1e-10 * np.eye(n_energy), grad)
        except np.linalg.LinAlgError:
            break
        # Backtracking line search on the log posterior
        alpha = 1.0
        improved = False
        for _ in range(25):
            candidate = theta + alpha * step
            candidate_logp = _logp(candidate)
            if np.isfinite(candidate_logp) and candidate_logp >= current_logp - 1e-4 * alpha * (grad @ step):
                improved = True
                break
            alpha *= 0.5
        if not improved or alpha < 1e-9:
            break
        theta = candidate
        current_logp = candidate_logp
        if np.linalg.norm(alpha * step) < 1e-12:
            break

    resid = b_readings - A_matrix @ np.exp(theta)
    jacobian = A_matrix * np.exp(theta)[None, :]
    curvature = prior_precision + jacobian.T @ (jacobian / sigma2[:, None])
    laplace_cov = np.linalg.inv(curvature + 1e-10 * np.eye(n_energy))
    whitening = np.linalg.cholesky(laplace_cov + 1e-10 * np.eye(n_energy))
    return theta, whitening


def _run_fixed_precision_chain(
    sampler: str,
    target: Any,
    n_samples: int,
    n_burnin: int,
    thin: int,
    seed: int | None,
    scale: float,
    max_depth: int | None,
    step_size: float | None,
    initial_point: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    """Run one MCMC chain with a fixed-precision CUQIpy sampler.

    ``target`` is either a CUQIpy ``Posterior`` (pcn/cwmh/nuts) or a
    ``UserDefinedDistribution`` (mala/ula).

    Returns
    -------
    Tuple[np.ndarray, float]
        ``(samples, acceptance_rate)`` where ``samples`` has shape
        ``(n_samples, n_energy)`` in the *sampled* coordinate (centered or
        whitened log-spectrum, depending on the sampler).
    """
    from cuqi.sampler import CWMH, MALA, NUTS, PCN, ULA

    # CUQIpy samplers draw from the legacy numpy global RNG, so seeding the
    # global state per chain gives reproducible, independent chains.
    if seed is not None:
        np.random.seed(seed)

    if sampler == "pcn":
        sampler_obj = PCN(target, scale=scale, initial_point=initial_point)
    elif sampler == "cwmh":
        sampler_obj = CWMH(target, scale=scale, initial_point=initial_point)
    elif sampler == "nuts":
        sampler_obj = NUTS(
            target,
            max_depth=max_depth,
            step_size=step_size,
            initial_point=initial_point,
        )
    elif sampler == "mala":
        sampler_obj = MALA(target, scale=scale, initial_point=initial_point)
    elif sampler == "ula":
        sampler_obj = ULA(target, scale=scale, initial_point=initial_point)
    else:  # pragma: no cover - validated upstream
        raise ValueError(f"Unknown sampler {sampler!r}")

    sampler_obj.warmup(n_burnin)
    sampler_obj.sample(n_samples, Nt=thin)

    samples = sampler_obj.get_samples()
    arr = np.asarray(samples.samples, dtype=float)  # (dim, Ns_total)
    # CUQIpy stores warmup draws followed by the sampling-phase draws; keep
    # only the sampling-phase samples (count depends on the thinning Nt).
    n_stored = max(int(n_samples) // max(int(thin), 1), 1)
    if arr.ndim == 2 and arr.shape[1] > n_stored:
        arr = arr[:, arr.shape[1] - n_stored :]

    acc = getattr(sampler_obj, "_acc", None)
    if acc is not None and len(acc):
        acc_tail = acc[-int(n_samples) :] if len(acc) >= n_samples else acc
        acc_rate = float(np.mean(acc_tail))
    else:
        acc_rate = np.nan
    return arr.T, acc_rate


def _run_gibbs_chain(
    sampler: str,
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    sigma2: float,
    mu: np.ndarray,
    gmrf_order: int,
    delta_alpha: float,
    delta_beta: float,
    n_samples: int,
    n_burnin: int,
    thin: int,
    seed: int | None,
    scale: float,
    max_depth: int | None,
    step_size: float | None,
    centered: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run one hierarchical Gibbs chain (CUQIpy ``HybridGibbs``).

    The hierarchical model is::

        delta ~ Gamma(delta_alpha, delta_beta)          # smoothness precision
        theta | delta ~ GMRF(mu, prec=delta)            # log-spectrum
        b | theta ~ Gaussian(A @ exp(theta), sigma^2)   # likelihood

    ``delta`` is updated with the analytically conjugate CUQIpy ``Conjugate``
    sampler (Gamma-GMRF pair) and the spectrum with ``PCN`` (``'gibbs'``) or
    ``NUTS`` (``'gibbs_nuts'``).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        ``(theta_samples, delta_samples, acceptance_rate)``.
    """
    cuqi_mod = _load_cuqi()
    if seed is not None:
        np.random.seed(seed)

    from cuqi.distribution import Gamma, Gaussian, GMRF, JointDistribution
    from cuqi.sampler import Conjugate, HybridGibbs, NUTS, PCN

    # PCN proposals contract towards the prior mean, so the 'gibbs' spectral
    # block runs on the *centered* log-spectrum t = theta - mu (statistically
    # identical model).  NUTS ('gibbs_nuts') is unaffected and samples the
    # native uncentered parameterization.
    forward = _build_forward_model(A_matrix, cuqi_mod, center=mu if centered else None)
    prior_mean = np.zeros_like(mu) if centered else mu

    # NOTE: variable names (delta/theta/y) define the CUQIpy parameter names
    # used by JointDistribution and the sampling-strategy dict — keep them.
    delta = Gamma(delta_alpha, delta_beta)
    theta = GMRF(mean=prior_mean, prec=lambda delta: delta, bc_type="zero", order=gmrf_order)
    y = Gaussian(mean=forward(theta), cov=sigma2)
    joint = JointDistribution(delta, theta, y)
    conditioned = joint(y=b_readings)

    if sampler == "gibbs":
        spectral = PCN(scale=scale)
    else:
        # A fixed leapfrog step size keeps the inner NUTS block stable: its
        # dual-averaging adaptation is restarted at every Gibbs scan and can
        # collapse on stiff posteriors.  0.05 is a robust default here; pass
        # ``step_size`` explicitly to override.
        spectral = NUTS(
            max_depth=max_depth, step_size=step_size if step_size is not None else 0.05
        )

    gibbs = HybridGibbs(
        conditioned,
        {"delta": Conjugate(), "theta": spectral},
    )
    gibbs.warmup(n_burnin)
    gibbs.sample(n_samples, Nt=thin)

    samples = gibbs.get_samples()
    theta_arr = np.asarray(samples["theta"].samples, dtype=float)
    delta_arr = np.asarray(samples["delta"].samples, dtype=float)
    n_stored = max(int(n_samples) // max(int(thin), 1), 1)
    if theta_arr.ndim == 2 and theta_arr.shape[1] > n_stored:
        theta_arr = theta_arr[:, theta_arr.shape[1] - n_stored :]
    if delta_arr.ndim == 2 and delta_arr.shape[1] > n_stored:
        delta_arr = delta_arr[:, delta_arr.shape[1] - n_stored :]
    theta_s = theta_arr.T + mu if centered else theta_arr.T
    delta_s = np.ravel(delta_arr)

    acc = getattr(spectral, "_acc", None)
    if acc is not None and len(acc):
        acc_tail = acc[-int(n_samples) :] if len(acc) >= n_samples else acc
        acc_rate = float(np.mean(acc_tail))
    else:
        acc_rate = np.nan
    return theta_s, delta_s, acc_rate


def _gelman_rubin(chains_stack: np.ndarray) -> np.ndarray:
    """Classic Gelman-Rubin potential scale reduction factor (R-hat).

    Parameters
    ----------
    chains_stack : np.ndarray
        Array of shape (n_chains, n_draws, n_params).

    Returns
    -------
    np.ndarray
        R-hat per parameter, shape (n_params,).
    """
    arr = np.asarray(chains_stack, dtype=float)
    m, n, n_params = arr.shape  # chains, draws, params
    if m < 2 or n < 2:
        return np.ones(n_params)

    chain_means = arr.mean(axis=1)  # (m, p)
    chain_vars = arr.var(axis=1, ddof=1)  # (m, p)
    grand_mean = chain_means.mean(axis=0)  # (p,)

    between = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)
    within = chain_vars.mean(axis=0)
    var_hat = (n - 1) / n * within + between / n
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / within)
    return np.where(np.isfinite(rhat), rhat, 1.0)


def theta_per_chain(theta_all: np.ndarray, chains: int, n_samples: int) -> list:
    """Split the stacked log-scale samples back into per-chain blocks."""
    return [theta_all[c * n_samples : (c + 1) * n_samples] for c in range(int(chains))]


def solve_cuqi_bayesian(
    A_matrix: np.ndarray,
    b_readings: np.ndarray,
    E: np.ndarray | None = None,
    log_steps: np.ndarray | None = None,
    sampler: str = "pcn",
    noise_level: float = 0.05,
    prior: str = "gmrf",
    gmrf_order: int = 1,
    lengthscale: float = 3.0,
    prec: float = 1.0,
    hierarchical: bool | None = None,
    delta_alpha: float = 1.0,
    delta_beta: float = 1e-4,
    n_samples: int = 2000,
    n_burnin: int = 1000,
    thin: int = 1,
    chains: int = 2,
    scale: float | None = None,
    max_depth: int = 8,
    step_size: float | None = None,
    credible_level: float = 95.0,
    initial_spectrum: np.ndarray | None = None,
    random_state: int | None = None,
    progressbar: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the unfolding problem with CUQIpy Bayesian samplers.

    The spectrum is modelled on the log scale ``f = exp(theta)`` with a
    smoothness prior centered on a data-driven guess (the non-negative
    least-squares solution of ``A @ x = b``, or the user-supplied
    ``initial_spectrum``) and a Gaussian likelihood with relative noise
    ``sigma = noise_level * |b|``.

    Priors
    ------
    - ``prior='gmrf'`` (default): ``theta ~ GMRF(mu, prec)`` — CUQIpy
      finite-difference precision operator, ``gmrf_order`` = 1 or 2.
    - ``prior='ou'``: ``theta ~ Gaussian(mu, C_ou / prec)`` — dense
      Ornstein-Uhlenbeck correlation with ``lengthscale`` bins.

    Samplers
    --------
    ``'pcn'``, ``'cwmh'``, ``'nuts'``, ``'mala'``, ``'ula'`` sample the
    fixed-precision posterior; ``'gibbs'`` and ``'gibbs_nuts'`` sample the
    hierarchical model where the GMRF precision has a Gamma hyperprior
    (``delta ~ Gamma(delta_alpha, delta_beta)``) and is inferred jointly
    with the spectrum.  Hierarchical sampling requires the ``'gmrf'`` prior.

    Internally each sampler uses the statistically equivalent formulation in
    which it mixes best on the severely ill-conditioned unfolding posterior:
    PCN/CWMH sample the centered log-spectrum ``theta - mu`` (required for
    the correct pCN proposal scaling), MALA/ULA are automatically
    Laplace-whitened around a Gauss-Newton MAP estimate, and NUTS and the
    Gibbs samplers use the native CUQIpy model.

    Parameters
    ----------
    A_matrix : np.ndarray
        Response matrix (n_detectors x n_energy).
    b_readings : np.ndarray
        Measured readings (n_detectors,).
    E : np.ndarray, optional
        Energy grid in MeV (unused by the model, kept for API consistency).
    log_steps : np.ndarray, optional
        Logarithmic energy steps (unused by the model, kept for API
        consistency; the forward model follows the package convention
        ``b = A @ spectrum``).
    sampler : str, optional
        CUQIpy sampler: ``'pcn'``, ``'cwmh'``, ``'ula'``, ``'mala'``,
        ``'nuts'``, ``'gibbs'`` or ``'gibbs_nuts'`` (default: ``'pcn'``).
    noise_level : float, optional
        Relative measurement noise scale (default: 0.05); the likelihood
        standard deviation is ``noise_level * |b|``.
    prior : str, optional
        Log-spectrum prior: ``'gmrf'`` (default) or ``'ou'``.
    gmrf_order : int, optional
        Order of the GMRF finite-difference operator, 1 or 2 (default: 1).
        Higher order yields smoother spectra.
    lengthscale : float, optional
        OU correlation length in energy bins for ``prior='ou'`` (default: 3).
    prec : float, optional
        Prior precision scale for ``theta`` (default: 1.0).  Ignored by the
        hierarchical Gibbs samplers, which infer it from the data.
    hierarchical : bool, optional
        Convenience switch: when True the Gibbs samplers are used.  If None
        (default) it is derived from ``sampler``.
    delta_alpha : float, optional
        Shape of the Gamma hyperprior on the GMRF precision (default: 1.0).
    delta_beta : float, optional
        Rate of the Gamma hyperprior on the GMRF precision (default: 1e-4).
    n_samples : int, optional
        Number of posterior samples per chain (default: 2000).
    n_burnin : int, optional
        Number of warmup/tuning iterations per chain (default: 1000).
        Increase (e.g. 3000+) for the Langevin samplers ``'mala'``/``'ula'``.
    thin : int, optional
        Thinning interval kept between stored samples (default: 1).
    chains : int, optional
        Number of independent chains (default: 2).
    scale : float, optional
        Proposal step size.  Defaults depend on the sampler (see
        ``_DEFAULT_SCALES``); for the whitened Langevin samplers the scale
        is in whitened units.
    max_depth : int, optional
        Maximum tree depth for NUTS (default: 8).
    step_size : float, optional
        Fixed leapfrog step size for NUTS; None lets CUQIpy tune it during
        warmup (default: None).
    credible_level : float, optional
        Credible mass (%) of the reported HPD interval (default: 95).
    initial_spectrum : np.ndarray, optional
        Prior center guess (n_energy,).  When None, the non-negative
        least-squares solution is used as the center.
    random_state : int, optional
        Random seed for reproducibility.
    progressbar : bool, optional
        Present for API consistency; CUQIpy progress bars are controlled via
        the ``TQDM_DISABLE`` environment variable (default: False).

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - spectrum: Mean posterior spectrum (n_energy,)
        - stats: Dictionary with 'samples' (linear scale, chains * n_samples
          x n_energy), 'theta_samples', 'mean', 'median', 'std',
          'hpd_lower'/'hpd_upper' (HPD interval), 'ess', 'rhat' (chains > 1),
          'acc_rate', 'delta_samples' (hierarchical samplers), and sampling
          metadata ('sampler', 'prior', 'n_chains', ...).

    Raises
    ------
    ImportError
        If CUQIpy is not installed.
    ValueError
        If ``sampler`` or ``prior`` is unknown, or an invalid combination is
        requested.
    """
    cuqi_mod = _load_cuqi()
    if not check_cuqi_available():
        raise ImportError(
            "CUQIpy is required for the CUQI-based unfolding methods. "
            "Install it with: pip install cuqipy"
        )

    sampler_l = str(sampler).lower()
    if sampler_l not in _VALID_SAMPLERS:
        raise ValueError(
            f"Unknown sampler {sampler!r}. Valid options: {_VALID_SAMPLERS}"
        )
    prior_l = str(prior).lower()
    if prior_l not in ("gmrf", "ou"):
        raise ValueError(f"Unknown prior {prior!r}. Valid options: 'gmrf', 'ou'")
    if hierarchical is None:
        hierarchical = sampler_l in _GIBBS_SAMPLERS
    if hierarchical and sampler_l not in _GIBBS_SAMPLERS:
        sampler_l = "gibbs"
    if hierarchical and prior_l != "gmrf":
        raise ValueError(
            "Hierarchical sampling (Gamma hyperprior on the precision) is "
            "only implemented for the 'gmrf' prior; set prior='gmrf' or "
            "use a non-hierarchical sampler."
        )
    if scale is None:
        scale = _DEFAULT_SCALES[sampler_l]

    A_matrix = np.asarray(A_matrix, dtype=float)
    b_readings = np.asarray(b_readings, dtype=float)
    n_detectors, n_energy = A_matrix.shape

    # Prior center on the log scale (data-driven NNLS or user supplied)
    mu = _prior_center_nnls(A_matrix, b_readings, initial_spectrum, n_energy)

    # Likelihood noise: relative scale sigma = noise_level * |b| per detector
    b_abs = np.abs(b_readings) + 1e-6
    sigma_vec = noise_level * b_abs
    sigma2_vec = sigma_vec**2

    chains = int(chains)
    n_samples_i = int(n_samples)
    thin_i = max(int(thin), 1)

    theta_per_chain_list: list[np.ndarray] = []
    delta_parts: list[np.ndarray] = []
    acc_rates: list[float] = []

    if hierarchical:
        for c in range(chains):
            seed_c = None if random_state is None else int(random_state) + c
            th, dl, ac = _run_gibbs_chain(
                sampler=sampler_l,
                A_matrix=A_matrix,
                b_readings=b_readings,
                sigma2=float(np.mean(sigma2_vec)),
                mu=mu,
                gmrf_order=int(gmrf_order),
                delta_alpha=float(delta_alpha),
                delta_beta=float(delta_beta),
                n_samples=n_samples_i,
                n_burnin=int(n_burnin),
                thin=thin_i,
                seed=seed_c,
                scale=float(scale),
                max_depth=max_depth,
                step_size=step_size,
                centered=(sampler_l == "gibbs"),
            )
            theta_per_chain_list.append(th)
            delta_parts.append(dl)
            acc_rates.append(ac)

        theta_all = np.vstack(theta_per_chain_list)
        delta_all = np.concatenate(delta_parts) if delta_parts else None
    elif sampler_l in _LANGEVIN_SAMPLERS:
        # Laplace-whitened UserDefined target: the whitened posterior is
        # approximately isotropic, which scalar-step Langevin samplers need
        # on the severely ill-conditioned unfolding problem.
        from cuqi.distribution import UserDefinedDistribution

        if prior_l == "gmrf":
            prior_precision = _gmrf_precision(n_energy, int(gmrf_order)) * float(prec)
        else:
            prior_precision = _ou_precision(n_energy, float(lengthscale)) * float(prec)

        t_map, whitening = _gauss_newton_map(
            A_matrix, b_readings, sigma2_vec, mu, prior_precision
        )
        # t_map is the ABSOLUTE log-spectrum MAP estimate (the Gauss-Newton
        # iteration starts from mu); the prior penalizes deviations from mu.
        center_theta = t_map

        def _logp(z):
            t = center_theta + whitening @ z
            resid = b_readings - A_matrix @ np.exp(t)
            prior_dev = t - mu
            return -0.5 * np.sum(resid**2 / sigma2_vec) - 0.5 * prior_dev @ prior_precision @ prior_dev

        def _grad(z):
            t = center_theta + whitening @ z
            resid = b_readings - A_matrix @ np.exp(t)
            grad_t = (
                np.exp(t) * (A_matrix.T @ (resid / sigma2_vec))
                - prior_precision @ (t - mu)
            )
            return whitening.T @ grad_t

        whitened_target = UserDefinedDistribution(
            dim=n_energy, logpdf_func=_logp, gradient_func=_grad
        )

        for c in range(chains):
            seed_c = None if random_state is None else int(random_state) + c
            z_samples, ac = _run_fixed_precision_chain(
                sampler=sampler_l,
                target=whitened_target,
                n_samples=n_samples_i,
                n_burnin=int(n_burnin),
                thin=thin_i,
                seed=seed_c,
                scale=float(scale),
                max_depth=max_depth,
                step_size=step_size,
                initial_point=np.zeros(n_energy),
            )
            # Map whitened draws back to the log-spectrum
            theta_per_chain_list.append(center_theta + z_samples @ whitening.T)
            acc_rates.append(ac)

        theta_all = np.vstack(theta_per_chain_list)
        delta_all = None
    else:
        # Fixed-precision native CUQIpy posterior.  PCN/CWMH sample the
        # *centered* log-spectrum t = theta - mu: the pCN proposal contracts
        # towards the prior mean, so centering is essential for the correct
        # proposal scaling.  NUTS samples theta directly.
        from cuqi.distribution import Gaussian, GMRF, Posterior

        centered = sampler_l in ("pcn", "cwmh")
        prior_mean = np.zeros(n_energy) if centered else mu
        if prior_l == "gmrf":
            theta_prior = GMRF(
                mean=prior_mean, prec=float(prec), bc_type="zero", order=int(gmrf_order)
            )
        else:
            corr = _ou_correlation(n_energy, float(lengthscale))
            theta_prior = Gaussian(mean=prior_mean, cov=corr / float(prec))

        forward = _build_forward_model(
            A_matrix, cuqi_mod, center=mu if centered else None
        )
        # NOTE: variable name 'y' defines the conditioning keyword below.
        y = Gaussian(mean=forward(theta_prior), cov=sigma2_vec)
        posterior = Posterior(y(y=b_readings), theta_prior)

        init = np.zeros(n_energy) if centered else mu

        for c in range(chains):
            seed_c = None if random_state is None else int(random_state) + c
            t_samples, ac = _run_fixed_precision_chain(
                sampler=sampler_l,
                target=posterior,
                n_samples=n_samples_i,
                n_burnin=int(n_burnin),
                thin=thin_i,
                seed=seed_c,
                scale=float(scale),
                max_depth=max_depth,
                step_size=step_size,
                initial_point=init,
            )
            theta_per_chain_list.append(t_samples + mu if centered else t_samples)
            acc_rates.append(ac)

        theta_all = np.vstack(theta_per_chain_list)
        delta_all = None

    acc_rate = float(np.nanmean(acc_rates)) if acc_rates else np.nan

    # Linear-scale spectrum samples and posterior statistics
    x_samples = np.exp(theta_all)
    mean_spectrum = np.mean(x_samples, axis=0)
    median_spectrum = np.median(x_samples, axis=0)
    std_spectrum = np.std(x_samples, axis=0)

    prob = float(credible_level) / 100.0
    if not 0 < prob < 1:
        raise ValueError(
            f"credible_level must be in (0, 100) percent, got {credible_level}"
        )
    hpd_lower, hpd_upper = _hpd_interval(x_samples, prob=prob)

    # Effective sample size on the log-scale chain (per energy bin)
    try:
        from cuqi.samples import Samples as _CuqiSamples

        ess = np.ravel(
            np.asarray(
                _CuqiSamples(theta_all.T, is_par=True, is_vec=True).compute_ess(),
                dtype=float,
            )
        )
    except Exception:
        ess = np.full(n_energy, np.nan)

    # Gelman-Rubin R-hat across chains (log scale), computed manually (same
    # version-robust philosophy as _hpd_interval in unfold_mcmc).
    rhat = None
    if chains > 1 and n_samples_i > 1:
        try:
            rhat = _gelman_rubin(np.stack(theta_per_chain(theta_all, chains, n_samples_i)))
        except Exception:
            rhat = None

    stats: dict[str, Any] = {
        "samples": x_samples,
        "theta_samples": theta_all,
        "mean": mean_spectrum,
        "median": median_spectrum,
        "std": std_spectrum,
        "hpd_lower": hpd_lower,
        "hpd_upper": hpd_upper,
        "ess": ess,
        "rhat": rhat,
        "acc_rate": acc_rate,
        "delta_samples": delta_all,
        "sampler": sampler_l,
        "prior": prior_l,
        "hierarchical": bool(hierarchical),
        "n_samples_total": int(theta_all.shape[0]),
        "n_chains": chains,
        "n_samples": n_samples_i,
        "n_burnin": int(n_burnin),
        "thin": thin_i,
        "noise_level": float(noise_level),
        "credible_level": float(credible_level),
        "gmrf_order": int(gmrf_order),
        "lengthscale": float(lengthscale),
        "prec": float(prec),
        "delta_alpha": float(delta_alpha),
        "delta_beta": float(delta_beta),
        "scale": float(scale),
        "prior_center": np.exp(mu),
        "cuqipy_backend": getattr(cuqi_mod, "__version__", "unknown"),
    }

    return mean_spectrum, stats


def unfold_cuqi(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    initial_spectrum: np.ndarray | None = None,
    sampler: str = "pcn",
    noise_level: float = 0.05,
    prior: str = "gmrf",
    gmrf_order: int = 1,
    lengthscale: float = 3.0,
    prec: float = 1.0,
    hierarchical: bool | None = None,
    delta_alpha: float = 1.0,
    delta_beta: float = 1e-4,
    n_samples: int = 2000,
    n_burnin: int = 1000,
    thin: int = 1,
    chains: int = 2,
    scale: float | None = None,
    max_depth: int = 8,
    step_size: float | None = None,
    credible_level: float = 95.0,
    calculate_errors: bool = False,
    mc_noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: int | None = None,
    progressbar: bool = False,
) -> dict[str, Any]:
    """Unfold neutron spectrum using CUQIpy Bayesian samplers.

    This is the workflow-level wrapper (same contract as
    :func:`bssunfold.core.unfold_mcmc.unfold_mcmc`) around
    :func:`solve_cuqi_bayesian`.  It builds the system matrix from the
    detector readings, runs the requested CUQIpy MCMC sampler(s) on the
    log-scale Bayesian model and returns the standardized unfolding result
    enriched with posterior samples, credible intervals and convergence
    diagnostics.

    Supported samplers: ``'pcn'``, ``'cwmh'``, ``'nuts'``, ``'mala'``,
    ``'ula'``, ``'gibbs'`` (hierarchical, data-driven smoothing) and
    ``'gibbs_nuts'``.

    Parameters
    ----------
    detector_names : List[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid in MeV.
    sensitivities : Dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : Dict[str, np.ndarray]
        ICRP-116 conversion coefficients for dose calculation.
    save_result_callback : callable
        Callback function to save result to history.
    readings : Dict[str, float]
        Detector readings (counts or count rates).
    initial_spectrum : Optional[np.ndarray], optional
        Prior center guess for the spectrum. When None, the non-negative
        least-squares solution of ``A @ x = b`` is used as the prior center.
    sampler : str, optional
        CUQIpy sampler (default: ``'pcn'``).
    noise_level : float, optional
        Relative likelihood noise scale (default: 0.05).
    prior : str, optional
        Log-spectrum prior: ``'gmrf'`` (default) or ``'ou'``.
    gmrf_order : int, optional
        GMRF operator order, 1 or 2 (default: 1).
    lengthscale : float, optional
        OU correlation length in energy bins (default: 3.0).
    prec : float, optional
        Fixed prior precision scale (default: 1.0); inferred from the data
        by the hierarchical Gibbs samplers.
    hierarchical : bool, optional
        Force the hierarchical Gibbs scheme (default: derived from sampler).
    delta_alpha : float, optional
        Gamma hyperprior shape (default: 1.0).
    delta_beta : float, optional
        Gamma hyperprior rate (default: 1e-4).
    n_samples : int, optional
        Posterior samples per chain (default: 2000).
    n_burnin : int, optional
        Warmup iterations per chain (default: 1000).  Increase for the
        Langevin samplers (``'mala'``/``'ula'``).
    thin : int, optional
        Thinning interval (default: 1).
    chains : int, optional
        Number of independent chains (default: 2).
    scale : float, optional
        Proposal step size (default: per-sampler ``_DEFAULT_SCALES``).
    max_depth : int, optional
        NUTS maximum tree depth (default: 8).
    step_size : float, optional
        NUTS leapfrog step size (default: None, tuned by CUQIpy).
    credible_level : float, optional
        Credible mass (%) of the HPD interval (default: 95).
    calculate_errors : bool, optional
        Calculate additional Monte-Carlo errors (default: False).
    mc_noise_level : float, optional
        Noise level for the additional Monte-Carlo loop (default: 0.01).
    n_montecarlo : int, optional
        Number of additional Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed for reproducibility.
    progressbar : bool, optional
        Present for API consistency (default: False).

    Returns
    -------
    Dict[str, Any]
        Standardized unfolding result with the usual keys (``energy``,
        ``spectrum``, ``effective_readings``, ``residual``, ``residual_norm``,
        ``method``, ``doserates``) plus ``spectrum_uncertainty``,
        ``spectrum_lower``, ``spectrum_upper`` and ``cuqi_stats`` (posterior
        samples, ESS, R-hat, acceptance rate, hyperparameter draws and
        sampling metadata).

    Raises
    ------
    ImportError
        If CUQIpy is not installed.
    RuntimeError
        If MCMC sampling fails.

    Examples
    --------
    >>> from bssunfold import Detector
    >>> detector = Detector()
    >>> result = detector.unfold_cuqi(
    ...     readings,
    ...     sampler='gibbs_nuts',
    ...     n_samples=1000,
    ...     n_burnin=500,
    ...     chains=2,
    ... )
    >>> spectrum = result['spectrum']
    >>> acc = result['cuqi_stats']['acc_rate']

    See Also
    --------
    unfold_mcmc : PyMC/NUTS Bayesian unfolding
    unfold_bayes : Bayesian iterative unfolding (D'Agostini)
    """
    if not check_cuqi_available():
        raise ImportError(
            "CUQIpy is required for the CUQI-based unfolding methods. "
            "Install it with: pip install cuqipy"
        )

    # The main solve is captured in a holder so its posterior statistics
    # (samples, diagnostics) can be merged into the standardized output.
    holder: dict[str, Any] = {}

    def _solve_cuqi(A, b, **kwargs):
        # NOTE: unlike iterative methods, the Bayesian prior center must NOT
        # fall back to run_unfolding's default ``x0 = ones``: when the user
        # provides no ``initial_spectrum`` the data-driven NNLS center is
        # used instead (see solve_cuqi_bayesian).
        mean_spectrum, stats = solve_cuqi_bayesian(
            A_matrix=A,
            b_readings=b,
            E=E_MeV,
            log_steps=np.ones(n_energy_bins),
            sampler=sampler,
            noise_level=noise_level,
            prior=prior,
            gmrf_order=gmrf_order,
            lengthscale=lengthscale,
            prec=prec,
            hierarchical=hierarchical,
            delta_alpha=delta_alpha,
            delta_beta=delta_beta,
            n_samples=n_samples,
            n_burnin=n_burnin,
            thin=thin,
            chains=chains,
            scale=scale,
            max_depth=max_depth,
            step_size=step_size,
            credible_level=credible_level,
            initial_spectrum=initial_spectrum,
            random_state=random_state,
            progressbar=progressbar,
        )
        holder.setdefault("stats", stats)
        return mean_spectrum

    result = run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=np.ones(n_energy_bins),
        solve_func=_solve_cuqi,
        solve_kwargs={},
        method_name="CUQI-Bayesian",
        extra_output={
            "sampler": sampler,
            "prior": prior,
            "n_samples": n_samples,
            "n_burnin": n_burnin,
            "chains": chains,
        },
        calculate_errors=calculate_errors,
        noise_level=mc_noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=False,
    )

    # Merge the posterior statistics into the standardized result *before*
    # it is handed to the history callback so saved results carry the full
    # output (pattern shared with unfold_mcmc).
    if "stats" in holder:
        stats = holder["stats"]
        result["spectrum_uncertainty"] = np.array(stats["std"])
        result["spectrum_lower"] = np.maximum(stats["hpd_lower"], 0)
        result["spectrum_upper"] = np.array(stats["hpd_upper"])
        result["cuqi_stats"] = dict(stats)

    if save_result and save_result_callback is not None:
        save_result_callback(result)

    return result
