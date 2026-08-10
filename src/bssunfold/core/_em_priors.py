"""Shared nearest-neighbour priors for penalized EM unfolding methods.

Port of the PyTomography nearest-neighbour priors (``QuadraticPrior``,
``LogCoshPrior``, ``RelativeDifferencePrior``) adapted to a one-dimensional
energy-bin spectrum. The prior reads

    V(f) = beta * sum_r sum_{s in NN(r)} w_{r,s} phi0(f_r, f_s)

and the gradient used by the one-step-late EM updates is

    grad_r V(f) = beta * sum_{s in NN(r)} w_{r,s} phi1(f_r, f_s)

where the nearest neighbours ``s`` are taken along the energy axis with unit
weights ``w_{r,s} = 1`` (zero-padded at the boundaries), mirroring the
Euclidean neighbour weight in one dimension.
"""

import numpy as np

__all__ = ["prior_gradient", "prior_value"]


def _neighbours(x: np.ndarray) -> tuple:
    """Return the (left, right) nearest-neighbour arrays, zero-padded."""
    n = x.shape[0]
    left = np.zeros(n)
    right = np.zeros(n)
    if n > 1:
        left[1:] = x[:-1]
        right[:-1] = x[1:]
    return left, right


def _phi1(fr: np.ndarray, fs: np.ndarray, prior: str, delta: float,
          gamma: float) -> np.ndarray:
    """First derivative phi1(f_r, f_s) = d/d f_r phi0(f_r, f_s)."""
    if prior == "quadratic":
        return (fr - fs) / delta
    if prior == "logcosh":
        return np.tanh((fr - fs) / delta)
    if prior == "relative_difference":
        absd = np.abs(fr - fs)
        denom = gamma * absd + fr + fs + delta
        return (fr - fs) * (gamma * absd + 3.0 * fs + fr + 2.0 * delta) / denom ** 2
    raise ValueError(
        f"Unknown prior {prior!r}. Choose from 'quadratic', 'logcosh', "
        f"'relative_difference'."
    )


def _phi0(fr: np.ndarray, fs: np.ndarray, prior: str, delta: float,
          gamma: float) -> np.ndarray:
    """Prior pair potential phi0(f_r, f_s)."""
    if prior == "quadratic":
        return 0.25 * ((fr - fs) / delta) ** 2
    if prior == "logcosh":
        t = (fr - fs) / delta
        t_abs = np.abs(t)
        return t_abs + np.log1p(np.exp(-2.0 * t_abs)) - np.log(2.0)
    if prior == "relative_difference":
        absd = np.abs(fr - fs)
        return (fr - fs) ** 2 / (fr + fs + gamma * absd + delta)
    raise ValueError(
        f"Unknown prior {prior!r}. Choose from 'quadratic', 'logcosh', "
        f"'relative_difference'."
    )


def prior_gradient(x: np.ndarray, prior: str = "quadratic", beta: float = 1e-3,
                   delta: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """Compute the beta-scaled prior gradient along the energy axis.

    Parameters
    ----------
    x : np.ndarray
        Current spectrum estimate (n,).
    prior : str, optional
        Prior type: ``'quadratic'``, ``'logcosh'`` or
        ``'relative_difference'`` (default: ``'quadratic'``).
    beta : float, optional
        Prior weight (default: 1e-3).
    delta : float, optional
        Width parameter used by the quadratic/logcosh priors and as the
        additive floor of the relative-difference prior (default: 1.0).
    gamma : float, optional
        Edge-preservation parameter of the relative-difference prior
        (default: 1.0).

    Returns
    -------
    np.ndarray
        Prior gradient evaluated at ``x`` (n,).
    """
    prior = str(prior).lower()
    x = np.asarray(x, dtype=float)
    left, right = _neighbours(x)
    grad = _phi1(x, left, prior, delta, gamma) + _phi1(x, right, prior, delta, gamma)
    return beta * grad


def prior_value(x: np.ndarray, prior: str = "quadratic", beta: float = 1e-3,
                delta: float = 1.0, gamma: float = 1.0) -> float:
    """Compute the (beta-scaled) nearest-neighbour prior value V(f).

    Parameters
    ----------
    x : np.ndarray
        Current spectrum estimate (n,).
    prior : str, optional
        Prior type (default: ``'quadratic'``).
    beta : float, optional
        Prior weight (default: 1e-3).
    delta : float, optional
        Width parameter (default: 1.0).
    gamma : float, optional
        Edge-preservation parameter of the relative-difference prior
        (default: 1.0).

    Returns
    -------
    float
        Prior value at ``x``.
    """
    prior = str(prior).lower()
    x = np.asarray(x, dtype=float)
    left, right = _neighbours(x)
    value = _phi0(x, left, prior, delta, gamma) + _phi0(x, right, prior, delta, gamma)
    return float(beta * np.sum(value))
