"""Conditioning diagnostics for unfolding response matrices.

This module provides the public conditioning_diagnostics function that
quantifies how sensitive an unfolding problem ``A x = b`` is to noise in
the readings ``b``, plus helpers to integrate the diagnostics into the
standardized unfolding result dictionary.

The conditioning of the response matrix is a pure property of the
detector set (it does not depend on the readings), so it can be
inspected before unfolding via Detector.conditioning_report.
"""

import warnings
from typing import Any

import numpy as np

__all__ = ["conditioning_diagnostics", "conditioning_summary", "ILL_THRESHOLD"]

#: Default threshold on cond_2(A) above which the unfolding problem is
#: flagged as ill-conditioned (loss of ~half of the double-precision
#: digits squared in the normal equations).
ILL_THRESHOLD: float = 1e10


def conditioning_diagnostics(
    A: np.ndarray,
    rcond: float = 1e-12,
    ill_threshold: float = ILL_THRESHOLD,
) -> dict[str, Any]:
    """Compute conditioning diagnostics of the response matrix.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    rcond : float, optional
        Relative singular-value cutoff used for the effective rank
        (default: 1e-12), i.e. singular values below ``rcond * s.max()``
        are treated as numerically zero.
    ill_threshold : float, optional
        Threshold on ``cond_2`` above which the problem is flagged as
        ill-conditioned (default: 1e10).

    Returns
    -------
    Dict[str, Any]
        Diagnostics with keys:

        - ``cond_2`` : float
          Spectral condition number ``sigma_max / sigma_min`` of A.
        - ``cond_normal`` : float
          Condition number of the normal equations A^T A, equal to
          ``cond_2**2``.
        - ``singular_values`` : np.ndarray
          Singular values of A in descending order.
        - ``effective_rank`` : int
          Number of singular values above ``rcond * s.max()``.
        - ``rank_deficient`` : bool
          True when ``effective_rank < min(m, n)``.
        - ``ill_conditioned`` : bool
          True when ``cond_2 > ill_threshold`` (or the matrix is
          rank-deficient).
        - ``shape`` : tuple[int, int]
          Shape of the response matrix.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-dimensional, got ndim={A.ndim}")

    s = np.linalg.svd(A, compute_uv=False)
    s_max = float(s[0]) if s.size else 0.0
    s_min = float(s[-1]) if s.size else 0.0

    if s_max == 0.0 or s_min == 0.0:
        # Zero (or numerically zero) smallest singular value: the matrix
        # is rank-deficient and the condition number is infinite.
        cond_2 = float("inf")
    else:
        cond_2 = s_max / s_min

    cond_normal = cond_2 * cond_2

    if s.size and s_max > 0.0:
        effective_rank = int(np.sum(s > rcond * s_max))
    else:
        effective_rank = 0

    rank_deficient = effective_rank < min(A.shape)
    ill_conditioned = bool(
        (not np.isfinite(cond_2)) or cond_2 > ill_threshold or rank_deficient
    )

    return {
        "cond_2": float(cond_2),
        "cond_normal": float(cond_normal),
        "singular_values": s,
        "effective_rank": effective_rank,
        "rank_deficient": bool(rank_deficient),
        "ill_conditioned": ill_conditioned,
        "shape": tuple(A.shape),
    }


def conditioning_summary(diag: dict[str, Any]) -> str:
    """Render a short human-readable summary of conditioning diagnostics.

    Parameters
    ----------
    diag : Dict[str, Any]
        Result of :func:`conditioning_diagnostics`.

    Returns
    -------
    str
        One-line summary suitable for logs and reports.
    """
    cond_2 = diag["cond_2"]
    if not np.isfinite(cond_2):
        cond_str = "inf"
    else:
        cond_str = f"{cond_2:.3e}"
    flags = []
    if diag.get("rank_deficient"):
        flags.append("rank-deficient")
    if diag.get("ill_conditioned"):
        flags.append("ill-conditioned")
    flag_str = f" [{', '.join(flags)}]" if flags else ""
    m, n = diag.get("shape", (float("nan"), float("nan")))
    return (
        f"cond_2(A)={cond_str}, cond(A^T A)={cond_str}^2, "
        f"effective_rank={diag.get('effective_rank')}/{min(m, n)}{flag_str}"
    )


def _attach_conditioning(
    output: dict[str, Any],
    A: np.ndarray,
    warn: bool = True,
) -> dict[str, Any]:
    """Compute diagnostics, store them in output and warn if unreliable.

    This is the internal hook used by _base_unfolder._standardize_output
    to attach diagnostics to every unfolding result.

    Parameters
    ----------
    output : Dict[str, Any]
        Standardized unfolding result dictionary (modified in place).
    A : np.ndarray
        Response matrix used for the unfolding.
    warn : bool, optional
        Emit a UserWarning when the problem is ill-conditioned or
        rank-deficient (default: True).

    Returns
    -------
    Dict[str, Any]
        The same output dictionary (for chaining).
    """
    diag = conditioning_diagnostics(A)
    output["conditioning"] = diag
    if warn and diag["ill_conditioned"]:
        warnings.warn(
            "Response matrix is ill-conditioned or rank-deficient "
            f"({conditioning_summary(diag)}); unfolding results may be "
            "unreliable. Consider reducing the number of energy bins or "
            "changing the detector set.",
            UserWarning,
            stacklevel=3,
        )
    return output
