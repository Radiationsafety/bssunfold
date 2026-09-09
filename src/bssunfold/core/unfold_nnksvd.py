"""Non-negative K-SVD unfolding for epithermal neutron spectrum recovery.

This module implements the unfolding method described in

    H.-l. Xu, S.-W. Jing, Z. Li, J.Y. Sun, Y. Gu, J. Hujia, S. Liu, G. Qu,
    "Application of Non-negative K-SVD in Epithermal Neutron Spectrum
    Unfolding for BNCT", Nuclear Instruments and Methods in Physics
    Research A (2026), https://doi.org/10.1016/j.nima.2026.172070

The article proposes a two-step unfolding pipeline:

1. **Non-negative K-SVD dictionary learning** -- K-SVD with non-negative
   truncation of dictionary atoms during the update stage.  This converts
   the high-dimensional, ill-posed, underdetermined neutron spectrum
   inversion problem into a low-dimensional sparse-reconstruction problem
   over a learned non-negative dictionary ``D``.
2. **Sparse inversion** -- three sparse coding strategies are supported:

   * ``nnls_topk`` (the article's proposed method): global NNLS coarse
     solution, top-``K`` atom screening, then local NNLS fine
     optimization on the screened support.
   * ``omp`` -- the classic Orthogonal Matching Pursuit greedy algorithm.
   * ``nn_omp`` -- OMP with a non-negativity constraint on the inner
     least-squares step (solved as NNLS on the selected support).

The reconstruction objective is a Tikhonov-regularized non-negative
least-squares (Eq. 2.5 of the article), solved via the augmented-matrix
form (Eq. 2.6) so that any off-the-shelf NNLS solver can be used directly::

    min  || y - M_norm @ alpha ||^2 + lambda_tik * || alpha ||^2
    s.t. alpha >= 0

A training-sample-driven prior constraint (``prior_wt``) further pulls the
recovered coefficients toward a prior coefficient pattern derived from
the dictionary-training stage (mean sparse code of the training signals).

The three evaluation metrics introduced in Section 2.2.3 of the article
(relative flux error, Pearson correlation coefficient for spectral shape,
and the comprehensive-score index) live in
:mod:`bssunfold.utils.comparison` and are re-exported here for
convenience.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import nnls

from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = [
    "solve_tikhonov_nnls",
    "solve_nn_omp",
    "solve_nnls_topk",
    "solve_nnksvd",
    "solve_nnksvd_unfold",
    "unfold_nnksvd",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _normalize_dictionary(D: np.ndarray) -> np.ndarray:
    """L2-normalize each column of ``D`` in-place safe (returns a copy)."""
    D = np.asarray(D, dtype=float).copy()
    norms = np.linalg.norm(D, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    return D / norms


def _nnls(A: np.ndarray, b: np.ndarray, max_iter: int | None = None) -> np.ndarray:
    """Solve ``min || A x - b ||`` s.t. ``x >= 0`` via scipy.optimize.nnls.

    Returns a 1-D array shaped like ``A.shape[1]``.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    if A.shape[1] == 0:
        return np.zeros(A.shape[1])
    if max_iter is None:
        x, _ = nnls(A, b)
    else:
        # ``scipy.optimize.nnls`` accepts ``maxiter`` only from scipy >= 1.7.
        try:
            x, _ = nnls(A, b, maxiter=max_iter)
        except TypeError:
            x, _ = nnls(A, b)
    return np.asarray(x, dtype=float)


def _build_equivalent_dictionary(
    R: np.ndarray, D: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Build the equivalent (normalized) detection dictionary ``M_norm``.

    ``M = R @ D`` is the effective measurement matrix mapping the sparse
    coefficient vector ``alpha`` to the detector counts ``y``; each column
    is L2-normalized to remove amplitude differences between dictionary
    atoms (Eq. 2.4 of the article).  ``R`` is the (m x n) detector
    response matrix, ``D`` is the (n x p) dictionary.

    The article writes the equivalent dictionary as ``M = R^T D``;
    dimensionally this only makes sense if ``R`` is interpreted as the
    transpose of the response matrix (i.e., energy-channels x detector
    channels).  We use the mathematically-correct ``M = R @ D`` so that
    ``M @ alpha`` has shape ``(m,)`` matching the measurement vector
    ``y``.
    """
    M = np.asarray(R, dtype=float) @ np.asarray(D, dtype=float)
    if normalize:
        M = _normalize_dictionary(M)
    return M


# ---------------------------------------------------------------------------
# Tikhonov-regularized NNLS via augmented form (Eq. 2.5 / 2.6)
# ---------------------------------------------------------------------------
def solve_tikhonov_nnls(
    M_norm: np.ndarray,
    y: np.ndarray,
    lambda_tik: float = 0.01,
    prior_wt: float = 0.0,
    alpha_prior: np.ndarray | None = None,
    max_iter: int | None = None,
) -> np.ndarray:
    """Tikhonov-regularized Non-Negative Least Squares.

    Solves Eq. (2.5) of the article

        min  || y - M_norm @ alpha ||^2 + lambda_tik * || alpha ||^2
        s.t. alpha >= 0

    via the augmented-matrix equivalent (Eq. 2.6)

        min  || [y; 0] - [M_norm; sqrt(lambda_tik) I] @ alpha ||^2
        s.t. alpha >= 0

    An additional training-sample-driven prior constraint
    ``prior_wt * || alpha - alpha_prior ||^2`` (article Section 2.2.2,
    "training sample-driven prior constraints") is appended to the
    augmented system when ``prior_wt > 0`` and ``alpha_prior`` is given.

    Parameters
    ----------
    M_norm : np.ndarray
        Normalized equivalent detection dictionary (m x p).
    y : np.ndarray
        Measurement / count vector (m,).
    lambda_tik : float, optional
        Tikhonov smoothing regularization weight (default: 0.01, as in
        the article).
    prior_wt : float, optional
        Training-sample-driven prior weight (default: 0.0).  When > 0,
        ``alpha_prior`` must be supplied.
    alpha_prior : np.ndarray, optional
        Prior coefficient vector (p,).  Required when ``prior_wt > 0``.
    max_iter : int, optional
        Maximum NNLS iterations.

    Returns
    -------
    np.ndarray
        Non-negative sparse coefficient vector (p,).
    """
    M_norm = np.asarray(M_norm, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    p = M_norm.shape[1]

    # Augmented matrix and RHS (Eq. 2.6).
    A_aug = np.vstack([M_norm, np.sqrt(max(lambda_tik, 0.0)) * np.eye(p)])
    b_aug = np.concatenate([y, np.zeros(p)])

    if prior_wt > 0.0:
        if alpha_prior is None:
            raise ValueError("alpha_prior must be provided when prior_wt > 0")
        alpha_prior = np.asarray(alpha_prior, dtype=float).ravel()
        if alpha_prior.shape[0] != p:
            raise ValueError(
                f"alpha_prior length ({alpha_prior.shape[0]}) must match "
                f"number of dictionary atoms ({p})"
            )
        # || alpha - alpha_prior ||^2  ==  || (alpha - alpha_prior) ||^2
        # Reformulate as an augmented residual: 0 - sqrt(prior_wt) * I @ alpha
        # = -sqrt(prior_wt) * alpha_prior, with augmented matrix row
        # sqrt(prior_wt) * I.
        A_aug = np.vstack([A_aug, np.sqrt(prior_wt) * np.eye(p)])
        b_aug = np.concatenate([b_aug, np.sqrt(prior_wt) * alpha_prior])

    return _nnls(A_aug, b_aug, max_iter=max_iter)


# ---------------------------------------------------------------------------
# Non-negative OMP (NN-OMP) sparse coding
# ---------------------------------------------------------------------------
def solve_nn_omp(
    D: np.ndarray,
    y: np.ndarray,
    sparsity: int,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Non-negative Orthogonal Matching Pursuit (NN-OMP).

    Greedy sparse coding with a non-negativity constraint on the
    coefficient vector.  At each iteration:

    1. Pick the atom whose (signed) projection onto the residual is the
       *largest positive* value -- non-negativity forbids negative
       coefficients so we only accept positively-correlated atoms.
    2. Solve NNLS on the selected support (instead of unconstrained LS).
    3. Update the residual.

    Parameters
    ----------
    D : np.ndarray
        Dictionary matrix (n x p).
    y : np.ndarray
        Signal to be represented (n,).
    sparsity : int
        Maximum number of non-zero coefficients (K in the article).
    tolerance : float, optional
        Early-stopping residual tolerance (default: 1e-6).

    Returns
    -------
    np.ndarray
        Non-negative sparse coefficient vector (p,).
    """
    D = np.asarray(D, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = D.shape
    alpha = np.zeros(p)
    residual = y.copy()
    support: list[int] = []

    # Normalize dictionary columns for atom selection (does not affect NNLS).
    norms = np.linalg.norm(D, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    D_norm = D / norms

    for _ in range(min(sparsity, p)):
        # Signed correlation: pick the largest *positive* projection
        # (negative correlations would require negative coefficients).
        correlations = D_norm.T @ residual
        correlations[support] = -np.inf  # exclude already-selected atoms
        idx = int(np.argmax(correlations))
        if correlations[idx] <= 0 or not np.isfinite(correlations[idx]):
            break
        support.append(idx)

        # Solve NNLS on the current support.
        D_s = D[:, support]
        coefs = _nnls(D_s, y)
        alpha[support] = coefs
        residual = y - D_s @ coefs

        if np.linalg.norm(residual) < tolerance:
            break

    return alpha


# ---------------------------------------------------------------------------
# NNLS+TopK sparse coding (the article's proposed strategy)
# ---------------------------------------------------------------------------
def solve_nnls_topk(
    M_norm: np.ndarray,
    y: np.ndarray,
    sparsity: int,
    lambda_tik: float = 0.01,
    prior_wt: float = 0.0,
    alpha_prior: np.ndarray | None = None,
    max_iter: int | None = None,
) -> np.ndarray:
    """NNLS+TopK sparse coding strategy (Xu et al. 2026, proposed method).

    Three-stage hierarchical sparse coding:

    1. **Global NNLS coarse solution**: solve the Tikhonov-NNLS problem
       on the full dictionary.
    2. **Top-K atom screening**: keep the ``K`` atoms with the largest
       coefficients from the coarse solution.
    3. **Local NNLS fine optimization**: re-solve NNLS on the screened
       support for a refined, sparse, non-negative coefficient vector.

    The hierarchical strategy avoids the cumulative selection error of
    greedy OMP-style algorithms and produces sparse, physically
    meaningful solutions.

    Parameters
    ----------
    M_norm : np.ndarray
        Normalized equivalent detection dictionary (m x p).
    y : np.ndarray
        Measurement / count vector (m,).
    sparsity : int
        Target sparsity ``K`` (number of non-zero coefficients).
    lambda_tik : float, optional
        Tikhonov smoothing weight (default: 0.01).
    prior_wt : float, optional
        Training-sample-driven prior weight (default: 0.0).
    alpha_prior : np.ndarray, optional
        Prior coefficient vector (p,). Required when ``prior_wt > 0``.
    max_iter : int, optional
        Maximum NNLS iterations.

    Returns
    -------
    np.ndarray
        Non-negative K-sparse coefficient vector (p,).
    """
    M_norm = np.asarray(M_norm, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    p = M_norm.shape[1]
    K = max(0, min(int(sparsity), p))

    if K == 0:
        return np.zeros(p)

    # Step 1: global NNLS coarse solution.
    alpha_full = solve_tikhonov_nnls(
        M_norm,
        y,
        lambda_tik=lambda_tik,
        prior_wt=prior_wt,
        alpha_prior=alpha_prior,
        max_iter=max_iter,
    )

    # Step 2: top-K atom screening.
    if K >= p:
        return alpha_full
    topk_idx = np.argsort(alpha_full)[-K:]

    # Step 3: local NNLS fine optimization on the screened support.
    M_topk = M_norm[:, topk_idx]
    if prior_wt > 0.0 and alpha_prior is not None:
        alpha_prior_topk = np.asarray(alpha_prior, dtype=float).ravel()[topk_idx]
    else:
        alpha_prior_topk = None
    alpha_topk = solve_tikhonov_nnls(
        M_topk,
        y,
        lambda_tik=lambda_tik,
        prior_wt=prior_wt,
        alpha_prior=alpha_prior_topk,
        max_iter=max_iter,
    )

    alpha = np.zeros(p)
    alpha[topk_idx] = alpha_topk
    return alpha


# ---------------------------------------------------------------------------
# Non-negative K-SVD dictionary learning
# ---------------------------------------------------------------------------
def solve_nnksvd(
    signals: np.ndarray,
    n_atoms: int,
    n_iterations: int = 80,
    sparsity: int = 2,
    lambda_tik: float = 0.01,
    prior_wt: float = 0.5,
    sparse_coder: str = "nnls_topk",
    random_state: int | None = None,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative K-SVD dictionary learning.

    K-SVD variant with non-negativity constraints on both dictionary
    atoms and sparse coefficients, following Xu et al. (2026).  The
    dictionary update stage applies non-negative truncation
    (``max(0, atom)``) after the SVD rank-1 update of each atom, then
    re-normalizes.  The sparse coding stage uses one of the three
    strategies supported by this module (``nnls_topk``, ``omp`` or
    ``nn_omp``).

    Parameters
    ----------
    signals : np.ndarray
        Training signals (n x m), one column per training sample.  Must
        be non-negative (neutron spectra are physically non-negative).
    n_atoms : int
        Number of dictionary atoms ``P``.
    n_iterations : int, optional
        Maximum K-SVD iterations (default: 80, as in the article).
    sparsity : int, optional
        Target sparsity ``K`` for sparse coding (default: 2).
    lambda_tik : float, optional
        Tikhonov weight for the NNLS+TopK sparse coder (default: 0.01).
    prior_wt : float, optional
        Prior weight for the NNLS+TopK sparse coder (default: 0.5).
    sparse_coder : str, optional
        Sparse-coding strategy: ``"nnls_topk"``, ``"omp"`` or
        ``"nn_omp"`` (default: ``"nnls_topk"``).
    random_state : int, optional
        Random seed for reproducibility.
    tolerance : float, optional
        Early-stopping tolerance on dictionary change.

    Returns
    -------
    D : np.ndarray
        Learned non-negative dictionary (n x p), columns L2-normalized.
    alpha_prior : np.ndarray
        Mean sparse code of the training signals (p,), used as the
        training-sample-driven prior during unfolding.
    """
    if sparse_coder not in ("nnls_topk", "omp", "nn_omp"):
        raise ValueError(
            f"Unknown sparse_coder '{sparse_coder}'. "
            "Expected 'nnls_topk', 'omp' or 'nn_omp'."
        )

    signals = np.asarray(signals, dtype=float)
    n, m = signals.shape
    # Enforce non-negativity on training signals (physical prior).
    signals = np.maximum(signals, 0.0)

    rng = np.random.default_rng(random_state)

    n_atoms = max(1, min(int(n_atoms), m))
    # Initialize dictionary with random training samples (non-negative).
    idx = rng.choice(m, size=n_atoms, replace=False)
    D = signals[:, idx].copy()
    # If a chosen training sample is all zeros, replace with a smooth bump.
    zero_cols = np.where(np.linalg.norm(D, axis=0) == 0)[0]
    for zc in zero_cols:
        bump = np.maximum(0.0, rng.normal(size=n))
        if np.linalg.norm(bump) == 0:
            bump = np.ones(n)
        D[:, zc] = bump
    D = _normalize_dictionary(D)

    coefficients = np.zeros((n_atoms, m))

    for _ in range(int(n_iterations)):
        D_prev = D.copy()

        # Sparse coding stage (per training sample).
        for j in range(m):
            y_j = signals[:, j]
            if sparse_coder == "omp":
                from .unfold_cs import solve_omp

                coefficients[:, j] = solve_omp(D, y_j, sparsity=sparsity)
            elif sparse_coder == "nn_omp":
                coefficients[:, j] = solve_nn_omp(
                    D, y_j, sparsity=sparsity, tolerance=tolerance
                )
            else:  # nnls_topk
                # NNLS+TopK operates on the dictionary directly (no response
                # matrix here, so M_norm = D normalized = D).
                coefficients[:, j] = solve_nnls_topk(
                    D,
                    y_j,
                    sparsity=sparsity,
                    lambda_tik=lambda_tik,
                    prior_wt=0.0,  # prior not yet available during training
                )

        # Dictionary update stage.
        for atom in range(n_atoms):
            used = np.where(coefficients[atom, :] != 0)[0]
            if len(used) == 0:
                # Re-initialize unused atom with a random training sample.
                j_new = int(rng.integers(0, m))
                new_atom = signals[:, j_new].copy()
                new_atom = np.maximum(new_atom, 0.0)
                norm = np.linalg.norm(new_atom)
                if norm == 0:
                    new_atom = np.ones(n)
                    norm = np.sqrt(n)
                D[:, atom] = new_atom / norm
                continue

            # Error matrix after removing the current atom's contribution.
            D_restricted = D.copy()
            D_restricted[:, atom] = 0.0
            E = signals[:, used] - D_restricted @ coefficients[:, used]

            # Rank-1 SVD approximation of E (classic K-SVD step).
            U, s, Vt = np.linalg.svd(E, full_matrices=False)
            new_atom = U[:, 0]
            new_coef = s[0] * Vt[0, :]

            # Non-negative truncation (article's key modification).
            new_atom = np.maximum(new_atom, 0.0)
            # Also enforce non-negative coefficients for this atom.
            new_coef = np.maximum(new_coef, 0.0)

            # Re-normalize the atom; rescale coefficients to preserve product.
            norm = np.linalg.norm(new_atom)
            if norm > 0:
                # Combine norm into coefficients so D @ alpha stays invariant.
                scale = norm
                D[:, atom] = new_atom / norm
                coefficients[atom, used] = new_coef * scale
            else:
                # Atom collapsed to zero; reinitialize from a training sample.
                j_new = int(rng.integers(0, m))
                new_atom = np.maximum(signals[:, j_new], 0.0)
                norm = np.linalg.norm(new_atom)
                if norm == 0:
                    new_atom = np.ones(n)
                    norm = np.sqrt(n)
                D[:, atom] = new_atom / norm
                # Recompute coefficient for this atom via NNLS per signal
                # (NNLS only accepts a single RHS vector).
                for u_idx in used:
                    coef_u = _nnls(D[:, [atom]], signals[:, u_idx])
                    coefficients[atom, u_idx] = coef_u[0]

        # Convergence check on the dictionary change.
        if np.linalg.norm(D - D_prev) < tolerance * max(1.0, np.linalg.norm(D_prev)):
            break

    # Final non-negative safeguard and re-normalization.
    D = np.maximum(D, 0.0)
    D = _normalize_dictionary(D)

    # Training-sample-driven prior: mean sparse code across training signals.
    alpha_prior = np.mean(coefficients, axis=1)

    return D, alpha_prior


# ---------------------------------------------------------------------------
# Top-level NN-KSVD unfolding solver
# ---------------------------------------------------------------------------
def solve_nnksvd_unfold(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    n_atoms: int = 15,
    sparsity: int = 2,
    dictionary: np.ndarray | None = None,
    training_signals: np.ndarray | None = None,
    n_dictionary_iterations: int = 80,
    lambda_tik: float = 0.01,
    prior_wt: float = 0.5,
    sparse_coder: str = "nnls_topk",
    random_state: int | None = None,
    tolerance: float = 1e-6,
    n_nnls_iter: int | None = None,
    E_MeV: np.ndarray | None = None,
) -> tuple[np.ndarray, int, bool]:
    """Unfold a neutron spectrum using the non-negative K-SVD pipeline.

    Two operating modes:

    * **Pre-learned dictionary** (``dictionary`` provided): the
      dictionary is used as-is; ``training_signals`` is ignored.
    * **Online dictionary learning** (``training_signals`` provided or
      ``x0`` used to synthesize them): the dictionary is learned on the
      fly with :func:`solve_nnksvd`.

    The forward model is ``y = A @ phi + eps`` and the spectrum is
    represented as ``phi = D @ alpha`` where ``D`` is the learned
    non-negative dictionary and ``alpha`` is a non-negative sparse
    coefficient vector.  Sparse coding on the equivalent detection
    dictionary ``M_norm = normalize(A @ D)`` is performed with the
    selected sparse-coding strategy (NNLS+TopK, OMP, or NN-OMP).

    Parameters
    ----------
    A : np.ndarray
        Detector response matrix (m x n).
    b : np.ndarray
        Measurement / count vector (m,).
    x0 : np.ndarray, optional
        Initial spectrum guess (n,).  Used to seed training signals when
        no ``training_signals`` is supplied.
    n_atoms : int, optional
        Number of dictionary atoms (default: 15, the article's optimum).
    sparsity : int, optional
        Target sparsity K (default: 2, the article's optimum).
    dictionary : np.ndarray, optional
        Pre-learned non-negative dictionary (n x p).  Bypasses online
        K-SVD training.
    training_signals : np.ndarray, optional
        Training signals for online K-SVD (n x m).  If not provided,
        log-spaced Gaussian bumps on the energy grid plus the initial
        guess are used.
    n_dictionary_iterations : int, optional
        K-SVD iterations (default: 80, as in the article).
    lambda_tik : float, optional
        Tikhonov regularization weight (default: 0.01).
    prior_wt : float, optional
        Training-sample prior weight (default: 0.5).
    sparse_coder : str, optional
        Sparse-coding strategy (default: ``"nnls_topk"``).
    random_state : int, optional
        Random seed for reproducibility (default: 42 in the article).
    tolerance : float, optional
        Convergence tolerance for K-SVD and OMP-style coders.
    n_nnls_iter : int, optional
        Maximum NNLS iterations (passed through to scipy.optimize.nnls).
    E_MeV : np.ndarray, optional
        Energy grid in MeV (n,).  When provided, the default training
        signals are log-spaced Gaussian bumps on the log-energy grid,
        which ensures the dictionary covers the full spectral range.
        Falls back to a uniform normalised index when not supplied.

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple ``(spectrum, iterations, converged)``.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape

    if random_state is None:
        random_state = 42

    # ── Dictionary acquisition ─────────────────────────────────────
    if dictionary is not None:
        D = np.asarray(dictionary, dtype=float)
        if D.shape[0] != n:
            raise ValueError(
                f"Dictionary first dimension ({D.shape[0]}) must match "
                f"the number of energy bins ({n})."
            )
        D = np.maximum(D, 0.0)
        D = _normalize_dictionary(D)
        alpha_prior = None
    else:
        # Build training signals for online K-SVD.
        if training_signals is not None:
            signals = np.asarray(training_signals, dtype=float)
            if signals.shape[0] != n:
                raise ValueError(
                    f"Training signals first dimension ({signals.shape[0]})"
                    f" must match the number of energy bins ({n})."
                )
            signals = np.maximum(signals, 0.0)
        else:
            # Synthesize training signals for online K-SVD.
            if x0 is not None and np.any(x0):
                base = np.maximum(x0, 0)
                norm = np.linalg.norm(base)
                if norm > 0:
                    base = base / norm
                else:
                    base = np.ones(n) / np.sqrt(n)
            else:
                base = np.ones(n) / np.sqrt(n)

            # Build non-negative training signals that cover the full
            # spectral range.  When E_MeV is supplied we place log-spaced
            # Gaussian bumps on the log-energy grid so that the learned
            # dictionary atoms span the entire energy domain (thermal
            # through fast).  Without E_MeV we fall back to a uniform
            # normalised index.
            n_basis = max(n_atoms * 2, 8)
            n_basis = min(n_basis, n)

            if E_MeV is not None and np.any(E_MeV > 0):
                log_E = np.log10(np.maximum(np.asarray(E_MeV, dtype=float), 1e-15))
                centers = np.linspace(log_E[0], log_E[-1], n_basis)
                width = (log_E[-1] - log_E[0]) / max(n_basis * 1.5, 1.0)
                signals = np.zeros((n, n_basis + 1))
                for i in range(n_basis):
                    col = np.exp(-((log_E - centers[i]) ** 2) / (2 * width ** 2))
                    norm = np.linalg.norm(col)
                    if norm > 0:
                        col = col / norm
                    signals[:, i] = col
            else:
                t = np.linspace(0.0, 1.0, n)
                signals = np.zeros((n, n_basis + 1))
                for i in range(n_basis):
                    center = (i + 1) / (n_basis + 1)
                    col = np.exp(-((t - center) ** 2) / (2 * (1.0 / n_basis) ** 2))
                    norm = np.linalg.norm(col)
                    if norm > 0:
                        col = col / norm
                    signals[:, i] = col

            signals[:, -1] = base

        D, alpha_prior = solve_nnksvd(
            signals,
            n_atoms=n_atoms,
            n_iterations=n_dictionary_iterations,
            sparsity=sparsity,
            lambda_tik=lambda_tik,
            prior_wt=prior_wt,
            sparse_coder=sparse_coder,
            random_state=random_state,
            tolerance=tolerance,
        )

    # ── Sparse inversion on the equivalent detection dictionary ────
    M = _build_equivalent_dictionary(A, D, normalize=True)

    # If a pre-learned dictionary is supplied (no online K-SVD), the
    # training-sample prior is unavailable; disable it silently.
    effective_prior_wt = prior_wt if alpha_prior is not None else 0.0

    if sparse_coder == "nnls_topk":
        alpha = solve_nnls_topk(
            M,
            b,
            sparsity=sparsity,
            lambda_tik=lambda_tik,
            prior_wt=effective_prior_wt,
            alpha_prior=alpha_prior,
            max_iter=n_nnls_iter,
        )
    elif sparse_coder == "omp":
        from .unfold_cs import solve_omp

        alpha = solve_omp(M, b, sparsity=sparsity, tolerance=tolerance)
    elif sparse_coder == "nn_omp":
        alpha = solve_nn_omp(M, b, sparsity=sparsity, tolerance=tolerance)
    else:
        raise ValueError(
            f"Unknown sparse_coder '{sparse_coder}'. "
            "Expected 'nnls_topk', 'omp' or 'nn_omp'."
        )

    # ── Spectrum reconstruction ───────────────────────────────────
    # phi = D @ alpha; account for the column-normalization of M:
    # if M[:, k] = (A @ D[:, k]) / ||A @ D[:, k]||, then the recovered
    # alpha is expressed in the normalized basis.  We undo the
    # normalization so that A @ phi ~= b.
    phi = D @ alpha
    # Compensation for M-normalization: rescale each atom's contribution
    # by ||A @ D[:, k]|| (which is the norm removed during normalization).
    unnorm = A @ D
    atom_norms = np.linalg.norm(unnorm, axis=0)
    atom_norms = np.where(atom_norms == 0, 1.0, atom_norms)
    phi = D @ (alpha * atom_norms)

    # Physical non-negativity safeguard.
    phi = np.maximum(phi, 0.0)

    # Optional scale alignment: make the predicted counts match the
    # measured counts in total magnitude.  This mirrors the scale-fix
    # used in ``unfold_cs.solve_cs``.
    computed = A @ phi
    if np.linalg.norm(computed) > 0 and np.linalg.norm(b) > 0:
        scale = float(np.dot(b, computed) / (np.dot(computed, computed) + 1e-12))
        if scale > 0:
            phi = phi * scale

    residual = float(np.linalg.norm(A @ phi - b))
    converged = bool(residual < tolerance * max(1.0, np.linalg.norm(b)))

    return phi, int(n_dictionary_iterations), converged


# ---------------------------------------------------------------------------
# Detector-level wrapper (for use with the Detector class)
# ---------------------------------------------------------------------------
def unfold_nnksvd(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback: Callable[[dict[str, Any]], str],
    readings: dict[str, float],
    initial_spectrum: np.ndarray | None = None,
    n_atoms: int = 15,
    sparsity: int = 2,
    dictionary: np.ndarray | None = None,
    training_signals: np.ndarray | None = None,
    n_dictionary_iterations: int = 80,
    lambda_tik: float = 0.01,
    prior_wt: float = 0.5,
    sparse_coder: str = "nnls_topk",
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: int | None = None,
    tolerance: float = 1e-6,
    n_nnls_iter: int | None = None,
) -> dict[str, Any]:
    """Detector-level wrapper for the non-negative K-SVD unfolding method.

    See :func:`solve_nnksvd_unfold` for the algorithmic details; this
    function adapts the solver to the
    :func:`bssunfold.core._base_unfolder.run_unfolding` workflow so that
    it integrates transparently with :class:`bssunfold.Detector`.

    Parameters
    ----------
    detector_names : List[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins.
    E_MeV : np.ndarray
        Energy grid (MeV).
    sensitivities : Dict[str, np.ndarray]
        Detector sensitivity arrays.
    cc_icrp116 : Dict[str, np.ndarray]
        ICRP-116 conversion coefficients.
    save_result_callback : callable
        Callback used to save the result to history.
    readings : Dict[str, float]
        Detector readings.
    initial_spectrum : np.ndarray, optional
        Initial spectrum guess.
    n_atoms : int, optional
        Number of dictionary atoms (default: 15).
    sparsity : int, optional
        Target sparsity K (default: 2).
    dictionary : np.ndarray, optional
        Pre-learned non-negative dictionary (n x p).
    training_signals : np.ndarray, optional
        Training signals for online K-SVD (n x m).  If not provided,
        log-spaced Gaussian bumps on the energy grid are used.
    n_dictionary_iterations : int, optional
        K-SVD iterations (default: 80).
    lambda_tik : float, optional
        Tikhonov regularization weight (default: 0.01).
    prior_wt : float, optional
        Training-sample-driven prior weight (default: 0.5).
    sparse_coder : str, optional
        Sparse-coding strategy (default: ``"nnls_topk"``).
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
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).
    n_nnls_iter : int, optional
        Maximum NNLS iterations.
    E_MeV : np.ndarray, optional
        Energy grid in MeV.  Passed through to
        :func:`solve_nnksvd_unfold` for log-spaced Gaussian training
        signal generation.  When ``None``, the energy grid stored in
        the ``E_MeV`` parameter of this function (the detector grid)
        is used.

    Returns
    -------
    Dict[str, Any]
        Unfolding results dictionary.
    """
    x0_default = np.zeros(n_energy_bins)

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
            solve_nnksvd_unfold,
            n_atoms=n_atoms,
            sparsity=sparsity,
            dictionary=dictionary,
            training_signals=training_signals,
            n_dictionary_iterations=n_dictionary_iterations,
            lambda_tik=lambda_tik,
            prior_wt=prior_wt,
            sparse_coder=sparse_coder,
            tolerance=tolerance,
            n_nnls_iter=n_nnls_iter,
            E_MeV=E_MeV,
        ),
        solve_kwargs={},
        method_name="NNKSVD",
        extra_output={
            "n_atoms": n_atoms,
            "sparsity": sparsity,
            "sparse_coder": sparse_coder,
            "lambda_tik": lambda_tik,
            "prior_wt": prior_wt,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
