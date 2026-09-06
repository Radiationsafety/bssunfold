"""Non-negative K-SVD dictionary learning unfolding for neutron spectrum reconstruction.

This module implements the two-step unfolding method from:

    Xu, Jing, Li et al., "Application of Non-negative K-SVD in Epithermal Neutron
    Spectrum Unfolding for BNCT", Nuclear Inst. and Methods in Physics Research, A
    (2026). https://doi.org/10.1016/j.nima.2026.172070

Key algorithms implemented
--------------------------
* **Non-negative K-SVD** (``solve_nksvd``): dictionary learning with non-negative
  truncation constraints imposed on dictionary atoms during the SVD update stage.
  This ensures all atoms remain element-wise non-negative throughout training,
  matching the physical prior that neutron spectra are non-negative.

* **NNLS+TopK** (``solve_nnls_topk``): hierarchical sparse coding strategy.
  Step 1 — solve global NNLS for all atoms; Step 2 — select the K atoms with
  the largest coefficients; Step 3 — re-solve NNLS restricted to the selected
  atoms. This avoids the cumulative selection error of OMP and achieves
  substantially better reconstruction (correlation ~0.97 vs ~0.55 for OMP
  as reported in the paper).

* **NN-OMP** (``solve_nn_omp``): Non-negative Orthogonal Matching Pursuit.
  Same greedy atom selection as OMP, but each least-squares step is replaced
  with NNLS so coefficients never go negative.

* **Tikhonov-augmented NNLS** (``solve_tikhonov_nnls``): solves
  ``min_{alpha>=0} ||y - M_norm alpha||^2 + lambda_tik ||alpha||^2``
  via the augmented formulation of Eq. (2.6), yielding a standard NNLS problem
  that can be solved directly with ``scipy.optimize.nnls``.

* **Full unfolding pipeline** (``solve_nksvd_unfold``): combines non-negative
  K-SVD dictionary learning, normalized equivalent dictionary construction
  (Eq. 2.4), Tikhonov-augmented NNLS, and NNLS+TopK sparse inversion to
  reconstruct the neutron spectrum from detector measurements.

Sparse coding strategies
-----------------------
Three sparse coding algorithms are available for the inversion step, selectable
via the ``sparse_method`` parameter of ``solve_nksvd_unfold``:

* ``"nnls_topk"`` (default, recommended) — the NNLS+TopK hierarchical strategy.
* ``"nn_omp"`` — Non-negative OMP greedy algorithm.
* ``"omp"`` — standard OMP (provided for comparison; no non-negativity guarantee).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import nnls

from ..utils.validators import validate_system
from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = [
    "solve_nksvd",
    "solve_nnls_topk",
    "solve_nn_omp",
    "solve_omp_standard",
    "solve_tikhonov_nnls",
    "solve_nksvd_unfold",
    "unfold_nksvd",
]


# ---------------------------------------------------------------------------
# Normalized equivalent dictionary  (Eq. 2.4)
# ---------------------------------------------------------------------------
def _build_normalized_equivalent_dictionary(
    R: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """Construct the normalized equivalent detection dictionary.

    Given the detector-response matrix *R* (M x N) and the dictionary *D*
    (N x P), the detection model under sparse representation is
    ``y = (R @ D) @ alpha``  (Eq. 2.3).  The equivalent detection dictionary
    is::

        M     = R @ D           (M x P)
        M_norm[:, k] = M[:, k] / ||M[:, k]||_2

    This follows Eqs. (2.3)–(2.4) of the reference paper.

    Parameters
    ----------
    R : np.ndarray
        Detector-response matrix (M x N).
    D : np.ndarray
        Dictionary matrix (N x P).

    Returns
    -------
    np.ndarray
        Normalized equivalent dictionary (M x P).
    """
    M = R @ D  # (M x P)
    norms = np.linalg.norm(M, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    M_norm = M / norms
    return M_norm


# ---------------------------------------------------------------------------
# Tikhonov-augmented NNLS  (Eq. 2.5 / 2.6)
# ---------------------------------------------------------------------------
def solve_tikhonov_nnls(
    M_norm: np.ndarray,
    y: np.ndarray,
    lambda_tik: float = 0.01,
) -> np.ndarray:
    """Solve Tikhonov-regularized NNLS via augmented formulation.

    Solves::

        min_{alpha >= 0}  ||y - M_norm alpha||^2  +  lambda_tik ||alpha||^2

    by constructing the augmented system of Eq. (2.6)::

        A_tilde = [M_norm; sqrt(lambda_tik) I_P]
        b_tilde = [y; 0_P]

    and calling ``scipy.optimize.nnls(A_tilde, b_tilde)``.

    Parameters
    ----------
    M_norm : np.ndarray
        Normalized equivalent dictionary (M x P).
    y : np.ndarray
        Measured-count vector (M,).
    lambda_tik : float, optional
        Tikhonov smoothing-regularization weight (default: 0.01, per paper).

    Returns
    -------
    np.ndarray
        Non-negative coefficient vector (P,).
    """
    P = M_norm.shape[1]
    sqrt_lam = np.sqrt(lambda_tik)

    # Augmented matrix: (M+P) x P
    A_tilde = np.vstack([M_norm, sqrt_lam * np.eye(P)])
    # Augmented RHS: (M+P,)
    b_tilde = np.concatenate([y, np.zeros(P)])

    alpha, _ = nnls(A_tilde, b_tilde)
    return alpha


# ---------------------------------------------------------------------------
# NNLS+TopK sparse coding  (hierarchical strategy)
# ---------------------------------------------------------------------------
def solve_nnls_topk(
    M_norm: np.ndarray,
    y: np.ndarray,
    sparsity: int,
    lambda_tik: float = 0.01,
) -> np.ndarray:
    """Sparse coding via NNLS + Top-K atom screening + local NNLS refinement.

    This is the three-step hierarchical strategy proposed in the paper:

    1. **Global NNLS**: solve Tikhonov-augmented NNLS over all P atoms to
       obtain a full coefficient vector.
    2. **Top-K selection**: retain the K atoms with the largest coefficients.
    3. **Local NNLS**: re-solve Tikhonov-augmented NNLS restricted to the
       selected K atoms for refined coefficients.

    This avoids the cumulative selection error of OMP and achieves
    significantly better reconstruction accuracy.

    Parameters
    ----------
    M_norm : np.ndarray
        Normalized equivalent dictionary (M x P).
    y : np.ndarray
        Measured-count vector (M,).
    sparsity : int
        Target sparsity K (number of non-zero coefficients).
    lambda_tik : float, optional
        Tikhonov regularization weight (default: 0.01).

    Returns
    -------
    np.ndarray
        Sparse non-negative coefficient vector (P,).
    """
    P = M_norm.shape[1]
    K = min(sparsity, P)

    # Step 1: Global NNLS over all atoms
    alpha_full = solve_tikhonov_nnls(M_norm, y, lambda_tik=lambda_tik)

    # Step 2: Select Top-K atoms by largest coefficients
    topk_idx = np.argsort(alpha_full)[-K:]

    # Step 3: Local NNLS restricted to selected atoms
    M_norm_sub = M_norm[:, topk_idx]
    alpha_sub = solve_tikhonov_nnls(M_norm_sub, y, lambda_tik=lambda_tik)

    # Assemble the sparse vector
    alpha = np.zeros(P)
    alpha[topk_idx] = alpha_sub

    return alpha


# ---------------------------------------------------------------------------
# Non-negative OMP  (NN-OMP)
# ---------------------------------------------------------------------------
def solve_nn_omp(
    M_norm: np.ndarray,
    y: np.ndarray,
    sparsity: int,
    lambda_tik: float = 0.01,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Sparse coding using Non-negative OMP (NN-OMP).

    Like standard OMP, atoms are greedily selected by maximum absolute
    correlation with the residual.  However, at each step the coefficients
    on the active support are found by NNLS (with Tikhonov regularization)
    instead of ordinary least-squares, ensuring non-negativity.

    Parameters
    ----------
    M_norm : np.ndarray
        Normalized equivalent dictionary (M x P).
    y : np.ndarray
        Measured-count vector (M,).
    sparsity : int
        Maximum number of non-zero coefficients.
    lambda_tik : float, optional
        Tikhonov weight for NNLS sub-problems (default: 0.01).
    tolerance : float, optional
        Residual tolerance for early stopping (default: 1e-6).

    Returns
    -------
    np.ndarray
        Sparse non-negative coefficient vector (P,).
    """
    M, P = M_norm.shape
    K = min(sparsity, P)
    alpha = np.zeros(P)
    residual = y.copy()
    support: list = []

    # Pre-compute column norms for correlation computation
    norms = np.linalg.norm(M_norm, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    D_norm = M_norm / norms

    for _ in range(K):
        # Greedy atom selection by absolute correlation
        correlations = np.abs(D_norm.T @ residual)
        correlations[support] = -1.0  # exclude already selected
        idx = int(np.argmax(correlations))
        if correlations[idx] <= 0:
            break
        support.append(idx)

        # NNLS on the active support (with Tikhonov)
        M_sub = M_norm[:, support]
        sqrt_lam = np.sqrt(lambda_tik)
        A_tilde = np.vstack([M_sub, sqrt_lam * np.eye(len(support))])
        b_tilde = np.concatenate([y, np.zeros(len(support))])
        coefs, _ = nnls(A_tilde, b_tilde)

        # Update residual
        residual = y - M_sub @ coefs

        if np.linalg.norm(residual) < tolerance * (np.linalg.norm(y) + 1e-12):
            break

    # Final NNLS on the selected support
    if support:
        M_sub = M_norm[:, support]
        sqrt_lam = np.sqrt(lambda_tik)
        A_tilde = np.vstack([M_sub, sqrt_lam * np.eye(len(support))])
        b_tilde = np.concatenate([y, np.zeros(len(support))])
        coefs, _ = nnls(A_tilde, b_tilde)
        alpha[support] = coefs

    return alpha


# ---------------------------------------------------------------------------
# Standard OMP (for comparison)
# ---------------------------------------------------------------------------
def solve_omp_standard(
    M_norm: np.ndarray,
    y: np.ndarray,
    sparsity: int,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Standard OMP sparse coding (no non-negativity constraint).

    Provided for benchmarking against NNLS+TopK and NN-OMP.  Identical
    to ``solve_omp`` in ``unfold_cs`` but operates on the normalized
    equivalent dictionary.

    Parameters
    ----------
    M_norm : np.ndarray
        Normalized equivalent dictionary (M x P).
    y : np.ndarray
        Measured-count vector (M,).
    sparsity : int
        Maximum number of non-zero coefficients.
    tolerance : float, optional
        Residual tolerance for early stopping (default: 1e-6).

    Returns
    -------
    np.ndarray
        Sparse coefficient vector (P,).  May contain negative values.
    """
    _, P = M_norm.shape
    K = min(sparsity, P)
    alpha = np.zeros(P)
    residual = y.copy()
    support: list = []

    norms = np.linalg.norm(M_norm, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    D_norm = M_norm / norms

    for _ in range(K):
        correlations = np.abs(D_norm.T @ residual)
        correlations[support] = -1.0
        idx = int(np.argmax(correlations))
        if correlations[idx] <= 0:
            break
        support.append(idx)

        D_s = M_norm[:, support]
        coefs, *_ = np.linalg.lstsq(D_s, y, rcond=None)
        residual = y - D_s @ coefs

        if np.linalg.norm(residual) < tolerance * (np.linalg.norm(y) + 1e-12):
            break

    if support:
        D_s = M_norm[:, support]
        coefs, *_ = np.linalg.lstsq(D_s, y, rcond=None)
        alpha[support] = coefs

    return alpha


# ---------------------------------------------------------------------------
# Non-negative K-SVD dictionary learning
# ---------------------------------------------------------------------------
def solve_nksvd(
    signals: np.ndarray,
    n_atoms: int,
    n_iterations: int = 80,
    sparsity: int = 2,
    lambda_tik: float = 0.01,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Learn a non-negative dictionary using the Non-negative K-SVD algorithm.

    This is the dual-improvement K-SVD proposed in the reference paper:

    1. **Dictionary update**: after the SVD-based atom update, non-negative
       truncation is applied — any negative entries in the updated atom are
       set to zero.  This enforces the physical prior that all dictionary
       atoms (basis-spectrum patterns) are element-wise non-negative.

    2. **Sparse coding**: NNLS+TopK is used instead of standard OMP,
       ensuring both dictionary atoms and sparse coefficients remain
       non-negative throughout all iterations.

    Parameters
    ----------
    signals : np.ndarray
        Training signals (n x m), one column per training sample.
        Must be element-wise non-negative.
    n_atoms : int
        Number of dictionary atoms (P).
    n_iterations : int, optional
        Maximum number of K-SVD iterations (default: 80, per paper).
    sparsity : int, optional
        Target sparsity for sparse coding (default: 2, per paper optimal).
    lambda_tik : float, optional
        Tikhonov regularization weight for NNLS+TopK (default: 0.01).
    random_state : int, optional
        Random seed for reproducibility (default: 42 per paper).

    Returns
    -------
    np.ndarray
        Learned non-negative dictionary (n x n_atoms).  All entries >= 0.
    """
    if random_state is None:
        random_state = 42
    rng = np.random.default_rng(random_state)
    n, m = signals.shape
    n_atoms = min(n_atoms, m)

    # Initialize dictionary with random training samples (non-negative)
    idx = rng.choice(m, size=n_atoms, replace=False)
    D = signals[:, idx].copy()
    # Apply non-negative truncation on initialization
    D = np.maximum(D, 0.0)

    # Normalize columns
    norms = np.linalg.norm(D, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    D = D / norms

    for iteration in range(n_iterations):
        # --- Sparse coding stage (NNLS+TopK for each signal) ---
        coefficients = np.zeros((n_atoms, m))
        for j in range(m):
            # Each signal is used as the "measurement" y, and D as the
            # dictionary.  Since we are in signal-space (not measurement-space),
            # M_norm = D (column-normalized).
            col_norms = np.linalg.norm(D, axis=0)
            col_norms = np.where(col_norms == 0, 1.0, col_norms)
            D_norm = D / col_norms
            coefficients[:, j] = solve_nnls_topk(
                D_norm, signals[:, j], sparsity=sparsity, lambda_tik=lambda_tik
            )

        # --- Dictionary update stage (atom-by-atom SVD + non-negative truncation) ---
        for atom in range(n_atoms):
            # Find signals that use this atom
            used = np.where(coefficients[atom, :] != 0)[0]
            if len(used) == 0:
                continue

            # Compute the error matrix excluding this atom
            D_restricted = D.copy()
            D_restricted[:, atom] = 0.0
            E = signals[:, used] - D_restricted @ coefficients[:, used]

            # SVD of the error matrix
            U, s, Vt = np.linalg.svd(E, full_matrices=False)

            # Non-negative truncation on the updated atom
            new_atom = np.maximum(U[:, 0], 0.0)
            atom_norm = np.linalg.norm(new_atom)
            if atom_norm > 0:
                new_atom = new_atom / atom_norm
            else:
                # If truncation zeroed the atom, keep the old one
                new_atom = D[:, atom]

            new_coef = s[0] * Vt[0, :]

            # Update atom and coefficients
            D[:, atom] = new_atom
            coefficients[atom, used] = new_coef

        # Re-normalize columns (and enforce non-negativity after float drift)
        D = np.maximum(D, 0.0)
        norms = np.linalg.norm(D, axis=0)
        norms = np.where(norms == 0, 1.0, norms)
        D = D / norms

    return D


# ---------------------------------------------------------------------------
# Full unfolding pipeline  (solve_nksvd_unfold)
# ---------------------------------------------------------------------------
def solve_nksvd_unfold(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    n_atoms: int = 15,
    sparsity: int = 2,
    dictionary: Optional[np.ndarray] = None,
    training_signals: Optional[np.ndarray] = None,
    n_dictionary_iterations: int = 80,
    lambda_tik: float = 0.01,
    sparse_method: str = "nnls_topk",
    tolerance: float = 1e-6,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, int, bool]:
    """Solve the unfolding problem using Non-negative K-SVD + sparse inversion.

    Implements the two-step method from the reference paper:

    **Step 1 — Dictionary learning**:  Non-negative K-SVD adaptively learns
    a sparse-representation dictionary from training spectra (either provided
    via ``training_signals`` or synthesized from the response matrix and
    initial guess).

    **Step 2 — Sparse inversion**:  Given measurements *b*, response matrix
    *A*, and learned dictionary *D*:

    1. Construct the normalized equivalent dictionary *M_norm* (Eq. 2.4).
    2. Solve the Tikhonov-augmented NNLS problem (Eq. 2.5/2.6) using the
       chosen sparse coding strategy (NNLS+TopK, NN-OMP, or OMP).
    3. Reconstruct the spectrum:  phi = D @ alpha.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (M x N).
    b : np.ndarray
        Measurement vector (M,).
    x0 : np.ndarray, optional
        Initial guess (N,).  Used to seed training signals if
        ``training_signals`` is not provided.
    n_atoms : int, optional
        Number of dictionary atoms (default: 15, per paper optimum).
    sparsity : int, optional
        Target sparsity K (default: 2, per paper optimum).
    dictionary : np.ndarray, optional
        Pre-learned dictionary (N x n_atoms).  If provided, dictionary
        learning is skipped.
    training_signals : np.ndarray, optional
        Training signal matrix (N x m) for dictionary learning.
        If not provided, synthetic training signals are generated from
        the response matrix and initial guess.
    n_dictionary_iterations : int, optional
        Max iterations for K-SVD (default: 80, per paper).
    lambda_tik : float, optional
        Tikhonov regularization weight (default: 0.01, per paper).
    sparse_method : str, optional
        Sparse coding strategy: ``"nnls_topk"`` (default), ``"nn_omp"``,
        or ``"omp"``.
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).
    random_state : int, optional
        Random seed (default: 42, per paper).

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        (reconstructed_spectrum, iterations, converged)
    """
    # Validate inputs
    A, b, x0 = validate_system(A, b, x0=x0)
    m, n = A.shape

    if random_state is None:
        random_state = 42

    # Validate sparse_method early
    valid_methods = {"nnls_topk", "nn_omp", "omp"}
    if sparse_method not in valid_methods:
        raise ValueError(
            f"Unknown sparse_method '{sparse_method}'. "
            f"Choose from {sorted(valid_methods)}."
        )

    # --- Dictionary learning (or use provided dictionary) ---
    if dictionary is not None:
        D = np.asarray(dictionary, dtype=float)
        D = np.maximum(D, 0.0)  # enforce non-negativity
        if D.shape[0] != n:
            raise ValueError(
                f"Dictionary first dimension ({D.shape[0]}) must match "
                f"number of energy bins ({n})"
            )
    else:
        # Build training signals
        if training_signals is not None:
            signals = np.asarray(training_signals, dtype=float)
            signals = np.maximum(signals, 0.0)
        else:
            signals = _build_training_signals(A, b, x0, n, m)

        D = solve_nksvd(
            signals,
            n_atoms=n_atoms,
            n_iterations=n_dictionary_iterations,
            sparsity=sparsity,
            lambda_tik=lambda_tik,
            random_state=random_state,
        )

    # --- Normalized equivalent dictionary (Eq. 2.4) ---
    M_norm = _build_normalized_equivalent_dictionary(A, D)

    # --- Sparse inversion ---
    if sparse_method == "nnls_topk":
        alpha = solve_nnls_topk(M_norm, b, sparsity=sparsity, lambda_tik=lambda_tik)
    elif sparse_method == "nn_omp":
        alpha = solve_nn_omp(M_norm, b, sparsity=sparsity, lambda_tik=lambda_tik)
    elif sparse_method == "omp":
        alpha = solve_omp_standard(M_norm, b, sparsity=sparsity)
    else:
        # Already validated above; this is defensive
        raise ValueError(
            f"Unknown sparse_method '{sparse_method}'. "
            f"Choose from {sorted(valid_methods)}."
        )

    # --- Reconstruct spectrum ---
    x = D @ alpha
    x = np.maximum(x, 0.0)  # enforce non-negativity

    # Scale normalization: match total measurement magnitude
    computed = A @ x
    if np.linalg.norm(computed) > 0 and np.linalg.norm(b) > 0:
        scale = np.dot(b, computed) / (np.dot(computed, computed) + 1e-12)
        scale = max(scale, 0.0)  # non-negative scale
        x = x * scale

    residual = np.linalg.norm(A @ x - b)
    converged = bool(residual < tolerance * max(1.0, np.linalg.norm(b)))
    iterations = n_dictionary_iterations

    return x, iterations, converged


def _build_training_signals(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray],
    n: int,
    m: int,
) -> np.ndarray:
    """Build synthetic training signals for dictionary learning.

    Uses the initial guess plus smooth cosine basis vectors to create
    a diverse training set that covers the typical shape space of
    neutron spectra.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray or None
        Initial spectrum guess (n,).
    n : int
        Number of energy bins.
    m : int
        Number of detectors.

    Returns
    -------
    np.ndarray
        Training signals (n x n_basis), all non-negative.
    """
    # Base signal from initial guess
    if x0 is not None and np.any(x0 > 0):
        base = np.maximum(x0, 0.0)
        norm = np.linalg.norm(base)
        if norm > 0:
            base = base / norm
        else:
            base = np.ones(n) / np.sqrt(n)
    else:
        base = np.ones(n) / np.sqrt(n)

    # Smooth basis vectors (cosine-like, non-negative via abs)
    t = np.linspace(0.0, np.pi, n)
    n_basis = min(n, max(2 * m, 8))
    signals = np.zeros((n, n_basis + 1))
    for i in range(n_basis):
        # Use abs(cos) to ensure non-negativity of training signals
        signals[:, i] = np.abs(np.cos(i * t))
        norm = np.linalg.norm(signals[:, i])
        if norm > 0:
            signals[:, i] /= norm
    signals[:, -1] = base

    return signals


# ---------------------------------------------------------------------------
# Evaluation metrics  (Eqs. 2.7, 2.8, 2.9)
# ---------------------------------------------------------------------------
def compute_flux_error(phi_true: np.ndarray, phi_recon: np.ndarray) -> float:
    """Compute relative flux error (Eq. 2.7).

    Parameters
    ----------
    phi_true : np.ndarray
        True energy spectrum.
    phi_recon : np.ndarray
        Reconstructed spectrum.

    Returns
    -------
    float
        Relative flux error in [0, inf).  Lower is better.
    """
    norm_true = np.linalg.norm(phi_true)
    if norm_true == 0:
        return 0.0 if np.linalg.norm(phi_recon) == 0 else np.inf
    return float(np.linalg.norm(phi_true - phi_recon) / norm_true)


def compute_spectral_correlation(phi_true: np.ndarray, phi_recon: np.ndarray) -> float:
    """Compute Pearson correlation coefficient for spectral shape (Eq. 2.8).

    Parameters
    ----------
    phi_true : np.ndarray
        True energy spectrum.
    phi_recon : np.ndarray
        Reconstructed spectrum.

    Returns
    -------
    float
        Correlation coefficient in [-1, 1].  Closer to 1 is better.
    """
    if np.std(phi_true) == 0 or np.std(phi_recon) == 0:
        return 0.0
    corr = np.corrcoef(phi_true, phi_recon)[0, 1]
    return float(corr)


def compute_comprehensive_score(phi_true: np.ndarray, phi_recon: np.ndarray) -> float:
    """Compute comprehensive-score index (Eq. 2.9).

    ``score = flux_err - 0.5 * flux_corr``

    Lower is better.

    Parameters
    ----------
    phi_true : np.ndarray
        True energy spectrum.
    phi_recon : np.ndarray
        Reconstructed spectrum.

    Returns
    -------
    float
        Comprehensive score.  Lower is better.
    """
    flux_err = compute_flux_error(phi_true, phi_recon)
    flux_corr = compute_spectral_correlation(phi_true, phi_recon)
    return flux_err - 0.5 * flux_corr


# ---------------------------------------------------------------------------
# unfold_nksvd wrapper
# ---------------------------------------------------------------------------
def unfold_nksvd(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    n_atoms: int = 15,
    sparsity: int = 2,
    dictionary: Optional[np.ndarray] = None,
    training_signals: Optional[np.ndarray] = None,
    n_dictionary_iterations: int = 80,
    lambda_tik: float = 0.01,
    sparse_method: str = "nnls_topk",
    tolerance: float = 1e-6,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using Non-negative K-SVD dictionary learning.

    Implements the two-step unfolding method of Xu et al. (2026): non-negative
    K-SVD dictionary learning followed by Tikhonov-regularized sparse
    inversion.

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
    initial_spectrum : np.ndarray, optional
        Initial spectrum guess.
    n_atoms : int, optional
        Number of dictionary atoms (default: 15).
    sparsity : int, optional
        Target sparsity (default: 2).
    dictionary : np.ndarray, optional
        Pre-learned dictionary (n x n_atoms).
    training_signals : np.ndarray, optional
        Training signals for dictionary learning (n x m).
    n_dictionary_iterations : int, optional
        Max K-SVD iterations (default: 80).
    lambda_tik : float, optional
        Tikhonov regularization weight (default: 0.01).
    sparse_method : str, optional
        Sparse coding strategy: ``"nnls_topk"`` (default),
        ``"nn_omp"``, or ``"omp"``.
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for Monte-Carlo (default: 0.01).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: 100).
    save_result : bool, optional
        Save result to history (default: False).
    random_state : int, optional
        Random seed (default: 42).

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
            solve_nksvd_unfold,
            n_atoms=n_atoms,
            sparsity=sparsity,
            dictionary=dictionary,
            training_signals=training_signals,
            n_dictionary_iterations=n_dictionary_iterations,
            lambda_tik=lambda_tik,
            sparse_method=sparse_method,
            tolerance=tolerance,
            random_state=random_state,
        ),
        solve_kwargs={},
        method_name="NonNegativeKSVD",
        extra_output={},
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )
