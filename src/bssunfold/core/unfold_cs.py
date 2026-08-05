"""Compressive Sensing (CS) unfolding method for neutron spectrum reconstruction.

This module implements a neutron spectrum unfolding method based on compressive
sensing (CS), following the approach described in the literature on BSS
(Bonner Sphere Spectrometer) unfolding:

* Sparse representation based on a learning dictionary is employed.
* The Orthogonal Matching Pursuit (OMP) algorithm is used for sparse coding.
* The K-SVD algorithm is used for dictionary update.
* The SL0 (Smoothed L0) algorithm is used for reconstruction.

The key idea is that the neutron spectrum ``x`` can be represented sparsely in a
learned dictionary ``D`` as ``x = D @ alpha``, where ``alpha`` is a sparse
coefficient vector. The measurement equation ``b = A @ x`` then becomes
``b = (A @ D) @ alpha``, which is solved for the sparse ``alpha`` using SL0.
Finally the spectrum is reconstructed as ``x = D @ alpha``.

This approach is particularly well suited for the highly underdetermined
problem where the number of energy groups (e.g. 300) greatly exceeds the number
of detector readings (e.g. 7).
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding, make_solve_wrapper

__all__ = [
    "solve_omp",
    "solve_ksvd",
    "solve_sl0",
    "solve_cs",
    "unfold_cs",
]


# ---------------------------------------------------------------------------
# Orthogonal Matching Pursuit (OMP)
# ---------------------------------------------------------------------------
def solve_omp(
    D: np.ndarray,
    y: np.ndarray,
    sparsity: int,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Solve sparse coding problem using Orthogonal Matching Pursuit (OMP).

    Finds a sparse coefficient vector ``alpha`` such that ``y ~= D @ alpha``
    with at most ``sparsity`` non-zero entries.

    Parameters
    ----------
    D : np.ndarray
        Dictionary matrix (n x k).
    y : np.ndarray
        Signal to be represented (n,).
    sparsity : int
        Maximum number of non-zero coefficients.
    tolerance : float, optional
        Residual tolerance for early stopping (default: 1e-6).

    Returns
    -------
    np.ndarray
        Sparse coefficient vector (k,).
    """
    n, k = D.shape
    alpha = np.zeros(k)
    residual = y.copy()
    support = []

    # Normalize dictionary columns for atom selection
    norms = np.linalg.norm(D, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    D_norm = D / norms

    for _ in range(min(sparsity, k)):
        # Select the atom most correlated with the residual
        correlations = np.abs(D_norm.T @ residual)
        # Exclude already selected atoms
        correlations[support] = -1.0
        idx = int(np.argmax(correlations))
        if correlations[idx] <= 0:
            break
        support.append(idx)

        # Least-squares solve on the support
        D_s = D[:, support]
        coefs, *_ = np.linalg.lstsq(D_s, y, rcond=None)
        residual = y - D_s @ coefs

        if np.linalg.norm(residual) < tolerance:
            break

    if support:
        D_s = D[:, support]
        coefs, *_ = np.linalg.lstsq(D_s, y, rcond=None)
        alpha[support] = coefs

    return alpha


# ---------------------------------------------------------------------------
# K-SVD dictionary learning
# ---------------------------------------------------------------------------
def solve_ksvd(
    signals: np.ndarray,
    n_atoms: int,
    n_iterations: int = 20,
    sparsity: int = 5,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Learn a dictionary using the K-SVD algorithm.

    Parameters
    ----------
    signals : np.ndarray
        Training signals (n x m), one column per training sample.
    n_atoms : int
        Number of dictionary atoms.
    n_iterations : int, optional
        Number of K-SVD iterations (default: 20).
    sparsity : int, optional
        Target sparsity for sparse coding (default: 5).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Learned dictionary (n x n_atoms).
    """
    rng = np.random.default_rng(random_state)
    n, m = signals.shape

    # Initialize dictionary with random training samples
    n_atoms = min(n_atoms, m)
    idx = rng.choice(m, size=n_atoms, replace=False)
    D = signals[:, idx].copy()

    # Normalize columns
    norms = np.linalg.norm(D, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    D = D / norms

    for _ in range(n_iterations):
        # Sparse coding stage (OMP for each signal)
        coefficients = np.zeros((n_atoms, m))
        for j in range(m):
            coefficients[:, j] = solve_omp(D, signals[:, j], sparsity)

        # Dictionary update stage
        for atom in range(n_atoms):
            # Find signals that use this atom
            used = np.where(coefficients[atom, :] != 0)[0]
            if len(used) == 0:
                continue

            # Restrict to used signals and their coefficients
            D_restricted = D.copy()
            D_restricted[:, atom] = 0.0
            E = signals[:, used] - D_restricted @ coefficients[:, used]

            # SVD of the error matrix restricted to the atom's row
            U, s, Vt = np.linalg.svd(E, full_matrices=False)
            new_atom = U[:, 0]
            new_coef = s[0] * Vt[0, :]

            # Update atom and coefficients
            D[:, atom] = new_atom
            coefficients[atom, used] = new_coef

        # Re-normalize columns
        norms = np.linalg.norm(D, axis=0)
        norms = np.where(norms == 0, 1.0, norms)
        D = D / norms

    return D


# ---------------------------------------------------------------------------
# SL0 (Smoothed L0) reconstruction
# ---------------------------------------------------------------------------
def solve_sl0(
    A: np.ndarray,
    b: np.ndarray,
    sigma_min: float = 0.01,
    sigma_decrease_factor: float = 0.5,
    mu_0: float = 1.0,
    L: int = 3,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Reconstruct a sparse signal using the Smoothed L0 (SL0) algorithm.

    SL0 approximates the L0 norm by a smooth surrogate and performs a
    steepest-descent / projection iteration to find the sparsest solution of
    the underdetermined linear system ``b = A @ x``.

    Parameters
    ----------
    A : np.ndarray
        Sensing matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    sigma_min : float, optional
        Minimum value of the smoothing parameter sigma (default: 0.01).
    sigma_decrease_factor : float, optional
        Factor by which sigma is decreased each outer iteration (default: 0.5).
    mu_0 : float, optional
        Step-size factor for the steepest descent (default: 1.0). The effective
        step is ``mu_0 * x * exp(-x^2 / (2 sigma^2))``; ``mu_0 = 1`` drives
        small coefficients toward zero without sign flips.
    L : int, optional
        Number of inner steepest-descent iterations per sigma (default: 3).
    max_iterations : int, optional
        Maximum number of outer iterations (default: 1000).
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).

    Returns
    -------
    np.ndarray
        Reconstructed sparse signal (n,).
    """
    m, n = A.shape

    # Initial solution: minimum-norm least-squares solution
    x = np.linalg.pinv(A) @ b

    # Projection matrix P = A^T (A A^T)^{-1} with shape (n, m), used to
    # project any vector back onto the feasible set {x : A x = b}.
    # x_new = x - P @ (A @ x - b)
    pinv_AT = A.T @ np.linalg.pinv(A @ A.T)

    sigma = 2.0 * np.max(np.abs(x))
    if sigma == 0:
        sigma = 1.0

    sigma = max(sigma, sigma_min)

    for _ in range(max_iterations):
        x_prev = x.copy()

        # Inner steepest descent on the smoothed L0 surrogate
        for _ in range(L):
            # Gaussian surrogate: f(x) = sum(1 - exp(-x^2 / (2 sigma^2)))
            # Gradient: grad_i = (x_i / sigma^2) * exp(-x_i^2 / (2 sigma^2))
            # With step size mu = mu_0 * sigma^2, the update becomes
            # x = x - mu_0 * x * exp(-x^2 / (2 sigma^2)).
            exp_term = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
            x = x - mu_0 * x * exp_term

            # Project back onto the feasible set {x : A x = b}
            x = x - pinv_AT @ (A @ x - b)

        sigma *= sigma_decrease_factor
        if sigma < sigma_min:
            break

        # Convergence check on the change in the solution (the residual is
        # always ~0 due to the projection, so it cannot be used here).
        if np.linalg.norm(x - x_prev) < tolerance * max(1.0, np.linalg.norm(x)):
            break

    return x


# ---------------------------------------------------------------------------
# Main CS solver
# ---------------------------------------------------------------------------
def solve_cs(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    n_atoms: Optional[int] = None,
    sparsity: Optional[int] = None,
    dictionary: Optional[np.ndarray] = None,
    n_dictionary_iterations: int = 20,
    sigma_min: float = 0.01,
    sigma_decrease_factor: float = 0.5,
    mu_0: float = 1.0,
    L: int = 3,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, int, bool]:
    """Solve the unfolding problem using Compressive Sensing (CS).

    The spectrum ``x`` is represented sparsely in a learned dictionary ``D`` as
    ``x = D @ alpha``. The measurement equation ``b = A @ x`` becomes
    ``b = (A @ D) @ alpha``, which is solved for the sparse ``alpha`` using SL0.
    Finally the spectrum is reconstructed as ``x = D @ alpha``.

    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial guess (n,). Used to seed the dictionary training signals.
    n_atoms : int, optional
        Number of dictionary atoms. Defaults to ``max(n, 2 * m)``.
    sparsity : int, optional
        Target sparsity for dictionary learning. Defaults to ``max(1, n // 20)``.
    dictionary : np.ndarray, optional
        Pre-learned dictionary (n x n_atoms). If provided, dictionary learning
        is skipped.
    n_dictionary_iterations : int, optional
        Number of K-SVD iterations (default: 20).
    sigma_min : float, optional
        SL0 minimum sigma (default: 0.01).
    sigma_decrease_factor : float, optional
        SL0 sigma decrease factor (default: 0.5).
    mu_0 : float, optional
        SL0 step-size factor (default: 1.0).
    L : int, optional
        SL0 inner iterations per sigma (default: 3).
    max_iterations : int, optional
        SL0 maximum outer iterations (default: 1000).
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, int, bool]
        Tuple of (solution, iterations, converged).
    """
    m, n = A.shape

    if n_atoms is None:
        n_atoms = max(n, 2 * m)
    if sparsity is None:
        sparsity = max(1, n // 20)

    # Build training signals for dictionary learning.
    # Use the initial guess (if provided) plus smooth basis vectors so that the
    # learned dictionary can represent typical smooth neutron spectra.
    if x0 is not None and np.any(x0):
        base = np.maximum(x0, 0)
        base = base / (np.linalg.norm(base) + 1e-12)
    else:
        base = np.ones(n) / np.sqrt(n)

    # Training signals: smooth basis (cosine-like) plus the initial guess.
    t = np.linspace(0.0, np.pi, n)
    n_basis = min(n, max(2 * m, 8))
    signals = np.zeros((n, n_basis + 1))
    for i in range(n_basis):
        signals[:, i] = np.cos(i * t)
        norm = np.linalg.norm(signals[:, i])
        if norm > 0:
            signals[:, i] /= norm
    signals[:, -1] = base

    # Learn the dictionary (or use the provided one)
    if dictionary is not None:
        D = np.asarray(dictionary, dtype=float)
        if D.shape[0] != n:
            raise ValueError(
                f"Dictionary first dimension ({D.shape[0]}) must match "
                f"number of energy bins ({n})"
            )
    else:
        D = solve_ksvd(
            signals,
            n_atoms=n_atoms,
            n_iterations=n_dictionary_iterations,
            sparsity=sparsity,
            random_state=random_state,
        )

    # Effective sensing matrix: Phi = A @ D
    Phi = A @ D

    # Solve for sparse coefficients using SL0
    alpha = solve_sl0(
        Phi,
        b,
        sigma_min=sigma_min,
        sigma_decrease_factor=sigma_decrease_factor,
        mu_0=mu_0,
        L=L,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    # Reconstruct the spectrum
    x = D @ alpha
    x = np.maximum(x, 0)

    # Normalize so that the reconstructed readings match the measurements
    # in total magnitude (scale-invariant unfolding).
    computed = A @ x
    if np.linalg.norm(computed) > 0 and np.linalg.norm(b) > 0:
        scale = np.dot(b, computed) / (np.dot(computed, computed) + 1e-12)
        x = x * scale

    residual = np.linalg.norm(A @ x - b)
    converged = bool(residual < tolerance * max(1.0, np.linalg.norm(b)))
    iterations = max_iterations

    return x, iterations, converged


# ---------------------------------------------------------------------------
# unfold_cs wrapper
# ---------------------------------------------------------------------------
def unfold_cs(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    n_atoms: Optional[int] = None,
    sparsity: Optional[int] = None,
    dictionary: Optional[np.ndarray] = None,
    n_dictionary_iterations: int = 20,
    sigma_min: float = 0.01,
    sigma_decrease_factor: float = 0.5,
    mu_0: float = 1.0,
    L: int = 3,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold neutron spectrum using Compressive Sensing (CS).

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
        Initial spectrum guess.
    n_atoms : int, optional
        Number of dictionary atoms.
    sparsity : int, optional
        Target sparsity for dictionary learning.
    dictionary : np.ndarray, optional
        Pre-learned dictionary (n x n_atoms).
    n_dictionary_iterations : int, optional
        Number of K-SVD iterations (default: 20).
    sigma_min : float, optional
        SL0 minimum sigma (default: 0.01).
    sigma_decrease_factor : float, optional
        SL0 sigma decrease factor (default: 0.5).
    mu_0 : float, optional
        SL0 step-size factor (default: 1.0).
    L : int, optional
        SL0 inner iterations per sigma (default: 3).
    max_iterations : int, optional
        SL0 maximum outer iterations (default: 1000).
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
        Random seed for reproducibility.

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
            solve_cs,
            n_atoms=n_atoms,
            sparsity=sparsity,
            dictionary=dictionary,
            n_dictionary_iterations=n_dictionary_iterations,
            sigma_min=sigma_min,
            sigma_decrease_factor=sigma_decrease_factor,
            mu_0=mu_0,
            L=L,
            max_iterations=max_iterations,
            tolerance=tolerance,
            random_state=random_state,
        ),
        solve_kwargs={},
        method_name="CompressiveSensing",
        extra_output={},
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )