"""Lavrentiev regularization unfolding.

This module implements spectrum recovery by **Lavrentiev regularization**
—the classical regularization scheme for Fredholm / Volterra integral
equations of the first kind proposed by M.M. Lavrentiev (1962) and
developed in Tikhonov & Arsenin, *Solutions of Ill-Posed Problems*
(Wiley, 1977), as well as the Russian ill-posed-problems tradition
(M.M. Lavrentiev & L.Ya. Savel'ev, *Operator Theory and Ill-Posed
Problems*, Nauka, 1990).

Mathematical formulation
------------------------
Given the ill-posed operator equation

    A z = b,    A : X → Y,  dim X = n, dim Y = m,

the **direct Lavrentiev scheme** (Ref. [1], eq. 0.2) replaces the
operator A by the shifted operator ``A + α I`` and solves

    (A + α I) z_α = b.                                         (L)

This is mathematically **distinct from** Tikhonov regularization
``(Aᵀ A + α I) z = Aᵀ b``: Lavrentiev perturbs the operator A itself,
whereas Tikhonov perturbs the (always self-adjoint PSD) Gram operator
``Aᵀ A``. Even for self-adjoint positive A the two methods differ —
Lavrentiev filters the spectrum of A by ``λ / (λ + α)``, Tikhonov by
``λ² / (λ² + α)`` (different regularisation power).

Applicability. The direct scheme (L) is well-posed only when ``A + α I``
is invertible with the resolvent bound ``‖(A + α I)⁻¹‖ = O(1/α)`` —
i.e. when A is **accretive / sectorial** (in the simplest case,
self-adjoint positive semi-definite; for the Volterra operators of
Ref. [1] this is the diagonal condition ``min |K(t,t)| = d > 0``).
For arbitrary rectangular A — the Bonner-sphere unfolding case
(``m`` detectors ≪ ``n`` energy bins) — the identity cannot be added
to A directly and the standard extension (Ref. [2], Ref. [3]) is the
**Lavrentiev scheme applied to the Gram operator**:

    Let  B := A Aᵀ  ∈ ℝ^{m × m}   (B = Bᵀ,  B ⪰ 0).
    Solve     (B + α I_m) y = b,                            (L-gram)
    Recover   z_α = Aᵀ y.

This is the genuine Lavrentiev perturbation ``(B + α I) y = b`` applied
to the *square* positive semi-definite operator ``B = A Aᵀ``, with the
recovery step ``z = Aᵀ y`` mapping the m-dimensional measurement-space
solution back to the n-dimensional spectrum space. It is the textbook
generalization of Lavrentiev regularization to non-square operators and
is **computationally cheaper** than the n×n Tikhonov system when
``m ≪ n``: it inverts an m × m system instead of an n × n one, at the
cost of one extra ``Aᵀ y`` mat-vec.

Equivalence with zeroth-order Tikhonov (for transparency).
By the push-through / Woodbury identity
``(Aᵀ A + α I)⁻¹ Aᵀ = Aᵀ (A Aᵀ + α I)⁻¹``, the recovered spectrum
of (L-gram) is *identically equal* to the zeroth-order Tikhonov
solution ``z_Tikh = (Aᵀ A + α I)⁻¹ Aᵀ b``. The two formulations are
therefore two algebraic routes to the same vector ``z_α``. We retain
the historical name "Lavrentiev" for the (L-gram) path because (i) the
shifted-operator interpretation ``(B + α I) y = b`` is genuinely
Lavrentiev's idea applied to B, (ii) it predates and motivates the
variational Tikhonov formulation, and (iii) the m×m system is the
practical reason to prefer this form for Bonner-sphere unfolding
(typically 6–20 detectors vs. 60–100 energy bins).

Two forms are exposed via the ``form`` keyword:

* ``form="gram"``  (default, any A): (L-gram), the Lavrentiev scheme
  applied to the m × m Gram operator ``A Aᵀ``.
* ``form="direct"``                : (L), the classical Lavrentiev
  scheme applied directly to A; requires a **square** response matrix
  (``m == n``). Raises ``ValueError`` for rectangular A, since the
  identity cannot be added to a non-square matrix.

Convergence (Ref. [1], Theorem 2). For an accretive A with noise
level δ on the data b, the a-priori parameter choice
``α = δ^ν,  ν ∈ (0, 1)``,  yields
``‖z_α − z*‖ → 0`` as ``δ → 0``; under the canonical source condition
``z* ∈ Range(A^μ)``  (μ ≤ 1 for single-step Lavrentiev) the rate is
``O(δ^{2μ/(2μ+1)})``  (attained at  ``α ~ δ^{1/(2μ+1)}``).

References
----------
[1] Muftahov I.R., Sidorov D.N., Sidorov N.A.
    *On Lavrentiev regularization of integral equations of the first
    kind in the space of continuous functions.*
    Izv. Irkutsk. Gos. Univ. Ser. Mat. **15** (2016), 62–77.
    https://cyberleninka.ru/article/n/o-regulyarizatsii-po-lavrentievu-integralnyh-uravneniy-pervogo-roda-v-prostranstve-nepreryvnyh-funktsiy
[2] Lavrentiev M.M., Savel'ev L.Ya. *Operator Theory and Ill-Posed
    Problems*. Nauka, Novosibirsk, 1990 (in Russian).
[3] Tikhonov A.N., Arsenin V.Y. *Solutions of Ill-Posed Problems*.
    Wiley, New York, 1977.
[4] *Method of Shift Regularization: Theory and Applications*
    («Метод регуляризации сдвигом»). MSU NIVC preprint, 370 pp.,
    2013. Chapter 1 §5 «Метод регуляризации М.М. Лаврентьева»;
    Introduction p. 4 (zero-padding reduction of rectangular A);
    Theorem 5.1 (semisimple-zero eigenvalue condition);
    §12 (a-priori parameter choice rules).
    https://num-anal.srcc.msu.ru/list_wrk/ps/b5.pdf
[5] Bakushinskii A.B., Kokurin M.Yu. *Iterative Methods for
    Approximate Solution of Inverse Problems*. Springer, 2004.
[6] Mahale P., Nair M.T. *Iterated Lavrentiev regularization for
    nonlinear ill-posed problems.* ANZIAM J. **51** (2009), 191–217.
"""

from typing import Any

import numpy as np

from ._base_unfolder import make_solve_wrapper, run_unfolding

__all__ = ["solve_lavrentiev", "unfold_lavrentiev"]


def solve_lavrentiev(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    alpha: float = 0.05,
    form: str = "gram",
    q: float = 0.5,
    n_iterations: int = 5,
) -> np.ndarray:
    """Solve unfolding using Lavrentiev regularization.

    Implements the classical Lavrentiev scheme
    ``(Operator + alpha * I) z = b`` in one of two forms:

    * ``form="gram"`` (default, works for any rectangular A):
      applies Lavrentiev's idea to the m × m Gram operator
      ``B = A A^T`` (which is square and positive semi-definite):

          (A A^T + alpha * I_m) y = b
          z                       = A^T y

      Mathematically equivalent to zeroth-order Tikhonov
      ``(A^T A + alpha I) z = A^T b`` via the push-through identity,
      but uses an m × m system (cheap when ``m ≪ n``, the
      Bonner-sphere case).

    * ``form="direct"`` (square A only): the classical Lavrentiev
      scheme applied directly to A:

          (A + alpha * I) z = b

      This is *genuinely different from Tikhonov* even for
      self-adjoint A — it filters the spectrum of A itself
      (``λ/(λ+α)``) rather than the spectrum of ``A^T A``
      (``λ²/(λ²+α)``). Requires A to be square; raises
      ``ValueError`` for rectangular A.

    * ``form="padded"`` (any A, including rectangular): zero-pads A
      to a square ``max(m, n) × max(m, n)`` matrix with zero rows
      (when ``m < n``) or zero columns (when ``m > n``) and applies
      the direct Lavrentiev scheme to the padded operator (Ref. [4],
      Introduction p. 4). Valid under Theorem 5.1 of Ref. [4]
      (semisimple zero eigenvalue of the padded matrix).

      Practical caveat: when ``m < n`` (the Bonner-sphere case)
      the bottom ``n − m`` rows of the padded operator are zero, so
      the corresponding diagonal entries of ``A_pad + αI`` reduce
      to ``α`` and the recovered ``z_i`` for ``i > m`` are forced
      to zero. The method is therefore mathematically legitimate
      but loses information in the padded energy bins — prefer
      ``form="gram"`` for production BSS unfolding.

    * ``form="iterated"`` (any A): the **iterated Lavrentiev
      scheme** with Bakushinsky's a-priori α-decay (Refs. [5], [6]).
      At iteration ``k = 0, 1, …, n_iterations − 1`` performs the
      defect-correction update on the m × m Gram operator
      ``B = A A^T``:

          y_{k+1} = y_k + (B + α_k I_m)^{-1} (b − B y_k),   y_0 = 0
          α_k     = alpha * q^k                                (Bakushinsky rule)

      and recovers the spectrum ``z = A^T y_{K}`` after ``K =
      n_iterations`` steps. Geometric decay ``0 < q < 1`` is the
      classical Bakushinsky a-priori rule; setting ``q = 1.0``
      recovers the constant-α iterated Lavrentiev of Mahale & Nair
      (2009), which achieves order-optimality for smoother source
      conditions (qualification ``μ ≤ K`` instead of ``μ ≤ 1``
      for the single-step scheme).

      Practical advantage over the single-step Gram form: better
      recovery of smooth spectral features for the same final
      ``α_{K-1}`` because the iteration effectively averages the
      regularized solutions across a range of α values. Cost:
      ``n_iterations`` solves of an m × m system (cheap for the
      BSS case ``m ≈ 7``).

    Parameters
    ----------
    A : np.ndarray
        Response matrix of shape ``(m, n)``.
    b : np.ndarray
        Measurement vector of shape ``(m,)``.
    x0 : np.ndarray, optional
        Not used (provided for API compatibility with the shared
        ``make_solve_wrapper`` machinery).
    alpha : float, optional
        Regularization parameter (default: ``0.05``). For the
        single-step forms (``gram``, ``direct``, ``padded``) this is
        the shift magnitude. For the iterated form (``iterated``)
        this is the initial shift ``α_0``; subsequent shifts decay
        geometrically as ``α_k = alpha * q^k``. Must be non-negative.
    form : {"gram", "direct", "padded", "iterated"}, optional
        Which Lavrentiev system to solve (default: ``"gram"``).
        See the module docstring for the precise mathematical
        formulation and the applicability conditions of each form.
    q : float, optional
        Geometric decay rate of the regularization parameter in the
        iterated form (default: ``0.5``). Only used when
        ``form="iterated"``. Must satisfy ``0 < q ≤ 1``.
        ``q = 1.0`` recovers the constant-α iterated Lavrentiev
        (higher qualification, no Bakushinsky decay); ``q < 1``
        gives the classical Bakushinsky a-priori rule.
    n_iterations : int, optional
        Number of defect-correction iterations in the iterated form
        (default: ``5``). Only used when ``form="iterated"``.
        Must be ≥ 1.

    Returns
    -------
    np.ndarray
        Unfolded spectrum of shape ``(n,)``, clipped to non-negative
        values.

    Raises
    ------
    ValueError
        If ``alpha`` is negative, or if ``form="direct"`` is requested
        with a non-square response matrix, or if ``form="iterated"``
        is requested with invalid ``q`` or ``n_iterations``, or if
        ``form`` is not one of ``"gram"``, ``"direct"``, ``"padded"``,
        ``"iterated"``.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha!r}")

    m, n = A.shape

    if form == "direct":
        if m != n:
            raise ValueError(
                f"Direct Lavrentiev form (A + alpha*I) z = b requires a "
                f"square response matrix, got A.shape={A.shape!r} "
                f"(m={m} != n={n}). Use form='gram' instead, which "
                f"applies Lavrentiev's scheme to the {m}x{m} Gram "
                f"operator A A^T and recovers z = A^T y."
            )
        # Classical Lavrentiev scheme on the operator A itself:
        #     (A + alpha * I) z = b
        M = A + alpha * np.eye(n)
        rhs = b
        try:
            z = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            # Singular A + alpha*I (only possible when alpha = 0 and
            # A is rank-deficient). Fall back to a least-squares
            # solve that returns the minimum-norm solution.
            z = np.linalg.lstsq(M, rhs, rcond=None)[0]
    elif form == "gram":
        # Lavrentiev's scheme applied to the m x m Gram operator
        # B = A A^T (square, PSD):
        #     (A A^T + alpha * I_m) y = b
        #     z                       = A^T y
        M = A @ A.T + alpha * np.eye(m)
        rhs = b
        try:
            y = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            y = np.linalg.lstsq(M, rhs, rcond=None)[0]
        z = A.T @ y
    elif form == "padded":
        # Zero-pad A to a square max(m, n) x max(m, n) operator and
        # apply the direct Lavrentiev scheme to the padded operator
        # (Ref. [4], Introduction p. 4). The semisimplicity condition
        # of Theorem 5.1 (Ref. [4]) is generically satisfied for a
        # zero-padded rectangular matrix (the padding-induced zero
        # eigenvalues are diagonal and therefore semisimple).
        #
        # When m < n (the Bonner-sphere case), we pad with (n-m) zero
        # rows; the bottom (n-m) equations of (A_pad + alpha*I) z = b_pad
        # reduce to  alpha * z_i = 0,  forcing z_i = 0 for i > m.
        # The method is mathematically legitimate but loses the last
        # (n-m) energy bins -- prefer form='gram' for production.
        size = max(m, n)
        A_pad = np.zeros((size, size))
        A_pad[:m, :n] = A
        b_pad = np.zeros(size)
        b_pad[:m] = b
        M = A_pad + alpha * np.eye(size)
        try:
            z_full = np.linalg.solve(M, b_pad)
        except np.linalg.LinAlgError:
            z_full = np.linalg.lstsq(M, b_pad, rcond=None)[0]
        # Truncate back to the original n bins (the padded components
        # beyond n correspond to padding columns and carry no
        # physical information; they are discarded).
        z = z_full[:n]
    elif form == "iterated":
        # Iterated Lavrentiev scheme with Bakushinsky's a-priori
        # α-decay (Refs. [5], [6]). Defect-correction iteration on
        # the m x m Gram operator B = A A^T:
        #
        #     y_{k+1} = y_k + (B + alpha_k I_m)^{-1} (b - B y_k),
        #     y_0     = 0,
        #     alpha_k = alpha * q^k.                        (Bakushinsky)
        #
        # After K = n_iterations steps, recover z = A^T y_K.
        # Geometric decay 0 < q < 1 is Bakushinsky's a-priori rule
        # (Ref. [5]); q = 1 recovers the constant-α iterated
        # Lavrentiev of Mahale & Nair (Ref. [6]) which achieves
        # higher qualification (source conditions with μ up to K
        # instead of μ ≤ 1).
        if not (0.0 < q <= 1.0):
            raise ValueError(
                f"q must satisfy 0 < q <= 1, got {q!r}"
            )
        if n_iterations < 1:
            raise ValueError(
                f"n_iterations must be >= 1, got {n_iterations!r}"
            )
        B = A @ A.T  # m x m Gram operator
        y = np.zeros(m)
        eye_m = np.eye(m)
        for k in range(n_iterations):
            alpha_k = alpha * (q ** k)
            residual = b - B @ y
            try:
                update = np.linalg.solve(
                    B + alpha_k * eye_m, residual,
                )
            except np.linalg.LinAlgError:
                update = np.linalg.lstsq(
                    B + alpha_k * eye_m, residual, rcond=None,
                )[0]
            y = y + update
        z = A.T @ y
    else:
        raise ValueError(
            f"form must be 'gram', 'direct', 'padded', or 'iterated', "
            f"got {form!r}"
        )

    return np.maximum(z, 0.0)


def unfold_lavrentiev(
    detector_names: list[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: dict[str, np.ndarray],
    cc_icrp116: dict[str, np.ndarray],
    save_result_callback,
    readings: dict[str, float],
    ln_steps: np.ndarray | None = None,
    initial_spectrum: np.ndarray | None = None,
    alpha: float = 0.05,
    form: str = "gram",
    q: float = 0.5,
    n_iterations: int = 5,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 100,
    save_result: bool = False,
    random_state: int | None = None,
    reading_uncertainties: dict[str, float] | np.ndarray | None = None,
    reading_covariance: np.ndarray | None = None,
    noise_model: str = "gaussian",
    measurement_time: float | None = None,
) -> dict[str, Any]:
    """Unfold neutron spectrum using Lavrentiev regularization.

    Solves the Lavrentiev-regularized system

        (A A^T + alpha * I_m) y = b,    z = A^T y        (form="gram")

    (the default; the textbook generalization of Lavrentiev's scheme
    to non-square A — see the module docstring for details), or the
    classical Lavrentiev scheme

        (A + alpha * I) z = b                            (form="direct")

    (square A only), and clips the recovered spectrum to non-negative
    values. The method operates **directly in the discrete n-bin
    energy space** (no projection onto a polynomial basis) — it is the
    appropriate choice for unfolding on the detector's 60-bin lethargy
    grid.

    Parameters
    ----------
    detector_names : List[str]
        Names of available detectors.
    n_energy_bins : int
        Number of energy bins on the detector grid.
    E_MeV : np.ndarray
        Energy grid (MeV).
    sensitivities : Dict[str, np.ndarray]
        Per-detector sensitivity arrays.
    cc_icrp116 : Dict[str, np.ndarray]
        ICRP-116 ambient-dose-equivalent conversion coefficients.
    save_result_callback : callable
        Callback to save the result to the detector history.
    readings : Dict[str, float]
        Detector readings.
    ln_steps : np.ndarray, optional
        Lethargy bin widths (used for the integration rule).
    initial_spectrum : np.ndarray, optional
        Not used by the Lavrentiev solver (provided for API
        compatibility with the shared ``run_unfolding`` engine).
    alpha : float, optional
        Regularization parameter (default: ``0.05``). Must be
        non-negative.
    form : {"gram", "direct"}, optional
        Which Lavrentiev system to solve (default: ``"gram"``).
        ``"gram"`` is the m × m Gram-operator form
        ``(A A^T + alpha I) y = b; z = A^T y`` and works for any
        rectangular A. ``"direct"`` is the classical Lavrentiev
        scheme ``(A + alpha I) z = b`` and requires a square response
        matrix.
    calculate_errors : bool, optional
        If ``True``, run a Monte-Carlo uncertainty propagation
        (default: ``False``).
    noise_level : float, optional
        Relative noise level for Monte-Carlo (default: ``0.01``).
    n_montecarlo : int, optional
        Number of Monte-Carlo samples (default: ``100``).
    save_result : bool, optional
        Save result to detector history (default: ``False``).
    random_state : int, optional
        Random seed for reproducible Monte-Carlo.
    reading_uncertainties : dict or np.ndarray, optional
        Per-detector reading uncertainties.
    reading_covariance : np.ndarray, optional
        Full reading covariance matrix.
    noise_model : str, optional
        Monte-Carlo noise model (default: ``"gaussian"``).
    measurement_time : float, optional
        Measurement time for Poisson noise scaling.

    Returns
    -------
    Dict[str, Any]
        Standard unfolding result dict, with extra keys ``alpha``
        and ``form`` recording the regularization parameter and the
        Lavrentiev form used.
    """
    x0_default = np.zeros(n_energy_bins)

    return run_unfolding(
        detector_names=detector_names,
        n_energy_bins=n_energy_bins,
        E_MeV=E_MeV,
        sensitivities=sensitivities,
        cc_icrp116=cc_icrp116,
        save_result_callback=save_result_callback,
        ln_steps=ln_steps,
        readings=readings,
        initial_spectrum=initial_spectrum,
        default_initial=x0_default,
        solve_func=make_solve_wrapper(
            solve_lavrentiev,
            alpha=alpha,
            form=form,
            q=q,
            n_iterations=n_iterations,
        ),
        solve_kwargs={},
        method_name="Lavrentiev",
        extra_output={
            "alpha": alpha,
            "form": form,
            "q": q,
            "n_iterations": n_iterations,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
            reading_uncertainties=reading_uncertainties,
            reading_covariance=reading_covariance,
                noise_model=noise_model,
                measurement_time=measurement_time,
    )
