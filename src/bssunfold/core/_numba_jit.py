"""Numba JIT-compiled inner loops for bssunfold solvers.

This module provides high-performance JIT-compiled versions of the
inner loops used in unfolding algorithms. These are called from the
main solver functions when numba is available.

The functions use numba.prange for automatic parallelization where beneficial.
"""

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _landweber_inner(AT, A, x, b, step_size, max_iterations, tolerance):
        """JIT-compiled Landweber iteration inner loop.

        Parameters
        ----------
        AT : np.ndarray
            Transposed response matrix (n x m).
        A : np.ndarray
            Response matrix (m x n).
        x : np.ndarray
            Solution vector (n,).
        b : np.ndarray
            Measurement vector (m,).
        step_size : float
            Step size (1 / sigma_max^2).
        max_iterations : int
            Maximum iterations.
        tolerance : float
            Convergence tolerance.

        Returns
        -------
        tuple
            (solution, iterations, converged)
        """
        m = A.shape[0]
        n = A.shape[1]
        converged = False
        iterations = 0

        for i in range(max_iterations):
            # Compute residual = A @ x - b
            residual_norm = 0.0
            for k in range(m):
                ax_k = 0.0
                for j in range(n):
                    ax_k += A[k, j] * x[j]
                r_k = ax_k - b[k]
                residual_norm += r_k * r_k
            residual_norm = np.sqrt(residual_norm)

            if residual_norm < tolerance:
                converged = True
                iterations = i
                break

            # x = x - step_size * (AT @ residual)
            for j in range(n):
                at_r = 0.0
                for k in range(m):
                    ax_k = 0.0
                    for l in range(n):
                        ax_k += A[k, l] * x[l]
                    at_r += AT[j, k] * (ax_k - b[k])
                x[j] -= step_size * at_r
                if x[j] < 0.0:
                    x[j] = 0.0

        if not converged:
            iterations = max_iterations

        return x, iterations, converged

    @njit(cache=True)
    def _mlem_inner(AT, A, x, b, max_iterations, tolerance):
        """JIT-compiled MLEM iteration inner loop.

        Parameters
        ----------
        AT : np.ndarray
            Transposed response matrix (n x m).
        A : np.ndarray
            Response matrix (m x n).
        x : np.ndarray
            Solution vector (n,).
        b : np.ndarray
            Measurement vector (m,).
        max_iterations : int
            Maximum iterations.
        tolerance : float
            Convergence tolerance.

        Returns
        -------
        tuple
            (solution, iterations, converged)
        """
        m = A.shape[0]
        n = A.shape[1]
        converged = False
        iterations = 0
        eps = 1e-10

        for i in range(max_iterations):
            # Compute Ax = A @ x
            Ax = np.empty(m)
            for k in range(m):
                ax_k = 0.0
                for j in range(n):
                    ax_k += A[k, j] * x[j]
                Ax[k] = max(ax_k, eps)

            # Compute correction = AT @ (b / Ax)
            correction = np.zeros(n)
            for j in range(n):
                corr_j = 0.0
                for k in range(m):
                    corr_j += AT[j, k] * (b[k] / Ax[k])
                correction[j] = corr_j

            # Update x
            diff_norm = 0.0
            x_norm = 0.0
            for j in range(n):
                x_new = x[j] * correction[j]
                if x_new < 0.0:
                    x_new = 0.0
                diff_norm += (x_new - x[j]) ** 2
                x_norm += x[j] ** 2
                x[j] = x_new

            iterations = i + 1
            if np.sqrt(diff_norm) / (np.sqrt(x_norm) + eps) < tolerance:
                converged = True
                break

        return x, iterations, converged

    @njit(cache=True)
    def _kaczmarz_inner(A, x, b, row_norms_sq, omega, max_iterations, tolerance):
        """JIT-compiled Kaczmarz iteration inner loop.

        Parameters
        ----------
        A : np.ndarray
            Response matrix (m x n).
        x : np.ndarray
            Solution vector (n,).
        b : np.ndarray
            Measurement vector (m,).
        row_norms_sq : np.ndarray
            Squared row norms of A.
        omega : float
            Relaxation parameter.
        max_iterations : int
            Maximum iterations.
        tolerance : float
            Convergence tolerance.

        Returns
        -------
        tuple
            (solution, iterations, converged)
        """
        m = A.shape[0]
        n = A.shape[1]
        converged = False
        iterations = 0
        x_old = x.copy()

        for k in range(max_iterations):
            i = k % m
            if row_norms_sq[i] > 0:
                # Compute dot(A[i], x)
                dot_val = 0.0
                for j in range(n):
                    dot_val += A[i, j] * x[j]
                update = (b[i] - dot_val) / row_norms_sq[i]
                for j in range(n):
                    x[j] += omega * update * A[i, j]
                    if x[j] < 0.0:
                        x[j] = 0.0

            if (k + 1) % m == 0:
                # Check convergence
                diff_norm = 0.0
                for j in range(n):
                    diff_norm += (x[j] - x_old[j]) ** 2
                if np.sqrt(diff_norm) < tolerance:
                    converged = True
                    iterations = k + 1
                    break
                for j in range(n):
                    x_old[j] = x[j]

        if not converged:
            iterations = max_iterations

        return x, iterations, converged

    @njit(cache=True)
    def _doroshenko_inner(A, x, b, denominator_cache, max_iterations, tolerance):
        """JIT-compiled Doroshenko coordinate update inner loop.

        Parameters
        ----------
        A : np.ndarray
            Response matrix (m x n).
        x : np.ndarray
            Solution vector (n,).
        b : np.ndarray
            Measurement vector (m,).
        denominator_cache : np.ndarray
            Precomputed sum of squared columns.
        max_iterations : int
            Maximum iterations.
        tolerance : float
            Convergence tolerance.

        Returns
        -------
        tuple
            (solution, iterations, converged)
        """
        m = A.shape[0]
        n = A.shape[1]
        converged = False
        iterations = 0

        # Compute initial residual = b - A @ x
        residual = np.empty(m)
        for k in range(m):
            ax_k = 0.0
            for j in range(n):
                ax_k += A[k, j] * x[j]
            residual[k] = b[k] - ax_k

        for i in range(max_iterations):
            x_old = x.copy()

            for j in range(n):
                if denominator_cache[j] <= 0:
                    continue
                # Compute dot(A[:, j], residual)
                Aj_dot_res = 0.0
                for k in range(m):
                    Aj_dot_res += A[k, j] * residual[k]
                numerator = Aj_dot_res + denominator_cache[j] * x[j]
                new_xj = numerator / denominator_cache[j]
                if new_xj < 0.0:
                    new_xj = 0.0
                delta = new_xj - x[j]
                if delta != 0.0:
                    for k in range(m):
                        residual[k] -= delta * A[k, j]
                    x[j] = new_xj

            # Check convergence
            diff_norm = 0.0
            for j in range(n):
                diff_norm += (x[j] - x_old[j]) ** 2
            if np.sqrt(diff_norm) < tolerance:
                converged = True
                iterations = i + 1
                break

        if not converged:
            iterations = max_iterations

        return x, iterations, converged

    @njit(cache=True)
    def _gravel_inner(A_valid, x, b_valid, regularization, max_iterations, tolerance):
        """JIT-compiled GRAVEL algorithm inner loop.

        Parameters
        ----------
        A_valid : np.ndarray
            Response matrix for valid measurements (m_valid x n).
        x : np.ndarray
            Solution vector (n,).
        b_valid : np.ndarray
            Valid measurement vector (m_valid,).
        regularization : float
            Regularization parameter.
        max_iterations : int
            Maximum iterations.
        tolerance : float
            Convergence tolerance.

        Returns
        -------
        tuple
            (solution, iterations, converged)
        """
        m_valid = A_valid.shape[0]
        n = A_valid.shape[1]
        eps = 1e-10

        J_prev = 0.0
        dJ_prev = 1.0

        for iteration in range(1, max_iterations + 1):
            # Compute computed = A_valid @ x
            computed = np.empty(m_valid)
            for i in range(m_valid):
                ax_i = 0.0
                for j in range(n):
                    ax_i += A_valid[i, j] * x[j]
                computed[i] = ax_i

            # Update each energy bin
            for j in range(n):
                numerator = 0.0
                denominator = 0.0
                for i in range(m_valid):
                    if computed[i] > eps and x[j] > eps and b_valid[i] > eps and A_valid[i, j] > eps:
                        W_ij = b_valid[i] * A_valid[i, j] * x[j] / computed[i]
                        log_ratio = np.log(b_valid[i] / computed[i])
                        numerator += W_ij * log_ratio
                        denominator += W_ij

                if denominator > eps:
                    reg_term = regularization * np.log(x[j] + eps)
                    update = np.exp((numerator - reg_term) / denominator)
                    x[j] *= update

            # Compute convergence criterion
            computed_final = np.empty(m_valid)
            for i in range(m_valid):
                ax_i = 0.0
                for j in range(n):
                    ax_i += A_valid[i, j] * x[j]
                computed_final[i] = ax_i

            chi_sq = 0.0
            sum_computed = 0.0
            for i in range(m_valid):
                sum_computed += computed_final[i]
                chi_sq += (computed_final[i] - b_valid[i]) ** 2 / max(b_valid[i], eps)

            J = chi_sq / max(sum_computed, eps)
            dJ = J_prev - J
            ddJ = abs(dJ - dJ_prev)

            if ddJ <= tolerance:
                return x, iteration, True

            J_prev = J
            dJ_prev = dJ

        return x, max_iterations, False

    @njit(cache=True)
    def _compute_log_steps_jit(energy):
        """JIT-compiled log step computation for dose calculations.

        Parameters
        ----------
        energy : np.ndarray
            Energy grid in MeV.

        Returns
        -------
        np.ndarray
            Logarithmic steps.
        """
        n = len(energy)
        log_steps = np.zeros(n)
        eps = 1e-15
        log10_val = np.log(10.0)

        if n < 2:
            return log_steps

        # Compute log10(energy + eps)
        log_e = np.empty(n)
        for i in range(n):
            log_e[i] = np.log10(energy[i] + eps)

        log_steps[0] = (log_e[1] - log_e[0]) * log10_val
        log_steps[n - 1] = (log_e[n - 1] - log_e[n - 2]) * log10_val
        for i in range(1, n - 1):
            log_steps[i] = (log_e[i + 1] - log_e[i - 1]) / 2.0 * log10_val

        return log_steps

    @njit(cache=True)
    def _dose_weighted_mse_jit(s1, s2, cc, ln_steps):
        """JIT-compiled dose-weighted MSE calculation.

        Parameters
        ----------
        s1 : np.ndarray
            First spectrum.
        s2 : np.ndarray
            Second spectrum.
        cc : np.ndarray
            Conversion coefficients.
        ln_steps : np.ndarray
            Logarithmic steps.

        Returns
        -------
        float
            Dose-weighted RMSE.
        """
        n = len(s1)
        total_weight = 0.0
        weighted_sq_sum = 0.0
        for i in range(n):
            w = cc[i] * ln_steps[i]
            total_weight += w
            weighted_sq_sum += w * (s1[i] - s2[i]) ** 2

        if total_weight < 1e-15:
            return 0.0
        return np.sqrt(weighted_sq_sum / total_weight)

    @njit(cache=True)
    def _monte_carlo_add_noise_jit(readings_values, noise_level, rng_state):
        """JIT-compiled noise addition for Monte Carlo.

        Parameters
        ----------
        readings_values : np.ndarray
            Reading values.
        noise_level : float
            Noise level.
        rng_state : np.ndarray
            RNG state array.

        Returns
        -------
        np.ndarray
            Noisy readings.
        """
        n = len(readings_values)
        noisy = np.empty(n)
        for i in range(n):
            # Simple Box-Muller transform using numpy rng
            noisy[i] = readings_values[i] * (1.0 + np.random.normal(0, noise_level))
        return noisy

else:
    # Fallback: no-op functions that will raise ImportError if called
    def _landweber_inner(*args, **kwargs):
        raise ImportError("numba is required for JIT-compiled solvers")

    def _mlem_inner(*args, **kwargs):
        raise ImportError("numba is required for JIT-compiled solvers")

    def _kaczmarz_inner(*args, **kwargs):
        raise ImportError("numba is required for JIT-compiled solvers")

    def _doroshenko_inner(*args, **kwargs):
        raise ImportError("numba is required for JIT-compiled solvers")

    def _gravel_inner(*args, **kwargs):
        raise ImportError("numba is required for JIT-compiled solvers")

    def _compute_log_steps_jit(*args, **kwargs):
        raise ImportError("numba is required for JIT-compiled functions")

    def _dose_weighted_mse_jit(*args, **kwargs):
        raise ImportError("numba is required for JIT-compiled functions")

    def _monte_carlo_add_noise_jit(*args, **kwargs):
        raise ImportError("numba is required for JIT-compiled functions")
