"""QUBO-based neutron spectrum unfolding using quantum-inspired annealing.

This module implements a QUBO (Quadratic Unconstrained Binary Optimization) 
formulation of the neutron spectrum unfolding problem, solvable with 
quantum-inspired simulated annealing (D-Wave Neal) or other QUBO solvers.

The approach discretizes the spectrum into binary variables and formulates
the unfolding as:

    min_x ||Ax - b||^2 + λ * R(x)
    
subject to x >= 0,

where the spectrum is represented in binary encoding for QUBO compatibility.

Requires: pyqubo, dwave-neal
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from ._base_unfolder import run_unfolding


def _spectrum_to_binary(
    spectrum: np.ndarray, 
    n_bits: int = 8,
    max_value: Optional[float] = None
) -> np.ndarray:
    """Convert continuous spectrum to binary representation.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Continuous spectrum values.
    n_bits : int
        Number of bits per energy bin.
    max_value : float, optional
        Maximum value for scaling. If None, uses max(spectrum).
    
    Returns
    -------
    np.ndarray
        Binary array of shape (n_bins * n_bits,).
    """
    if max_value is None:
        max_value = np.max(spectrum)
    if max_value <= 0:
        max_value = 1.0
    
    # Normalize to [0, 1]
    normalized = np.clip(spectrum / max_value, 0, 1)
    
    # Convert to binary
    n_bins = len(spectrum)
    binary = np.zeros(n_bins * n_bits, dtype=int)
    
    for i in range(n_bins):
        val = normalized[i]
        for j in range(n_bits):
            bit = int(val * 2)
            binary[i * n_bits + j] = bit
            val = (val * 2) % 1
    
    return binary


def _binary_to_spectrum(
    binary: np.ndarray,
    n_bins: int,
    n_bits: int = 8,
    max_value: float = 1.0
) -> np.ndarray:
    """Convert binary representation back to continuous spectrum.
    
    Parameters
    ----------
    binary : np.ndarray
        Binary array of shape (n_bins * n_bits,).
    n_bins : int
        Number of energy bins.
    n_bits : int
        Number of bits per energy bin.
    max_value : float
        Maximum value for scaling.
    
    Returns
    -------
    np.ndarray
        Continuous spectrum of shape (n_bins,).
    """
    spectrum = np.zeros(n_bins)
    
    for i in range(n_bins):
        val = 0.0
        for j in range(n_bits):
            val += binary[i * n_bits + j] * (2 ** -(j + 1))
        spectrum[i] = val * max_value
    
    return np.maximum(spectrum, 0)


def solve_qubo_unfold(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    n_bits: int = 6,
    max_value: Optional[float] = None,
    regularization: float = 0.01,
    max_iterations: int = 1000,
    annealing_time: int = 1000,
    num_reads: int = 10,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, int, bool]:
    """Solve unfolding problem using QUBO formulation with simulated annealing.
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n).
    b : np.ndarray
        Measurement vector (m,).
    x0 : np.ndarray, optional
        Initial spectrum guess for scaling.
    n_bits : int, optional
        Number of bits per energy bin (default: 6).
    max_value : float, optional
        Maximum spectrum value for scaling. If None, estimated from data.
    regularization : float, optional
        Regularization parameter (default: 0.01).
    max_iterations : int, optional
        Maximum annealing iterations (default: 1000).
    annealing_time : int, optional
        Annealing time in sweeps (default: 1000).
    num_reads : int, optional
        Number of independent reads (default: 10).
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    tuple
        (spectrum, iterations, converged)
    """
    try:
        from pyqubo import Array, Binary
        import dwave.samplers as ds
    except ImportError as e:
        raise ImportError(
            "pyqubo and dwave-neal are required for QUBO unfolding. "
            "Install with: pip install pyqubo dwave-neal"
        ) from e
    
    if random_state is not None:
        np.random.seed(random_state)
    
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m_len, n_bins = A.shape
    
    # Estimate max value if not provided
    if max_value is None:
        if x0 is not None:
            max_value = float(np.max(x0)) * 2
        else:
            # Rough estimate from pseudo-inverse
            try:
                x_est = np.linalg.lstsq(A, b, rcond=None)[0]
                max_value = float(np.max(np.abs(x_est))) * 2
            except Exception:
                max_value = 1.0
        
        if max_value <= 0:
            max_value = 1.0
    
    # Total number of binary variables
    n_binary = n_bins * n_bits
    
    # Create binary variables
    qubits = Array([Binary(f'x[{i}]') for i in range(n_binary)])
    
    # Build QUBO objective: ||A * x_binary - b||^2
    # First, create transformation matrix from binary to continuous
    # x_cont[i] = sum_j (binary[i*n_bits + j] * 2^-(j+1)) * max_value
    
    # Binary-to-continuous transformation: x_cont = T @ qubits, where
    # T[i, i*n_bits + j] = 2^-(j+1) * max_value. The spectrum has n_bins
    # entries, each encoded with n_bits qubits, so T is (n_bins, n_binary).
    T = np.zeros((n_bins, n_binary))
    for i in range(n_bins):
        for j in range(n_bits):
            T[i, i * n_bits + j] = (2 ** -(j + 1)) * max_value

    # Effective response matrix acting on the qubit vector: A @ x_cont equals
    # A @ (T @ qubits) = (A @ T) @ qubits, so A_scaled = A @ T (shape
    # m_len x n_binary), consistent with _binary_to_spectrum's decoder.
    A_scaled = A @ T
    
    # Objective: ||A_scaled @ qubits - b||^2
    # = qubits.T @ (A_scaled.T @ A_scaled) @ qubits - 2 * b.T @ A_scaled @ qubits + b.T @ b
    
    Q_matrix = A_scaled.T @ A_scaled
    linear_term = -2 * (b.T @ A_scaled)
    
    # Add regularization term (encourages sparsity/smoothness)
    # Simple L2-like penalty on binary variables
    reg_matrix = regularization * np.eye(n_binary)
    Q_matrix += reg_matrix
    
    # Build Hamiltonian using PyQUBO
    hamiltonian = 0.0
    
    # Quadratic terms
    for i in range(n_binary):
        for j in range(i + 1, n_binary):
            if abs(Q_matrix[i, j]) > 1e-10:
                hamiltonian += Q_matrix[i, j] * qubits[i] * qubits[j]
    
    # Linear terms
    for i in range(n_binary):
        qubo_linear = float(Q_matrix[i, i]) + float(linear_term[i])
        hamiltonian += qubo_linear * qubits[i]
    
    # Compile to QUBO
    model = hamiltonian.compile()
    qubo, offset = model.to_qubo()
    
    # Solve with simulated annealing
    sampler = ds.SimulatedAnnealingSampler()
    
    response = sampler.sample_qubo(
        qubo,
        num_reads=num_reads,
        num_sweeps=annealing_time,
        seed=random_state,
    )
    
    # Get best solution
    best_sample = response.first.sample
    
    # Convert binary solution back to continuous
    binary_solution = np.array([best_sample[f'x[{i}]'] for i in range(n_binary)], dtype=float)
    spectrum = _binary_to_spectrum(binary_solution, n_bins, n_bits, max_value)
    
    # Ensure non-negativity
    spectrum = np.maximum(spectrum, 0)
    
    converged = response.first.energy < float('inf')
    return spectrum, max_iterations, converged


def unfold_qubo(
    detector_names: List[str],
    n_energy_bins: int,
    E_MeV: np.ndarray,
    sensitivities: Dict[str, np.ndarray],
    cc_icrp116: Dict[str, np.ndarray],
    save_result_callback,
    readings: Dict[str, float],
    initial_spectrum: Optional[np.ndarray] = None,
    n_bits: int = 6,
    max_value: Optional[float] = None,
    regularization: float = 0.01,
    max_iterations: int = 1000,
    annealing_time: int = 1000,
    num_reads: int = 10,
    calculate_errors: bool = False,
    noise_level: float = 0.01,
    n_montecarlo: int = 50,  # Reduced due to computational cost
    save_result: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Unfold using QUBO formulation with quantum-inspired annealing.
    
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
        Initial spectrum approximation.
    n_bits : int, optional
        Bits per energy bin (default: 6).
    max_value : float, optional
        Maximum spectrum value for scaling.
    regularization : float, optional
        Regularization parameter (default: 0.01).
    max_iterations : int, optional
        Maximum iterations (default: 1000).
    annealing_time : int, optional
        Annealing sweeps (default: 1000).
    num_reads : int, optional
        Number of independent reads (default: 10).
    calculate_errors : bool, optional
        Calculate Monte-Carlo errors (default: False).
    noise_level : float, optional
        Noise level for error calculation (default: 0.01).
    n_montecarlo : int, optional
        Number of MC samples (default: 50).
    save_result : bool, optional
        Save to history (default: False).
    random_state : int, optional
        Random seed.
    
    Returns
    -------
    Dict
        Unfolding results dictionary.
    """
    x0_default = np.ones(n_energy_bins) * 0.5
    
    def solve_wrapper(A, b, **kwargs):
        x_init = kwargs.pop('x0', initial_spectrum)
        return solve_qubo_unfold(
            A, b, x0=x_init,
            n_bits=n_bits,
            max_value=max_value,
            regularization=regularization,
            max_iterations=max_iterations,
            annealing_time=annealing_time,
            num_reads=num_reads,
            random_state=random_state,
        )
    
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
        solve_func=solve_wrapper,
        solve_kwargs={},
        method_name="QUBO-Annealing",
        extra_output={
            "n_bits": n_bits,
            "regularization": regularization,
            "annealing_time": annealing_time,
            "num_reads": num_reads,
        },
        calculate_errors=calculate_errors,
        noise_level=noise_level,
        n_montecarlo=n_montecarlo,
        random_state=random_state,
        save_result=save_result,
    )


__all__ = [
    "solve_qubo_unfold",
    "unfold_qubo",
    "_spectrum_to_binary",
    "_binary_to_spectrum",
]
