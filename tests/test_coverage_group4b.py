"""Coverage group 4b: small files edge cases.

Targets:
- unfold_composite.py: lines 104, 114, 187, 245, 267-268, 271-272
- platform_check.py: lines 50, 52-53, 119-120, 137, 139-140
- unfold_gravel.py: line 54
"""

import signal

import numpy as np
import pytest


# ============================================================================
# unfold_composite.py — lines 104, 114, 187, 245, 267-268, 271-272
# ============================================================================


class TestCompositeCoverage:
    """Cover timeout and edge-case paths in composite unfolding."""

    def test_timeout_handler_raises(self):
        """Line 104: _MethodTimeout is raised by signal handler."""
        from bssunfold.core.unfold_composite import _MethodTimeout, _timeout_handler

        with pytest.raises(_MethodTimeout):
            _timeout_handler(signal.SIGALRM, None)

    def test_run_with_timeout_no_sigalrm(self):
        """Line 114: No SIGALRM -> just call fn()."""
        from bssunfold.core.unfold_composite import _run_with_timeout

        result = _run_with_timeout(lambda: 42, timeout=0.5)
        assert result == 42

    def test_run_with_timeout_none_timeout(self):
        """Line 114: timeout=None -> just call fn()."""
        from bssunfold.core.unfold_composite import _run_with_timeout

        result = _run_with_timeout(lambda: 99, timeout=None)
        assert result == 99

    def test_run_with_timeout_sigalrm_fires(self):
        """Lines 245, 267-268, 271-272: SIGALRM actually fires."""
        from bssunfold.core.unfold_composite import _run_with_timeout

        if not hasattr(signal, "SIGALRM"):
            pytest.skip("No SIGALRM on this platform")

        def slow_fn():
            import time
            time.sleep(2)
            return "done"

        # 1 second timeout, slow_fn takes 2s
        with pytest.raises(Exception):
            _run_with_timeout(slow_fn, timeout=1)

    def test_composite_no_sigalrm_platform(self):
        """Line 187: Platform without SIGALRM."""
        from bssunfold.core.unfold_composite import _run_with_timeout
        from unittest.mock import patch

        def fn():
            return "completed"

        # Remove SIGALRM temporarily
        orig = getattr(signal, 'SIGALRM', None)
        try:
            if hasattr(signal, 'SIGALRM'):
                delattr(signal, 'SIGALRM')
            result = _run_with_timeout(fn, timeout=1)
        finally:
            if orig is not None:
                signal.SIGALRM = orig
        assert result == "completed"


# ============================================================================
# platform_check.py — lines 50, 52-53, 119-120, 137, 139-140
# ============================================================================


class TestPlatformCheckCoverage:
    """Cover optional dependency checks."""

    def test_check_jax_available(self):
        """Lines 50, 52-53: JAX check."""
        from bssunfold.platform_check import check_jax_availability
        result = check_jax_availability()
        assert isinstance(result, bool)

    def test_check_scip_available(self):
        """Lines 119-120: SCIP check."""
        from bssunfold.platform_check import check_scip_availability
        result = check_scip_availability()
        assert isinstance(result, bool)

    def test_check_docplex_available(self):
        """Lines 137, 139-140: Docplex check."""
        from bssunfold.platform_check import check_docplex_availability
        result = check_docplex_availability()
        assert isinstance(result, bool)


# ============================================================================
# unfold_gravel.py — line 54
# ============================================================================


class TestGravelCoverage:
    """Cover error paths in gravel."""

    def test_gravel_all_zero_b_raises(self):
        """Line 54: raise ValueError when all b <= 0."""
        from bssunfold.core.unfold_gravel import solve_gravel

        A = np.eye(3)
        b = np.array([-1.0, -2.0, -3.0])
        x0 = np.ones(3)
        with pytest.raises(ValueError, match="zero or negative"):
            solve_gravel(A, b, x0, max_iterations=5)
