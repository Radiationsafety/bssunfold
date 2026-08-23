"""Shared pytest fixtures and helpers for the bssunfold test suite.

This conftest centralizes the patterns that were previously duplicated
across individual test modules:

- ``detector`` — the default :class:`~bssunfold.Detector` instance built
  from the packaged GSF response functions.  A test module may still define
  its own ``detector`` fixture; the local definition simply shadows this one.
- ``block_import`` — the documented recipe for simulating a missing optional
  dependency by patching ``builtins.__import__`` to raise ``ImportError`` for
  selected module names (see AGENTS.md).  Simply popping from ``sys.modules``
  does NOT work for packages such as pytikhonov because the re-import
  succeeds.
"""

from __future__ import annotations

import builtins
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch

import pytest

from bssunfold import Detector


@pytest.fixture
def detector() -> Detector:
    """Default Detector instance with default response functions."""
    return Detector()


@contextmanager
def block_import(*module_names: str) -> Iterator[None]:
    """Make importing the given top-level module names fail with ImportError.

    Patches ``builtins.__import__`` so that any import of *module_names*
    (including submodule imports like ``pkg.sub``) raises ``ImportError``.
    All other imports pass through untouched.

    Note
    ----
    For lazy-loading modules that cache their availability decision, reset
    the corresponding cache before/after entering the context, e.g.
    ``bssunfold.core.unfold_interpret._pyopt._loaded = None`` for pyoptexplain.

    Examples
    --------
    >>> with block_import("pytikhonov"):
    ...     detector.unfold_cvxpy(readings)   # exercises numpy fallback
    """
    names = tuple(module_names)
    original = builtins.__import__

    def _mock_import(name: str, *args, **kwargs):
        if name in names or name.startswith(tuple(f"{m}." for m in names)):
            raise ImportError(f"{names[0]} not installed (blocked in test)")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        yield
