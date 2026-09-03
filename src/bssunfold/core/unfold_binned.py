"""Bin-wise adaptive spectrum unfolding.

For each of the 60 energy bins, the best unfolding method is selected from a
pre-computed benchmark lookup table, and the final spectrum is assembled by
picking the winning method's value at every bin.  This exploits the empirical
observation that different unfolding methods excel in different energy regions.

References
----------
- The grid-search benchmark evaluated 67 methods
  on 271 reference spectra across 41 quality metrics.
- The pre-computed lookup is built by ``tools/build_bin_lookup.py`` and ships
  as ``data/bin_lookup.json`` inside the package.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..logging_config import get_logger

logger = get_logger("binned")

# ── Default lookup path (ships with the package) ──────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_DEFAULT_LOOKUP = _DATA_DIR / "bin_lookup.json"

# Mapping short name -> Detector.unfold_* attribute.
METHOD_DISPATCH: Dict[str, str] = {
    "tsvd": "unfold_tsvd",
    "bayes": "unfold_bayes",
    "cvxpy": "unfold_cvxpy",
    "statreg": "unfold_statreg",
    "lanczos": "unfold_lanczos",
    "mlem": "unfold_mlem",
    "landweber": "unfold_landweber",
    "bayes_spline": "unfold_bayes_spline_regularization",
    "gravel": "unfold_gravel",
    "qpsolvers": "unfold_qpsolvers",
    "hybrid_parametric": "unfold_hybrid_parametric",
    "parametric2": "unfold_parametric2",
    "genetic": "unfold_genetic",
    "interpret": "unfold_interpret",
    "maeo_ensemble": "unfold_maeo",
    "mystic": "unfold_mystic",
    "mystic_hybrid": "unfold_mystic_hybrid",
    "cs": "unfold_cs",
    "scip": "unfold_scip",
    "docplex": "unfold_docplex",
    "epic": "unfold_epic",
    "kaczmarz": "unfold_kaczmarz",
    "sart": "unfold_sart",
    "osem": "unfold_osem",
    "bsrem": "unfold_bsrem",
    "mapem": "unfold_mapem",
    "ferdor": "unfold_ferdor",
    "rebunki": "unfold_rebunki",
    "nsduaz": "unfold_nsduaz",
    "doroshenko": "unfold_doroshenko",
    "sandii": "unfold_sandii",
    "bunki": "unfold_bunki",
    "bunkiut": "unfold_bunkiut",
    "reconst": "unfold_reconst",
    "amaxed": "unfold_amaxed",
    "amaxed_regularization": "unfold_amaxed_regularization",
    "imaxed": "unfold_imaxed",
    "maxed": "unfold_maxed",
    "mlem_odl": "unfold_mlem_odl",
    "mlem_stop": "unfold_mlem_stop",
    "cgls": "unfold_cgls",
    "gks": "unfold_gks",
    "hybrid_gmres": "unfold_hybrid_gmres",
    "tikhonov_legendre": "unfold_tikhonov_legendre",
    "tikhonov_tv": "unfold_tikhonov_tv",
    "fista": "unfold_fista",
    "crystal_ball": "unfold_crystal_ball",
    "rfsp_jul": "unfold_rfsp_jul",
    "staysl": "unfold_staysl",
    "parametric": "unfold_parametric",
    "parametric_cvxpy": "unfold_parametric",
    "parametric_qpsolvers": "unfold_parametric",
    "parametric_combined": "unfold_parametric",
    "lmfit": "unfold_lmfit",
    "lmfit_ic": "unfold_lmfit",
    "scipy_direct_method": "unfold_scipy_direct_method",
    "qubo": "unfold_qubo",
    "zfit": "unfold_zfit",
    "mcmc": "unfold_mcmc",
    "bayesian_parametric": "unfold_bayesian_parametric",
    "eki": "unfold_eki",
    "maeo": "unfold_maeo",
    "odl_pdhg": "unfold_odl_pdhg",
    "odl_douglas_rachford": "unfold_odl_douglas_rachford",
    "combined": "unfold_combined",
    "cascade": "unfold_cascade",
    "composite": "unfold_composite",
}

# Aliases that map to a base method with fixed extra params.
_ALIASES: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "unfold_parametric_cvxpy": ("unfold_parametric", {"optimizer": "cvxpy"}),
    "unfold_parametric_qpsolvers": ("unfold_parametric",
                                    {"optimizer": "qpsolvers"}),
    "unfold_parametric_combined": ("unfold_parametric",
                                   {"optimizer": "combined"}),
    "unfold_lmfit_ic": ("unfold_lmfit", {}),
}


# ── Timeout helpers (POSIX SIGALRM, same as unfold_composite) ─────────────

class _MethodTimeout(Exception):
    """Raised when an individual method exceeds its wall-clock timeout."""


def _timeout_handler(signum, frame):  # noqa: ANN001, ARG001
    raise _MethodTimeout()


def _run_with_timeout(fn, timeout: float):  # noqa: ANN001
    """Execute *fn* with a per-method wall-clock timeout.

    Uses ``SIGALRM`` on Unix.  On platforms without ``SIGALRM`` (e.g.
    Windows) a ``threading``-based fallback is used instead.
    """
    import signal
    import threading

    if timeout is None or timeout <= 0:
        return fn()

    if hasattr(signal, "SIGALRM"):
        old = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(np.ceil(timeout)))
        try:
            return fn()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    result: list = []
    exc: list = []

    def _target():
        try:
            result.append(fn())
        except Exception as e:
            exc.append(e)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise _MethodTimeout()
    if exc:
        raise exc[0]
    return result[0]


# ── Lookup I/O ────────────────────────────────────────────────────────────

def load_bin_lookup(path: str | Path | None = None) -> Dict[str, Any]:
    """Load a pre-computed bin lookup table from a JSON file.

    Parameters
    ----------
    path : str or Path, optional
        Path to the JSON file.  Falls back to the built-in lookup shipped
        with the package.

    Returns
    -------
    dict
        Keys: ``"bin_to_methods"`` (dict[int, list]),
        ``"unique_methods"`` (list[str]), ``"n_bins"`` (int).
    """
    if path is None:
        path = _DEFAULT_LOOKUP
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Bin lookup not found at {path}.  "
            "Run ``python tools/build_bin_lookup.py`` to generate it."
        )
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    raw["bin_to_methods"] = {
        int(k): v for k, v in raw["bin_to_methods"].items()
    }
    return raw


def save_bin_lookup(lookup: Dict[str, Any], path: str | Path) -> None:
    """Persist a bin lookup table to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "bin_to_methods": {
            str(k): v for k, v in lookup["bin_to_methods"].items()
        },
        "unique_methods": lookup["unique_methods"],
        "n_bins": lookup["n_bins"],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2, ensure_ascii=False)
    logger.info("Saved bin lookup -> %s", path)


# ── Building the lookup from benchmark data ───────────────────────────────

def build_bin_lookup(
    spectra_dir: str | Path,
    references_csv: str | Path,
    n_bins: int = 60,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Analyse benchmark unfolded spectra and build a per-bin method ranking.

    Parameters
    ----------
    spectra_dir : path
        Directory containing per-method ``.npz`` files (e.g. ``spectra/mc/``).
    references_csv : path
        CSV with reference spectra.  Must contain a ``key`` column (hash) and
        at least ``n_bins`` numeric energy columns (``energy_1`` … ``energy_60``
        or the first N numeric columns).
    n_bins : int
        Number of energy bins (default 60).
    top_k : int
        Number of best methods to keep per bin.

    Returns
    -------
    dict
        ``{"bin_to_methods": {bin_idx: [(method_short, score), ...]},
           "unique_methods": [...], "n_bins": n_bins}``
    """
    from pathlib import Path as _P

    spectra_dir = _P(spectra_dir)
    references_csv = _P(references_csv)

    # ── 1. Load reference spectra (key → 60-vector) ──────────────────────
    import pandas as _pd

    ref_df = _pd.read_csv(references_csv)
    energy_cols = [c for c in ref_df.columns if c.startswith("energy_")]
    if not energy_cols:
        numeric_cols = ref_df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) >= n_bins:
            energy_cols = numeric_cols[:n_bins]
        else:
            raise ValueError(
                f"Cannot locate {n_bins} energy columns in {references_csv}."
            )
    ref_keys = (
        ref_df["key"].values if "key" in ref_df.columns else ref_df.index
    )
    ref_spectra = {
        str(k): np.asarray(row, dtype=float)
        for k, row in zip(ref_keys, ref_df[energy_cols].values)
    }

    # ── 2. Load unfolded spectra per method ──────────────────────────────
    method_files = sorted(spectra_dir.glob("unfold_*.npz"))
    if not method_files:
        raise FileNotFoundError(f"No unfold_*.npz files in {spectra_dir}")

    method_data: Dict[str, Dict[str, np.ndarray]] = {}
    for fpath in method_files:
        method_short = fpath.stem.replace("unfold_", "")
        data = np.load(fpath, allow_pickle=True)
        method_data[method_short] = {
            str(k): np.asarray(data[k], dtype=float) for k in data.files
        }

    # ── 3. Compute per-bin mean absolute error ───────────────────────────
    methods = sorted(method_data.keys())
    bin_errors = np.full((len(methods), n_bins), np.nan)

    for m_idx, m_name in enumerate(methods):
        per_bin_accum = np.zeros(n_bins)
        count = 0
        for key, ref_spec in ref_spectra.items():
            if key in method_data[m_name]:
                unf = method_data[m_name][key]
                if unf.shape == (n_bins,) and np.all(np.isfinite(unf)):
                    per_bin_accum += np.abs(unf - ref_spec)
                    count += 1
        if count > 0:
            bin_errors[m_idx, :] = per_bin_accum / count

    # ── 4. Rank methods per bin, keep top_k ──────────────────────────────
    bin_to_methods: Dict[int, List[Tuple[str, float]]] = {}
    all_method_names: set = set()

    for b in range(n_bins):
        col = bin_errors[:, b]
        valid = np.where(np.isfinite(col))[0]
        if len(valid) == 0:
            bin_to_methods[b] = []
            continue
        order = valid[np.argsort(col[valid])]
        top = [(methods[i], float(col[i])) for i in order[:top_k]]
        bin_to_methods[b] = top
        all_method_names.update(m for m, _ in top)

    return {
        "bin_to_methods": bin_to_methods,
        "unique_methods": sorted(all_method_names),
        "n_bins": n_bins,
    }


# ── Core solver (low-level, operates on A, b) ─────────────────────────────

def solve_binned(
    A: np.ndarray,
    b: np.ndarray,
    bin_lookup: Dict[str, Any],
    methods: Dict[str, Tuple[callable, dict]],
    x0: Optional[np.ndarray] = None,
    timeout_per_method: float = 30.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run candidate methods and assemble a spectrum bin-by-bin.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Response matrix.
    b : ndarray, shape (m,)
        Measurement vector.
    bin_lookup : dict
        Pre-computed per-bin method ranking (from :func:`build_bin_lookup`).
    methods : dict
        Mapping ``method_short → (solver_callable, kwargs)``.  Each callable
        must accept ``(A, b, x0=..., **kwargs)`` and return a spectrum array
        of shape ``(n,)``.
    x0 : ndarray, optional
        Initial guess forwarded to every solver.
    timeout_per_method : float
        Wall-clock timeout per method (seconds).

    Returns
    -------
    spectrum : ndarray, shape (n,)
        Assembled spectrum.
    meta : dict
        Metadata including ``method_map``, ``successful_methods``, and
        per-method spectra.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n_bins = A.shape[1]
    bin_to_methods = bin_lookup["bin_to_methods"]

    candidate_names = list(bin_lookup.get("unique_methods", []))
    if not candidate_names:
        seen: set = set()
        for ranking in bin_to_methods.values():
            for name, _ in ranking:
                seen.add(name)
        candidate_names = sorted(seen)

    # ── Run each candidate ───────────────────────────────────────────────
    spectra: Dict[str, np.ndarray] = {}
    successes: List[str] = []
    errors: Dict[str, str] = {}

    for name in candidate_names:
        entry = methods.get(name)
        if entry is None:
            errors[name] = "not provided"
            continue
        solver_fn, kw = entry
        try:
            result = _run_with_timeout(
                partial(solver_fn, A, b, x0=x0, **kw),
                timeout_per_method,
            )
            spec = np.asarray(result, dtype=float)
            if (spec.shape == (n_bins,)
                    and np.all(np.isfinite(spec))
                    and np.sum(spec) > 0):
                spectra[name] = np.maximum(spec, 0.0)
                successes.append(name)
            else:
                errors[name] = "invalid output"
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"

    # ── Assemble spectrum bin-by-bin ─────────────────────────────────────
    assembled = np.zeros(n_bins)
    name_to_idx = {n: i for i, n in enumerate(candidate_names)}
    method_map = np.full(n_bins, -1, dtype=int)

    for b_idx in range(n_bins):
        ranking = bin_to_methods.get(b_idx, [])
        picked = False
        for method_name, _score in ranking:
            if method_name in spectra:
                assembled[b_idx] = spectra[method_name][b_idx]
                method_map[b_idx] = name_to_idx.get(method_name, -1)
                picked = True
                break
        if not picked:
            vals = [s[b_idx] for s in spectra.values()]
            assembled[b_idx] = float(np.median(vals)) if vals else 0.0

    meta = {
        "method_map": method_map,
        "candidate_methods": candidate_names,
        "successful_methods": successes,
        "individual_spectra": spectra,
        "errors": errors,
        "n_bins": n_bins,
    }
    return assembled, meta


# ── High-level Detector wrapper ───────────────────────────────────────────

def unfold_binned(
    detector,
    readings: Dict[str, float],
    bin_lookup: Optional[Dict[str, Any]] = None,
    lookup_path: Optional[str | Path] = None,
    timeout_per_method: float = 30.0,
    save_result: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Bin-wise adaptive unfolding: best method per energy bin.

    Parameters
    ----------
    detector : Detector
        Configured Bonner-sphere detector.
    readings : Dict[str, float]
        Detector readings.
    bin_lookup : dict, optional
        Pre-computed lookup table.  If *None*, loaded from *lookup_path*
        (or the built-in default).
    lookup_path : str or Path, optional
        Path to a JSON lookup file.  Ignored when *bin_lookup* is provided.
    timeout_per_method : float
        Wall-clock timeout per individual method (seconds).
    save_result : bool
        Persist result to detector history.

    Returns
    -------
    dict
        Standard bssunfold result dict with extra keys ``method_map``,
        ``successful_methods``, ``individual_spectra``.
    """
    # Load lookup table.
    if bin_lookup is None:
        bin_lookup = load_bin_lookup(lookup_path)

    n_bins = detector.n_energy_bins
    energy = detector.E_MeV
    selected = [n for n in detector.detector_names if n in readings]
    A = np.array([detector.sensitivities[n] for n in selected], dtype=float)
    b = np.array([readings[n] for n in selected], dtype=float)

    default_initial = np.ones(n_bins) / n_bins
    x0 = default_initial.copy()

    # Build solver callables for each candidate method.
    candidate_names = bin_lookup.get("unique_methods", [])
    solver_dict: Dict[str, Tuple[callable, dict]] = {}

    for name in candidate_names:
        dispatch_name = METHOD_DISPATCH.get(name)
        if dispatch_name is None:
            continue
        fn = getattr(detector, dispatch_name, None)
        if fn is None:
            continue

        # Resolve aliases.
        base_method, alias_fixed = _ALIASES.get(
            dispatch_name, (dispatch_name, {})
        )

        solver_dict[name] = (
            _make_detector_solver(fn, selected),
            {**alias_fixed, "save_result": False, "calculate_errors": False,
             "verbose": False},
        )

    assembled, meta = solve_binned(
        A, b,
        bin_lookup=bin_lookup,
        methods=solver_dict,
        x0=x0,
        timeout_per_method=timeout_per_method,
    )

    # Build standardised output.
    from .dose_calculation import calculate_dose_rates

    spectrum_nonneg = np.maximum(assembled, 0)
    computed_readings = A @ spectrum_nonneg
    residual = b - computed_readings

    result: Dict[str, Any] = {
        "energy": energy.copy(),
        "spectrum": spectrum_nonneg,
        "spectrum_absolute": spectrum_nonneg.copy(),
        "effective_readings": {
            name: float(val)
            for name, val in zip(selected, computed_readings)
        },
        "residual": residual,
        "residual_norm": float(np.linalg.norm(residual)),
        "method": "binned",
        "doserates": calculate_dose_rates(
            spectrum_nonneg, detector.cc_icrp116
        ),
        "method_map": meta["method_map"],
        "successful_methods": meta["successful_methods"],
        "individual_spectra": meta["individual_spectra"],
        "candidate_methods": meta["candidate_methods"],
        "bin_lookup": bin_lookup,
    }

    if save_result:
        detector._save_result(result)

    return result


def _make_detector_solver(method_fn, selected_detectors):
    """Create a solver callable compatible with solve_binned.

    The Detector methods accept ``readings`` as a dict, so we adapt the
    ``(A, b, x0)`` interface to the Detector-style interface.
    """

    def _solver(A_mat, b_vec, x0=None, **kwargs):
        readings_dict = {
            name: float(b_vec[i])
            for i, name in enumerate(selected_detectors)
        }
        call_kwargs = dict(kwargs)
        if x0 is not None:
            call_kwargs["initial_spectrum"] = x0
        try:
            import inspect
            sig = inspect.signature(method_fn)
            accepted = set(sig.parameters)
            call_kwargs = {
                k: v for k, v in call_kwargs.items() if k in accepted
            }
        except (TypeError, ValueError):
            pass
        res = method_fn(readings=readings_dict, **call_kwargs)
        if isinstance(res, dict):
            return np.asarray(res.get("spectrum", []), dtype=float)
        return np.asarray(res, dtype=float)

    return _solver


__all__ = [
    "METHOD_DISPATCH",
    "load_bin_lookup",
    "save_bin_lookup",
    "build_bin_lookup",
    "solve_binned",
    "unfold_binned",
]
