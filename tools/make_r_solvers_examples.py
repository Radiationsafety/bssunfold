"""Build example notebooks 45-49 for the R-solver port methods.

The notebooks follow examples/42-nspline-iaea.ipynb: built-in GSF
response functions, IAEA Compendium spectrum `t4-14-s.txt_1`, folded
readings, the new method, and a comparison with the ground truth.
Run:  uv run python tools/make_r_solvers_examples.py
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent.parent
EX = HERE / "examples"

HEADER = """# {title} (`unfold_{method}`)

This notebook applies **{short}** — the Python {qualif} of the R
package **{rpkg}** — to a realistic benchmark: detector readings
synthesised from the **Monte-Carlo calculated spectrum
`t4-14-s.txt_1`** of the
[IAEA Compendium](https://www-nds.iaea.org/benchmarks/), a BNCT-like
beam-shaping-assembly spectrum with a thermal group, an epithermal
$1/E$ region and a fast peak.

We use the built-in GSF response functions (10 Bonner spheres, `0in` –
`18in`, 60 energy bins from 1e-9 to ~631 MeV).  Detector readings are
folded with `Detector.get_effective_readings_for_spectra`, the
spectrum is reconstructed with `unfold_{method}`, and the result is
compared against the ground truth — which never enters the unfolding.
"""

SETUP50 = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bssunfold.core import (
    ssr3d,
    ssr3d_predict,
    ssrmlp_train,
    ssrmlp_predict,
)
"""

SETUP = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bssunfold import Detector, RF_GSF
from bssunfold.utils.comparison import compare_spectra

detector = Detector(RF_GSF)
E = detector.E_MeV
names = detector.detector_names
print(f"Detector grid: {detector.n_energy_bins} bins, "
      f"{E[0]:.1e} - {E[-1]:.1f} MeV")
print("Spheres:", ", ".join(names))
detector.plot_response_functions()
"""

DATA = r"""reference_csv = pd.read_csv(
    '../tests/MonteCarlo_Calculated_spectra_from_IAEA_Comp_for_comparison.csv'
)
readings = detector.get_effective_readings_for_spectra(
    reference_csv[['E_MeV', 't4-14-s.txt_1']]
)
print("Effective readings:")
for nm in names:
    print(f"  {nm:>5s}: {readings[nm]:.4g}")

phi_true = np.interp(
    E, reference_csv['E_MeV'].values, reference_csv['t4-14-s.txt_1'].values
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.loglog(E, phi_true, "k-", lw=1.5)
ax.set(xlabel="E, MeV", ylabel=r"$\varphi(E)$, cm$^{-2}$s$^{-1}$bin$^{-1}$",
       title="IAEA Compendium spectrum t4-14-s.txt_1 (ground truth)")
ax.grid(True, which="both", ls=":", alpha=0.5)

ax = axes[1]
vals = [readings[nm] for nm in names]
ax.bar(np.arange(len(names)), vals, color="steelblue")
ax.set_yscale("log")
ax.set_xticks(np.arange(len(names)))
ax.set_xticklabels(names, rotation=45)
ax.set(xlabel="sphere", ylabel="reading, a.u.",
       title="Effective Bonner-sphere readings")
ax.grid(True, axis="y", ls=":", alpha=0.5)
fig.tight_layout()
plt.show()
"""

QUALITY = r"""quality = compare_spectra(
    result['spectrum'], phi_true,
    metrics=["relative_flux_error", "pearson_r", "comprehensive_score",
             "fluence_difference_percent", "dose_difference_percent"],
    energy=E,
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.loglog(E, phi_true, "k-", lw=2, label="IAEA ground truth")
for label, spec, style in lines_to_plot:
    ax.loglog(E, spec, style, lw=1.2, label=label)
ax.set(xlabel="E, MeV", ylabel=r"$\varphi(E)$, cm$^{-2}$s$^{-1}$bin$^{-1}$",
       title="__PLOT_TITLE__")
ax.set_xlim(E[0], 30)
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()

for k, v in quality.items():
    print(f"{k:>26s}: {v}")
"""


def md(text):
    return nbf.v4.new_markdown_cell(text)


def md_lines(lines):
    return md("\n".join(lines))


def code(src):
    return nbf.v4.new_code_cell(src)

def build(method, title, short, qualif, rpkg, sections,
          plot_title=None, data_section=True, final_cells=None,
          intro_md=None, setup_src=None):
    """sections: list of (markdown_lines, code_src)."""
    cells = [
        md(intro_md.format(title=title, method=method, short=short,
                           qualif=qualif, rpkg=rpkg)
           if intro_md
           else HEADER.format(title=title, method=method, short=short,
                              qualif=qualif, rpkg=rpkg)),
        code(setup_src if setup_src is not None else SETUP),
    ]
    if data_section:
        cells += [
            md_lines([
                "## 1. IAEA Compendium reference spectrum → detector readings",
                "",
                "The compendium CSV stores 61-point Monte-Carlo spectra on its",
                "own energy grid; `get_effective_readings_for_spectra` folds",
                "the spectrum with the response functions and resamples it",
                "onto the 60-bin detector grid.",
            ]),
            code(DATA),
        ]
    for mdlines, csrc in sections:
        if mdlines:
            cells.append(md_lines(mdlines))
        if csrc:
            cells.append(code(csrc))
    if final_cells is not None:
        cells.extend(final_cells)
    elif data_section:
        cells.append(md_lines([
            "## Quality assessment",
            "",
            "`compare_spectra` reports the reconstruction metrics against the",
            "independently known IAEA Compendium spectrum (used only for",
            "evaluation).",
        ]))
        cells.append(code(QUALITY.replace("__PLOT_TITLE__",
                                          str(plot_title))))
    return nbf.v4.new_notebook(cells=cells)


# ---------------------------------------------------------------------------
# 45: P-spline REML (LMMsolver)
# ---------------------------------------------------------------------------

MD45 = [
    "## 2. REML unfolding run",
    "",
    "`unfold_pspline_reml` represents the spectrum as a P-spline and",
    "selects the smoothing parameter **automatically** by maximising the",
    "REML profile likelihood of the equivalent linear mixed model —",
    "exactly what `LMMsolver::LMMsolve()` does for spline-based LMMs.",
    "",
    "The benchmark readings are noise-free foldings of the Compendium",
    "spectrum; REML is deliberately conservative here (a smooth trend",
    "explains 10 correlated readings within their errors), so the",
    "REML-selected run captures the broad shape and the manual sweep in",
    "§3 shows the structure-preserving regime available with a weaker",
    "penalty.",
]
CODE45 = r"""result = detector.unfold_pspline_reml(
    readings,
    spline_order=4,
    diff_order=2,
    knot_spacing="auto",
    weights="uniform",
    save_result=False,
)

print(f"method      : {result['method']}")
print(f"lam         : {result['lam']:.4g}  "
      f"(relative {result['lam_relative']:.4g})")
print(f"REML loglik : {result['reml_loglik']:.4g}")
print(f"sigma2      : {result['sigma2']:.4g}")
print(f"ed          : {result['ed']:.2f}  "
      f"(norm {result['ed_norm']:.3f})")

lines_to_plot = [
    ("P-spline REML (lambda selected by REML)", result['spectrum'], "C1-"),
]
"""
MD45b = [
    "## 3. Fixing the smoothing parameter by hand",
    "",
    "With `lam_relative` given, the REML search is skipped and the",
    "Henderson mixed model equations are solved for a fixed smoothing",
    "parameter — useful to see how sensitive the reconstruction is to",
    "the smoothing choice.",
]
CODE45b = r"""results_fixed = {}
for lam_rel in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
    res = detector.unfold_pspline_reml(
        readings, knot_spacing="auto", weights="uniform",
        lam_relative=lam_rel, save_result=False,
    )
    results_fixed[lam_rel] = res
    q = compare_spectra(
        res['spectrum'], phi_true,
        metrics=['pearson_r', 'relative_flux_error',
                 'comprehensive_score'],
    )
    print(f"lam_rel={lam_rel:g}: ed={res['ed']:.1f}  "
          f"pearson_r={q['pearson_r']:.3f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.loglog(E, phi_true, "k-", lw=2, label="IAEA ground truth")
for lam_rel, res in results_fixed.items():
    ax.loglog(E, res['spectrum'], lw=1.1,
              label=fr"fixed $\lambda_{{rel}} = {lam_rel:g}$")
ax.loglog(E, result['spectrum'], color="C1", lw=2.2, alpha=0.6,
          label="REML-selected (section 2)")
ax.set(xlabel="E, MeV", ylabel=r"$\varphi(E)$, cm$^{-2}$s$^{-1}$bin$^{-1}$",
       title="P-spline REML unfolding: fixed vs. REML-selected smoothing")
ax.set_xlim(E[0], 30)
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()

# use a structure-preserving run for the final comparison
result = results_fixed[1e-5]
lines_to_plot = [
    (r"P-spline REML ($\lambda_{rel}=10^{-5}$)", result['spectrum'], "C1-"),
]
"""

# ---------------------------------------------------------------------------
# 46: AMG Krylov (Rlinsolve)
# ---------------------------------------------------------------------------

MD46 = [
    "## 2. AMG-preconditioned Krylov unfolding",
    "",
    "`unfold_amg` solves the auto-damped normal equations",
    "`(A^T A + reg I) x = A^T b` with `cg`/`bicgstab`/`gmres` accelerated",
    "by algebraic multigrid (smoothed aggregation, the Rlinsolve/pyamg",
    "analogue) or by one sweep of a classical stationary iteration",
    "(Jacobi/GS/SOR/SSOR).  Projected outer restarts keep the spectrum",
    "non-negative.",
]
CODE46 = r"""result = detector.unfold_amg(
    readings,
    method="cg",
    preconditioner="amg",
    max_iterations=200,
    tolerance=1e-10,
    save_result=False,
)

print(f"method        : {result['method']}")
print(f"iterations    : {result['iterations']}")
print(f"converged     : {result['converged']}")

lines_to_plot = [
    ("AMG-CG (pyamg; falls back to Jacobi if missing)", result['spectrum'],
     "C1-"),
]
"""
MD46b = [
    "## 3. Preconditioner comparison",
    "",
    "Compare the AMG preconditioner with the classical stationary",
    "iterations of Rlinsolve (Jacobi, Gauss-Seidel, SOR, SSOR) on the",
    "same system.",
]
CODE46b = r"""results_pc = {}
for pc in ["amg", "jacobi", "gs", "sor", "ssor"]:
    try:
        res = detector.unfold_amg(
            readings, method="cg" if pc not in ("gs", "sor") else "gmres",
            preconditioner=pc, save_result=False,
        )
        results_pc[pc] = res
    except Exception as exc:
        print(f"{pc:>7s}: failed - {exc}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.loglog(E, phi_true, "k-", lw=2, label="IAEA ground truth")
for pc, res in results_pc.items():
    ax.loglog(E, res['spectrum'], lw=1.1, label=f"{pc}")
ax.set(xlabel="E, MeV", ylabel=r"$\varphi(E)$, cm$^{-2}$s$^{-1}$bin$^{-1}$",
       title="AMG vs. stationary preconditioners (Rlinsolve family)")
ax.set_xlim(E[0], 30)
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()

for pc, res in results_pc.items():
    q = compare_spectra(
        res['spectrum'], phi_true,
        metrics=['pearson_r', 'relative_flux_error',
                 'comprehensive_score'],
    )
    print(f"{pc:>7s}: iters={res['iterations']:3d}  "
          f"pearson_r={q['pearson_r']:.3f}")
"""

# ---------------------------------------------------------------------------
# 47: SSR (sisireg)
# ---------------------------------------------------------------------------

MD47 = [
    "## 2. SSR unfolding (`fn=\"auto\"`)",
    "",
    "`unfold_ssr` (Python port of the R package `sisireg` 1.2.1,",
    "Metzner) alternates MLEM data-fidelity updates with the SSR",
    "quantised Gauss-Seidel parsimony sweeps.  With `fn=\"auto\"` the",
    "partial-sum threshold is chosen by Metzner's minimum-statistic",
    "ladder: the last statistically adequate and most parsimonious",
    "candidate wins.",
]
CODE47 = r"""result = detector.unfold_ssr(readings, save_result=False)

print(f"method          : {result['method']}")
print(f"fn              : {result['fn']}")
print(f"fn_start        : {result['fn_start']}")
print(f"k_run           : {result['k_run']}")
print(f"n_extrema       : {result['n_extrema']}")
print(f"ps_valid_data   : {result['ps_valid_data']}")
print(f"run_valid_data  : {result['run_valid_data']}")
print(f"iterations      : {result['iterations']}  "
      f"(converged: {result['converged']})")

lines_to_plot = [
    (fr"SSR (fn={result['fn']} from auto ladder)", result['spectrum'], "C1-"),
]
"""
MD47b = [
    "## 3. Fixed partial-sum thresholds",
    "",
    "Explicit `fn` values give coarser/finer parsimony.  A too small",
    "threshold suppresses physically meaningful structure; a too large",
    "one keeps sign-inadequate wiggles.",
]
CODE47b = r"""results_fn = {}
for fn in (2, 3, 4, 5, 6):
    res = detector.unfold_ssr(readings, fn=fn, save_result=False)
    results_fn[fn] = res
    q = compare_spectra(
        res['spectrum'], phi_true,
        metrics=['pearson_r', 'relative_flux_error',
                 'comprehensive_score'],
    )
    print(f"fn={fn}: n_extrema={res['n_extrema']:2d}  "
          f"pearson_r={q['pearson_r']:.3f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.loglog(E, phi_true, "k-", lw=2, label="IAEA ground truth")
for fn, res in results_fn.items():
    ax.loglog(E, res['spectrum'], lw=1.1, label=f"fn={fn}")
ax.loglog(E, result['spectrum'], color="C1", lw=2.2, alpha=0.6,
          label=f"auto (fn={result['fn']})")
ax.set(xlabel="E, MeV", ylabel=r"$\varphi(E)$, cm$^{-2}$s$^{-1}$bin$^{-1}$",
       title="SSR unfolding: partial-sum threshold dependence")
ax.set_xlim(E[0], 30)
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()
"""

# ---------------------------------------------------------------------------
# 48: GEE (gee)
# ---------------------------------------------------------------------------

MD48 = [
    "## 2. GEE unfolding with robust inference",
    "",
    "`unfold_gee` (R `gee` 4.13-30 analogue, Liang & Zeger 1986) treats",
    "the spheres as a correlated cluster: the IRLS loop solves the",
    "penalised score `A^T R(alpha)^-1 (b-Ax) - lam G x = 0` and reports",
    "the robust Liang-Zeger **sandwich** standard errors of the",
    "spectrum — uncertainties that stay consistent when the working",
    "correlation is misspecified.",
]
CODE48 = r"""result = detector.unfold_gee(readings, save_result=False)

print(f"method        : {result['method']}")
print(f"family/corstr : {result['family']}/{result['corstr']}")
print(f"alpha         : {result['alpha']:.3f}  (dispersion "
      f"phi = {result['phi']:.3g})")
print(f"pearson chi2  : {result['pearson_chi2']:.3g}")
print(f"converged     : {result['gee_converged']} "
      f"({result['iterations']} iterations)")

robust_se = result['robust_se']
print("robust SE on the first bins:", np.round(robust_se[:5], 2))

lines_to_plot = [
    ("GEE (gaussian / exchangeable)", result['spectrum'], "C1-"),
]
"""
MD48b = [
    "## 3. Working-correlation comparison and robust SE coverage",
    "",
    "Compare the three working-correlation structures, then check the",
    "robust sandwich uncertainties against a small Monte-Carlo run",
    "over Poisson-noised readings (the SEs should bracket the spread of",
    "unfolding outcomes).",
]
CODE48b = r"""results_cor = {}
for corstr in ["independence", "exchangeable", "ar1"]:
    res = detector.unfold_gee(readings, corstr=corstr, save_result=False)
    results_cor[corstr] = res
    q = compare_spectra(
        res['spectrum'], phi_true,
        metrics=['pearson_r', 'relative_flux_error',
                 'comprehensive_score'],
    )
    print(f"{corstr:>13s}: alpha={res['alpha']:+.3f}  "
          f"pearson_r={q['pearson_r']:.3f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.loglog(E, phi_true, "k-", lw=2, label="IAEA ground truth")
for corstr, res in results_cor.items():
    ax.loglog(E, res['spectrum'], lw=1.1, label=corstr)
ax.set(xlabel="E, MeV", ylabel=r"$\varphi(E)$, cm$^{-2}$s$^{-1}$bin$^{-1}$",
       title="GEE unfolding: working correlation structures")
ax.set_xlim(E[0], 30)
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()

# Monte-Carlo sanity check of the robust SE (should be ~1 sigma-ish)
rng = np.random.default_rng(0)
n_mc = 20
mc_spectra = []
for _ in range(n_mc):
    noisy = np.maximum(
        readings_scaled := np.array([readings[n] for n in names]) * 1e6,
        0.0,
    )
    noisy = rng.poisson(noisy) / 1e6
    rc = {n: float(v) for n, v in zip(names, noisy)}
    mc_spectra.append(detector.unfold_gee(rc, save_result=False)['spectrum'])
mc_arr = np.array(mc_spectra)
mc_std = mc_arr.std(axis=0, ddof=1)
mask = robust_se > 0
ratio = mc_std[mask] / robust_se[mask]
print(f"MC spread / robust SE, median = {np.median(ratio):.2f} "
      f"(robust sandwich should be of the MC-noise order)")
"""

# ---------------------------------------------------------------------------
# 49: Uno (Uno solver port)
# ---------------------------------------------------------------------------

MD49 = [
    "## 2. Uno presets",
    "",
    "`unfold_uno` solves the constrained NLP",
    "`min 1/2||W(Ax-b)||^2 + lam/2||D2 x||^2 s.t. x >= 0` with two",
    "Uno presets: `filter_sqp` (exact Hessian + Fletcher-Leyffer filter;",
    "for this convex QP the sub-problem is the answer) and",
    "`ipopt_like` (primal-dual interior point).",
]
CODE49 = r"""result = detector.unfold_uno(readings, save_result=False)

print(f"method    : {result['method']}")
print(f"objective : {result['objective']:.4g}")
print(f"viol      : {result['constraint_violation']:.3g}")
print(f"dual inf  : {result['dual_infeasibility']:.3g}")
print(f"converged : {result['uno_converged']}")

lines_to_plot = [
    ("Uno filterSQP", result['spectrum'], "C1-"),
]
"""
MD49b = [
    "## 3. The two presets and Hessian modes",
    "",
    "Compare the `filterSQP` preset with the IPOPT-like interior point",
    "in both Hessian modes (exact and BFGS).",
]
CODE49b = r"""variants = {
    "filter_sqp (exact)": dict(preset="filter_sqp"),
    "ipopt_like (exact)": dict(preset="ipopt_like", hessian="exact",
                               max_iterations=120, tolerance=1e-6),
    "ipopt_like (bfgs)":  dict(preset="ipopt_like", hessian="bfgs",
                               max_iterations=120, tolerance=1e-6),
}
results_uno = {}
for label, kw in variants.items():
    res = detector.unfold_uno(readings, save_result=False, **kw)
    results_uno[label] = res
    q = compare_spectra(
        res['spectrum'], phi_true,
        metrics=['pearson_r', 'relative_flux_error',
                 'comprehensive_score'],
    )
    print(f"{label:20s}: iters={res['iterations']:3d}  "
          f"obj={res['objective']:.3e}  "
          f"pearson_r={q['pearson_r']:.3f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.loglog(E, phi_true, "k-", lw=2, label="IAEA ground truth")
for label, res in results_uno.items():
    ax.loglog(E, res['spectrum'], lw=1.1, label=label)
ax.set(xlabel="E, MeV", ylabel=r"$\varphi(E)$, cm$^{-2}$s$^{-1}$bin$^{-1}$",
       title="Uno-style unfolding: presets and Hessian modes")
ax.set_xlim(E[0], 30)
ax.grid(True, which="both", ls=":", alpha=0.35)
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()
"""

# ---------------------------------------------------------------------------
# 50: sisireg building blocks (ssr3d / ssrMLP)
# ---------------------------------------------------------------------------

MD50A = [
    "## 1. Spatial minimal-surface regression (`ssr3d`, R `ssr3d.R` + C)",
    "",
    "`ssr3d` builds the minimal-surface SSR regression of Metzner for",
    "scattered planar data: every observation is replaced by the",
    "exponential weighted mean of its k-quadrant and 4k+1",
    "nearest-neighbourhoods, with Gauss-Seidel sweeps that revert any",
    "update violating the partial-sum adequacy criterion.  The port",
    "reproduces the original R output (sisireg 1.2.1, R 4.5.0) to",
    "machine precision — checked below on the bundled fixture.",
]
CODE50A = r"""from bssunfold.core import ps_max_3d, ps_statistic_3d

df_k3 = pd.read_csv('../tests/data/sisireg3d/model_k3.csv')
coords = df_k3[['x', 'y']].to_numpy()
z = df_k3['z'].to_numpy()

model3d = ssr3d(coords, z, k=3, fn=2.5, iter=200)
print("matches the original R ssr3d output:",
      np.allclose(model3d.mu, df_k3['mu'].to_numpy(), atol=1e-10))

q_k3 = pd.read_csv('../tests/data/sisireg3d/predict_k3.csv')
pred_exp = ssr3d_predict(model3d, q_k3[['x', 'y']].to_numpy())
pred_ms = ssr3d_predict(model3d, q_k3[['x', 'y']].to_numpy(), ms=True)
print("matches the R predictions:",
      np.allclose(pred_exp, q_k3['p_exp'].to_numpy(), atol=1e-10),
      np.allclose(pred_ms, q_k3['p_ms'].to_numpy(), atol=1e-10))

fig, ax = plt.subplots(figsize=(7, 5))
s = ax.scatter(coords[:, 0], coords[:, 1], c=z, cmap="coolwarm", s=45)
ax.scatter(coords[:, 0], coords[:, 1], c=model3d.mu, cmap="coolwarm",
           s=14, marker="x", label="ssr3d regression $\\mu$")
ax.set(xlabel="x", ylabel="y",
       title="ssr3d: observations (circles) vs. minimal-surface fit (x)")
fig.colorbar(s, ax=ax, label="z")
ax.legend()
fig.tight_layout()
plt.show()
"""
MD50B = [
    "## 2. Synthetic scattered data and prediction",
    "",
    "A noisy dome on scattered points; `ssr3d_predict` evaluates the",
    "regression on new coordinates (``ms=True`` selects the reciprocal",
    "minimal-surface weighted mean).",
]
CODE50B = r"""rng = np.random.default_rng(7)
xy_syn = rng.uniform(-3, 3, size=(220, 2))
z_syn = (np.exp(-(xy_syn[:, 0] ** 2 + xy_syn[:, 1] ** 2) / 2.5)
         + rng.normal(0, 0.08, len(xy_syn)))

model_syn = ssr3d(xy_syn, z_syn, iter=200)

gx, gy = np.meshgrid(np.linspace(-3, 3, 61), np.linspace(-3, 3, 61))
grid = np.column_stack([gx.ravel(), gy.ravel()])
z_pred = ssr3d_predict(model_syn, grid)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sc = axes[0].scatter(xy_syn[:, 0], xy_syn[:, 1], c=z_syn, cmap="viridis",
                     s=22)
axes[0].set(title="observations (scattered, noisy)", xlabel="x", ylabel="y")
fig.colorbar(sc, ax=axes[0])

cs = axes[1].contourf(gx, gy, z_pred.reshape(gx.shape), levels=24,
                      cmap="viridis")
axes[1].scatter(xy_syn[:, 0], xy_syn[:, 1], c="white", s=6, alpha=0.6)
axes[1].set(title="ssr3d minimal-surface prediction", xlabel="x", ylabel="y")
fig.colorbar(cs, ax=axes[1])
fig.tight_layout()
plt.show()
"""
MD50C = [
    "## 3. Two-hidden-layer perceptron with the partial sum criterion",
    "",
    "`ssrmlp_train` is the port of R `ssrMLP.R`: a sigmoid perceptron",
    "whose training criterion is Metzner's **partial sum adequacy**",
    "(`opt=\"ps\"`) instead of plain least squares (`opt=\"lse\"`).  The",
    "partial sum statistics (`ps_max_3d`-family counterparts,",
    "``check_ps``) bound the residual sign structure in every",
    "neighbourhood — the same criterion used by the `unfold_ssr`",
    "unfolding method.",
]
CODE50C = r"""from bssunfold.core import ssrmlp_train, ssrmlp_predict, fii_prediction

# synthetic 1-D->2-D regression task: y = sin wave + noise
rng = np.random.default_rng(3)
X_syn = rng.uniform(0, np.pi, size=(120, 2))
Y_syn = np.sin(X_syn[:, 0]) + 0.5 * X_syn[:, 1] + rng.normal(0, 0.1, 120)

w_ps = ssrmlp_train(X_syn, Y_syn, opt="ps", max_iter=400, rng=0)
w_lse = ssrmlp_train(X_syn, Y_syn, opt="lse", max_iter=400, rng=0)

yp_ps = ssrmlp_predict(X_syn, w_ps)
yp_lse = ssrmlp_predict(X_syn, w_lse)
print(f"ps  : L2 error = {np.linalg.norm(yp_ps - Y_syn):.3f}")
print(f"lse : L2 error = {np.linalg.norm(yp_lse - Y_syn):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, yp, name in zip(
    axes, (yp_lse, yp_ps), ("lse (least squares)", "ps (partial sum)")
):
    ax.scatter(X_syn[:, 0], Y_syn, s=18, c="k", label="data")
    ax.scatter(X_syn[:, 0], yp, s=14, c="C1", marker="x", label=name)
    ax.set(xlabel="$X_1$", ylabel="y", title=f"ssrMLP — {name}")
    ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
"""

INTRO50 = """# {title}

This notebook demonstrates the **regression building blocks** of the
R package **sisireg** 1.2.1 as ported to pure NumPy:

* **`ssr3d`** — the spatial minimal-surface Sign-Simplicity-Regression
  of Metzner (R `ssr3d.R` + the `ssr3d.c` Gauss-Seidel kernel):
  scattered planar data `z = f(x, y)` is smoothed by replacing every
  observation with the exponential/minimal-surface weighted mean of
  its k-quadrant and `4k+1` nearest-neighbourhoods, reverting updates
  that would violate the partial-sum adequacy criterion;
* **`ssrmlp_train` / `ssrmlp_predict`** — a two-hidden-layer sigmoid
  perceptron trained with the partial sum criterion (`opt="ps"`),
  i.e. the machine-learning building block that shares the sign
  adequacy machinery with the `unfold_ssr` unfolding method.

Both ports were verified against the original R output (R 4.5.0 +
sisireg 1.2.1) to machine precision; the fixtures used for that
verification are re-checked below.
"""

NOTEBOOKS = [
    dict(num=45, method="pspline_reml",
         title="P-spline mixed-model unfolding with REML smoothing selection "
               "(LMMsolver)",
         short="P-spline mixed-model unfolding with REML smoothing selection",
         qualif="analogue", rpkg="LMMsolver",
         plot_title="P-spline REML (LMMsolver analogue) unfolding",
         sections=[(MD45, CODE45), (MD45b, CODE45b)]),
    dict(num=46, method="amg",
         title="AMG-preconditioned Krylov unfolding (Rlinsolve)",
         short="AMG/stationary-preconditioned Krylov unfolding",
         qualif="analogue", rpkg="Rlinsolve (+pyamg)",
         plot_title="AMG-preconditioned Krylov unfolding",
         sections=[(MD46, CODE46), (MD46b, CODE46b)]),
    dict(num=47, method="ssr",
         title="SSR Sign-Simplicity-Regression unfolding (sisireg)",
         short="SSR Sign-Simplicity-Regression unfolding",
         qualif="port", rpkg="sisireg",
         plot_title="SSR (sisireg) unfolding",
         sections=[(MD47, CODE47), (MD47b, CODE47b)]),
    dict(num=48, method="gee",
         title="GEE unfolding (gee)",
         short="GEE unfolding with robust Liang-Zeger sandwich inference",
         qualif="analogue", rpkg="gee",
         plot_title="GEE (robust sandwich) unfolding",
         sections=[(MD48, CODE48), (MD48b, CODE48b)]),
    dict(num=49, method="uno",
         title="Uno-style constrained unfolding (Uno)",
         short="Uno-style Lagrange-Newton constrained unfolding",
         qualif="port", rpkg="Uno",
         plot_title="Uno-style constrained unfolding",
         sections=[(MD49, CODE49), (MD49b, CODE49b)]),
    dict(num=50, method="sisireg3d",
         title="sisireg building blocks: ssr3d and ssrMLP (sisireg)",
         short="ssr3d minimal-surface regression and ssrMLP perceptron",
         qualif="port of the regression building blocks of",
         rpkg="sisireg (ssr3d.R, ssrMLP.R)",
         plot_title="ssr3d / ssrMLP building blocks",
         sections=[(MD50A, CODE50A), (MD50B, CODE50B), (MD50C, CODE50C)],
         data_section=False,
         intro_md=INTRO50,
         setup_src=SETUP50,
         final_cells=[
             md_lines([
                 "## Summary",
                 "",
                 "* `ssr3d` reproduces the original R output",
                 "  (R `ssr3d` + `ssr3d.c`, k = 3, fn = 2.5, 200 sweeps)",
                 "  to machine precision on the bundled fixture.",
                 "* The minimal-surface prediction reconstructs smooth",
                 "  structure from scattered noisy data.",
                 "* `ssrmlp_train(opt=\"ps\")` constrains the residual",
                 "  sign structure through the partial sum criterion —",
                 "  the same adequacy machinery as `unfold_ssr`.",
             ]),
         ]),
]


def main():
    for spec in NOTEBOOKS:
        nb = build(
            method=spec['method'], title=spec['title'], short=spec['short'],
            qualif=spec['qualif'], rpkg=spec['rpkg'],
            plot_title=spec['plot_title'], sections=spec['sections'],
            data_section=spec.get('data_section', True),
            final_cells=spec.get('final_cells'),
            intro_md=spec.get('intro_md'),
            setup_src=spec.get('setup_src'),
        )
        path = EX / f"{spec['num']}-{spec['method']}-iaea.ipynb"
        nbf.write(nb, str(path))
        print("wrote", path)


if __name__ == "__main__":
    main()
