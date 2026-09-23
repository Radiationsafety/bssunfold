# BSSunfold - Neutron Spectrum Unfolding Package for Bonner Sphere Spectrometers
[![PyPI - Version](https://img.shields.io/pypi/v/BSSUnfold)](https://pypi.org/project/bssunfold/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/bssunfold)](https://anaconda.org/conda-forge/bssunfold)
[![Python 3.11–3.15](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13%20|%203.14%20|%203.15-blue)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue)](https://bssunfold.readthedocs.io/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/7dd7cc75ab654b879b80abe8476907f6)](https://app.codacy.com/gh/Radiationsafety/bssunfold/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/7dd7cc75ab654b879b80abe8476907f6)](https://app.codacy.com/gh/Radiationsafety/bssunfold/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![CodeFactor](https://www.codefactor.io/repository/github/radiationsafety/bssunfold/badge/main)](https://www.codefactor.io/repository/github/radiationsafety/bssunfold/overview/main)
[![DOI](https://zenodo.org/badge/1122800086.svg)](https://doi.org/10.5281/zenodo.18056376)
[![Tests: Ubuntu](https://img.shields.io/github/actions/workflow/status/Radiationsafety/bssunfold/cross-platform-tests.yml?branch=main&label=ubuntu&logo=ubuntu)](https://github.com/Radiationsafety/bssunfold/actions/workflows/cross-platform-tests.yml)
[![Tests: Windows](https://img.shields.io/github/actions/workflow/status/Radiationsafety/bssunfold/cross-platform-tests.yml?branch=main&label=windows&logo=windows)](https://github.com/Radiationsafety/bssunfold/actions/workflows/cross-platform-tests.yml)
[![Tests: macOS](https://img.shields.io/github/actions/workflow/status/Radiationsafety/bssunfold/cross-platform-tests.yml?branch=main&label=macOS&logo=apple)](https://github.com/Radiationsafety/bssunfold/actions/workflows/cross-platform-tests.yml)

## 🔍 Overview

**BSSUnfold** is a Python package for neutron spectrum unfolding from measurements obtained with Bonner Sphere Spectrometers (BSS). The package implements several mathematical algorithms for solving the inverse problem of unfolding neutron energy spectra from detector readings, with applications in radiation protection, nuclear physics research, and accelerator facilities.

![logo](assets/bssunfold_logo.png)

**Contents**
- [Features](#-features)
- [Installation](#-installation)
- [Quick start](#-quick-start)
- [Available Unfolding Methods](#-available-unfolding-methods)
- [Spectrum Comparison](#-spectrum-comparison)
- [Technical requirements](#-technical-requirements)
- [Citation](#-citation)
- [Documentation](#-documentation)
- [Publications](#-publications)
- [AI Disclosure](#-ai-disclosure)

## 📦 Features

- **100+ unfolding algorithms** in one package
- **Maximum Energy Cutoff**: max_neutron_energy parameter on all methods — forces zero fluence above a user-specified energy. QP solvers receive the full response matrix with a ub array; iterative solvers use matrix trimming with automatic result expansion
- **Numba JIT acceleration** for iterative solvers (3–50x, graceful fallback)
- **Radiation dose calculations**: ICRP-116/ICRP-74/NRB99 effective dose, per irradiation geometry (AP/PA/LLAT/RLAT/ROT/ISO)
- **Uncertainty quantification**: Monte Carlo with optional variance reduction; Poisson/gaussian noise models, reading covariance
- **41 spectrum comparison metrics** (integral quantities + spectral diagnostics)
- **7 built-in response-function datasets** and 4 dose-conversion datasets
- **Visualization**: spectrum plotting with uncertainty bands, detector-reading comparison

Full method reference (per-method parameters, dependencies, descriptions):
[docs — Package Overview](https://bssunfold.readthedocs.io/en/latest/overview.html)

## 📥 Installation

### Using uv (recommended)
```bash
uv add bssunfold

# With ECOS solver (recommended for CVXPY-based methods)
uv add "bssunfold[ecos]"
```

### Using pip
```bash
pip install bssunfold

# With ECOS solver (recommended for CVXPY-based methods, uses pre-built wheels)
pip install "bssunfold[ecos]"
```

### Using conda
```bash
conda install conda-forge::bssunfold
```

### From Source
```bash
git clone https://github.com/radiationsafety/bssunfold.git
cd bssunfold
pip install -e .
```

### Optional dependencies

```bash
# Basic installation (without optional solvers that require compilation)
pip install bssunfold

# With ECOS solver (recommended for CVXPY-based methods, uses pre-built wheels)
pip install "bssunfold[ecos]"

# all methods
pip install "bssunfold[all]"

# With numba JIT acceleration (recommended for iterative solvers)
pip install "bssunfold[numba]"

# With additional cross-platform solvers (recommended)
pip install "bssunfold[solvers-core]"

# All solvers (Unix/Linux/macOS)
pip install "bssunfold[all-solvers]"

# Windows (all except proxsuite)
pip install "bssunfold[windows]"

# With QP interpretation via pyoptexplain
pip install "bssunfold[interpret]"

# With Bayesian MCMC unfolding (PyMC + ArviZ)
pip install "bssunfold[mcmc]"

# Other method-specific extras
pip install "bssunfold[maeo]"     # unfold_maeo (pymoo)
pip install "bssunfold[qubo]"     # unfold_qubo (pyqubo + dwave-neal)
pip install "bssunfold[zfit]"     # unfold_zfit (zfit + tensorflow)
pip install "bssunfold[cuqi]"     # unfold_cuqi (cuqipy)
pip install "bssunfold[amg]"      # unfold_amg (pyamg)

# Commercial QP engines — LICENSE REQUIRED (not distributed with bssunfold;
# you must hold a valid license for the engine, academic/community
# editions available for Gurobi/MOSEK/CPLEX/COPT/Xpress):
pip install "bssunfold[gurobi]"    # unfold_gurobi
pip install "bssunfold[mosek]"     # unfold_mosek
pip install "bssunfold[cplex]"     # unfold_cplex (cvxpy interface)
pip install "bssunfold[copt]"      # unfold_copt
pip install "bssunfold[xpress]"    # unfold_xpress
pip install "bssunfold[commercial]"  # all five engines
```

For Windows it is recommended to use the following command because of the
problem with proxsuite:
```bash
uv add bssunfold[windows]
```

## 🎯 Quick Start

Open in interactive notebooks:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Radiationsafety/bssunfold/blob/main/examples/01-basic-example.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Radiationsafety/bssunfold.git/HEAD?urlpath=%2Fdoc%2Ftree%2Fexamples%2F02-basic-example-for-mybinder.ipynb)

```python
import pandas as pd
from bssunfold import Detector

# Load response functions
rf_df = pd.read_csv("../data/response_functions/rf_GSF.csv")

# Initialize detector
detector = Detector(rf_df)

# Provide detector readings [reading per second]
readings = {
    "0in": 0.0003,
    "2in": 0.0099,
    "3in": 0.0536,
    "5in": 0.1841,
    "6in": 0.2196,
    "8in": 0.2200,
    "10in": 0.172,
    "12in": 0.120,
    "15in": 0.066,
    "18in": 0.034,
}

# Unfold spectrum using convex optimization
result = detector.unfold_cvxpy(
    readings,
    regularization=1e-4,
    calculate_errors=True
)

# Visualize results
detector.plot_with_uncertainty(result, plot_style='errorbar')

# Calculate and display dose rates
print("Dose rates [pcSv/s]:", result['doserates'])

# Restrict unfolding to energies below 10 MeV (zero fluence above cutoff)
result_limited = detector.unfold_cvxpy(
    readings,
    regularization=1e-4,
    max_neutron_energy=10.0
)
```

Response functions are loaded from a CSV (`E_MeV` + one column per sphere);
readings are a `{sphere_name: value}` dictionary. Built-in response-function
datasets (`RF_GSF`, `RF_PTB`, `RF_LANL`, `RF_JINR`, `RF_FERMILAB`,
`RF_EURADOS`, `RF_IHEP`) can be imported directly:

```python
from bssunfold import Detector, RF_JINR

detector = Detector(RF_JINR)
result = detector.unfold_cvxpy(readings, regularization=1e-4)
```

Full input-format details: [docs — Input Data](https://bssunfold.readthedocs.io/en/latest/data_formats.html).

### Pipeline Example

```python
result = detector.unfold_combined(
    readings=readings,
    pipeline=[
        {"method": "cvxpy", "params": {"regularization": 1e-4}},
        {"method": "landweber", "params": {"max_iterations": 2000}},
    ],
    calculate_errors=True,
)
```

### Parametric Example

```python
# FRUIT-style parametric model (thermal + epithermal + fast)
result = detector.unfold_parametric(
    readings=readings,
    parametric_method='thermal+epithermal+fast',
    optimizer='cvxpy',           # or 'lmfit', 'qpsolvers', 'combined'
    solver_backend='cvxpy:ECOS', # or 'qpsolvers:osqp'
    calculate_errors=True,
)

print(result['doserates'])
```

## ⚙️ Available Unfolding Methods

Methods are grouped into the following categories (all accessible as
``Detector.unfold_*`` instance methods):

| Category | Examples |
|----------|----------|
| Tikhonov-type | `unfold_cvxpy`, `unfold_qpsolvers`, `unfold_tsvd`, `unfold_tikhonov_legendre`, `unfold_pspline_reml` |
| Krylov/hybrid | `unfold_lanczos`, `unfold_gks`, `unfold_cgls`, `unfold_fista`, `unfold_hybrid_gmres`, `unfold_amg` |
| Iterative | `unfold_landweber`, `unfold_mlem`, `unfold_mlem_stop`, `unfold_gravel`, `unfold_doroshenko`, `unfold_kaczmarz`, `unfold_sart` |
| EM family | `unfold_osem`, `unfold_mapem`, `unfold_bsrem`, `unfold_osem_anlm`, `unfold_mlem_bs` |
| Multi-sphere ratio | `unfold_sandii`, `unfold_bunki`, `unfold_bunkiut`, `unfold_rebunki`, `unfold_nsduaz`, `unfold_ferdor` |
| Bayesian | `unfold_bayes`, `unfold_mcmc`, `unfold_cuqi`, `unfold_zfit`, `unfold_eki` |
| Maximum entropy | `unfold_maxed`, `unfold_imaxed`, `unfold_amaxed`, `unfold_nspline` |
| Statistical regularization | `unfold_statreg`, `unfold_reconst`, `unfold_ssr`, `unfold_gee`, `unfold_louhi`, `unfold_uno` |
| Optimization-based | `unfold_lmfit`, `unfold_mystic`, `unfold_smt`, `unfold_genetic`, `unfold_gnowee`, `unfold_scip`, `unfold_docplex`, `unfold_qubo`, `unfold_nnqp`, `unfold_qpmad` |
| Commercial QP (license) | `unfold_gurobi`, `unfold_mosek`, `unfold_cplex`, `unfold_copt`, `unfold_xpress` |
| Optimization course | `unfold_pgd`, `unfold_frank_wolfe`, `unfold_mirror_descent`, `unfold_admm`, `unfold_lbfgsb`, `unfold_coordinate_descent`, `unfold_subgradient`, `unfold_extragradient` |
| Dictionary/sparse | `unfold_cs`, `unfold_nnksvd` |
| Classic codes | `unfold_crystal_ball`, `unfold_rfsp_jul`, `unfold_staysl` |
| Pipeline & ensemble | `unfold_combined`, `unfold_cascade`, `unfold_composite`, `unfold_ensemble`, `unfold_iterative_refinement`, `unfold_binned`, `unfold_maeo` |
| Parametric | `unfold_parametric`, `unfold_parametric2`, `unfold_fruit_like`, `unfold_express`, `unfold_fission_ga` |
| Advanced proximal | `unfold_odl_pdhg`, `unfold_odl_douglas_rachford` |
| Regularization | `unfold_tikhonov_tv`, `unfold_tikhonov_sobolev_dp`, `unfold_epic`, `unfold_interpret` |

Per-method parameters, dependencies and descriptions:
[docs — Method Reference](https://bssunfold.readthedocs.io/en/latest/overview.html#method-reference)

Built-in response functions, dose conversion coefficients, and the full
regularization-parameter selection table:
[docs — Package Overview](https://bssunfold.readthedocs.io/en/latest/overview.html)

## 📊 Spectrum Comparison

Compare two or more unfolded spectra using 41 metrics.

```python
import numpy as np
from bssunfold import Detector

detector = Detector()

r1 = detector.unfold_qpsolvers(readings, save_result=False)
r2 = detector.unfold_cvxpy(readings, save_result=False)

# Compare two results (all simple metrics)
result = detector.compare(r1, r2)
print(result['cosine_similarity'], result['mean_squared_error'])

# Compare with specific metrics
detector.compare(r1, r2, metrics=['cosine_similarity', 'kl_divergence'])

# Visual comparison
detector.compare(r1, r2, plot=True, save_to='comparison.png')

# Independent usage
from bssunfold.utils.comparison import compare_spectra, kl_divergence
all_metrics = compare_spectra(s1, s2)
print(kl_divergence(s1, s2))
```

Full metric catalogue: [docs — Spectrum Comparison Metrics](https://bssunfold.readthedocs.io/en/latest/overview.html#spectrum-comparison-metrics)

## 📈 Output Data & Spectrum Convention

Output-format, dose-calculation and conversion-coefficient details:
[docs — Output Data](https://bssunfold.readthedocs.io/en/latest/data_formats.html#output-data)

Additional advanced features (grid-aware regularization, result management,
custom Monte Carlo uncertainty analysis, `max_neutron_energy` cutoff):
[docs — Advanced Features](https://bssunfold.readthedocs.io/en/latest/data_formats.html#advanced-features)

## 🔧 Technical Requirements

### Core Requirements
- Python 3.11+
- NumPy, SciPy, Pandas, Matplotlib

### Optional Backends

Installable extras (see [pyproject.toml](https://github.com/Radiationsafety/bssunfold/blob/main/pyproject.toml) for version constraints):

| Extra | Provides | Used by |
|-------|----------|---------|
| `ecos` | ECOS conic solver (pre-built wheels) | `unfold_cvxpy` (default solver backend) |
| `numba` | JIT compilation (3–50x speedup) | Landweber, Bayes, Doroshenko, Kaczmarz, MLEM, GRAVEL |
| `tikhonov` | `pytikhonov` — L-curve / GCV / DP selection | Tikhonov-type methods |
| `qpsolvers` | `qpsolvers` QP backends | `unfold_qpsolvers` |
| `solvers-core` | osqp, piqp, qpalm, scs, clarabel + ecos (cross-platform) | QP methods |
| `solvers-proxqp` | `proxsuite` (**not on Windows**) | QP methods |
| `solvers-jax` | jax, jaxlib, jaxopt | JAX-based solvers |
| `all-solvers` | `solvers-core` + `solvers-proxqp` + `solvers-jax` (Unix/Linux/macOS) | QP methods |
| `windows` | `solvers-core` + `solvers-jax` (no proxsuite) | QP methods |
| `lmfit` | `lmfit` | `unfold_lmfit`, parametric NLS fits |
| `mlem` | `odl` | `unfold_mlem_odl` |
| `amg` | `pyamg` | `unfold_amg` |
| `mystic` | `mystic` | `unfold_mystic`, `unfold_mystic_hybrid` |
| `mealpy` | `mealpy` | `unfold_genetic` |
| `smt` | `z3-solver` | `unfold_smt` |
| `scip` | `pyscipopt` | `unfold_scip` |
| `docplex` | `docplex` + `cplex` | `unfold_docplex` |
| `interpret` | `pyoptexplain` | `unfold_interpret` |
| `mcmc` | `pymc` + `arviz` | `unfold_mcmc` |
| `maeo` | `pymoo` | `unfold_maeo` |
| `qubo` | `pyqubo` + `dwave-neal` | `unfold_qubo` |
| `zfit` | `zfit` + `tensorflow` | `unfold_zfit` |
| `cuqi` | `cuqipy` (NumPy ≥ 2.4: [maintained fork](https://github.com/Radiationsafety/CUQIpy/tree/numpy2-support)) | `unfold_cuqi` |
| `gurobi` / `mosek` / `cplex` / `copt` / `xpress` / `commercial` | proprietary QP engines — **LICENSE REQUIRED**, not distributed with bssunfold, excluded from `all` | `unfold_gurobi`, `unfold_mosek`, `unfold_cplex`, `unfold_copt`, `unfold_xpress` |
| `all` | all open-source optional backends above (except commercial engines and `solvers-proxqp`) | — |
| `test` | `pytest` | test suite |

All other methods (GRAVEL, MAXED, Bayes, StatReg, Reconst, TSVD, ScipyDirect, Landweber, Kaczmarz, Doroshenko, MLEM, TikhonovLegendre, NSpline, SSR, GEE, Uno, LOUHI, and the optimization-course solvers) have **no extra dependencies** beyond NumPy/SciPy.

## Performance

All iterative solvers use Numba JIT-compiled inner loops when numba is
installed, with automatic fallback to pure Python — 3–50x speedups
(e.g. Doroshenko 50x, Kaczmarz 14x, MLEM 7x).

```bash
uv add bssunfold[numba]
```

Benchmarks: [docs — Performance](https://bssunfold.readthedocs.io/en/latest/overview.html#performance)

## 📖 Citation
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=CtXdf28AAAAJ&hl=en)

If you use BSSUnfold in your research, please cite paper:
```bibtex
@article{chizhov2024neutron,
  title={Neutron spectra unfolding from Bonner spectrometer readings by the regularization method using the Legendre polynomials},
  author={Chizhov, K and Beskrovnaya, L and Chizhov, A},
  journal={Physics of Particles and Nuclei},
  volume={55},
  number={3},
  pages={532--534},
  year={2024},
  publisher={Springer}
}
```

or software:
```bibtex
@software{bssunfold,
  author       = {Chizhov, Konstantin},
  title        = {BSSUnfold},
  month        = sep,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.24.0},
  doi          = {10.5281/zenodo.22766603},
  url          = {https://doi.org/10.5281/zenodo.22766603},
}
```

## 💬 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📘 Documentation

Documentation and API reference is available in the /docs folder. Theory and
methodology in the research paper, usage examples in the /examples folder.
Check https://bssunfold.readthedocs.io/en/latest/

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 💬 Support

For questions, bug reports, or feature requests:

- Open an issue on [GitHub](https://github.com/radiationsafety/bssunfold/issues)
- Contact: kchizhov@jinr.ru

## 💻 Authors

- Konstantin Chizhov

## 💻 Contributors
- Alexei Chizhov
- Dmitry Borschev
- Maria Akimochkina

## 🌐 Acknowledgments

- ICRP and IAEA for data
- Contributors and testers
- Joint Institute for Nuclear Research (JINR)
- University "Dubna", School of Big Data Analytics

## 🎓 Publications
1. Chizhov A. V., Chizhov K. A. TSVD-Based Iterative Algorithm of Landweber for Neutron Spectra Unfolding by Bonner Multi-Sphere Spectrometer Readings // Phys. Part. Nuclei. 2026. Т. 57. № 4. С. 750–752. https://doi.org/10.1134/S1063779626700735
1. Чижов К.А., Чижов А.В., Борщев Д.С., Акимочкина М.А. Методы решения обратных задач для обработки результатов измерений на примере восстановления спектра нейтронов, Тридцать третья международная конференция "Математика. Компьютер. Образование, г. Дубна, 26 – 31 января 2026 г., [https://mce.su](https://mce.su/rus/presentations/p507586/)
1. Chizhov, K., Chizhov, A. Optimization of the Neutron Spectrum Unfolding Algorithm Using Shifted Legendre Polynomials Based on Weighted Tikhonov Regularization. Phys. Part. Nuclei 56, 1395–1399 (2025). https://doi.org/10.1134/S106377962570056X
2. Chizhov K., Beskrovnaya L., Chizhov A. Neutron spectrum unfolding method based on shifted Legendre polynomials, its application to the IREN facility // Phys. Part. Nucl. Lett. — 2025. — V. 22, no. 2. — P. 337–340. — DOI: https://doi.org/10.1134/S154747712470239X
3. Chizhov K., Beskrovnaya L., Chizhov A. Neutron spectra unfolding from Bonner spectrometer readings by the regularization method using the Legendre polynomials // Phys. Part. Nucl. — 2024. — V. 55. — P. 532–534. — DOI: https://doi.org/10.1134/S1063779624030298
4. Chizhov K., Chizhov A. Optimization approach to neutron spectra unfolding with Bonner multi-sphere spectrometer // Math. Model. — 2024. — V. 7. — P. 89–90.
5. Чижов А. В., Чижов К. А. Восстановление спектров опорных нейтронных полей на Фазотроне (ОИЯИ) на основе показаний многошарового спектрометра Боннера методом усеченного сингулярного разложения Тезисы Трудов LXI Всероссийской конференции по физике РУДН 19 - 23 мая 2025.
6. Chizhov, K., Chizhov, A., TSVD-based neutron spectra unfolding by Bonner multi-sphere spectrometer readings with iteration procedure, proceedings of the International Conference "Distributed Computing and Grid-technologies in Science and Education".
1. Белый А.А., Стариковская М.Д., Чижов К.А. Разработка веб-приложения для эксперимента по восстановлению спектра нейтронов с применением алгоритмов нейронный сетей. Системный анализ в науке и образовании. 2025;(2):49–57.
1. Starikovskaya MD, Chizhov KA. Neutron spectrum unfolding based on random forest algorithm and generated training sample. In Российский университет дружбы народов им. П. Лумумбы; 2025 [cited 2025 Dec 25]. p. 389–94. Available from: https://www.elibrary.ru/item.asp?id=83014906
1. Chizhov KA, Bely AA, Starikovskaia MD, Volkov EN. Восстановление энергетического спектра потока нейтронного излучения с помощью алгоритма машинного обучения «случайный лес». Современные информационные технологии и ИТ-образование. 2024 Dec 15 [cited 2025 Apr 9]; 20(4). Available from: http://sitito.cs.msu.ru/index.php/SITITO/article/view/1167

## 📘 References
1. Compendium of neutron spectra and detector responses for radiation protection purposes: supplement to technical reports series no. 318. — Vienna: International Atomic Energy Agency, 2001. — Technical reports series no. 403. — STI/DOC/010/403. — ISBN 92-0-102201-8.
2. Diamond, S. and Boyd, S., 2016. CVXPY: A Python-embedded modeling language for convex optimization. Journal of Machine Learning Research, 17(83), pp.1-5.

## 🤖 AI Disclosure

This repository is developed with AI-assisted workflows. In accordance with
the project's working model (see [AGENTS.md](AGENTS.md)), AI coding
assistants are used through [OpenCode](https://opencode.ai) with a range of
large language models of different versions (DeepSeek, Qwen, GLM, MiMo and others).

**How AI is used:**

- **Implementation** — drafting and refactoring Python source for unfolding
  methods, ports of published algorithms, solver backends and utility code.
- **Testing** — writing and extending the pytest suite,
  including fixture-based validation against reference implementations.
- **Documentation** — maintaining README, the Sphinx documentation,
  docstrings, and changelog entries.
- **Maintenance** — bug fixes, CI/test-failure diagnosis, dependency updates
  and release chores.

**Human role and verification:**

- Maintainers specify the work, review all changes, and validate results
  against the scientific literature, reference codes and measured data.
- **Scientific responsibility for the package and its results remains with
  the human authors.** AI output is treated as a draft that must pass review
  and tests before acceptance; AI does not make scientific claims on its own
  authority.

---

**BSSUnfold** - Professional neutron spectrum unfolding for radiation science and nuclear applications.
