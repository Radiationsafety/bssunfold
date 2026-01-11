# BSSunfold - Neutron Spectrum Unfolding Package for Bonner Sphere Spectrometers
[![PyPI - Version](https://img.shields.io/pypi/v/BSSUnfold)](https://pypi.org/project/bssunfold/)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue)](https://bssunfold.readthedocs.io/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/7dd7cc75ab654b879b80abe8476907f6)](https://app.codacy.com/gh/Radiationsafety/bssunfold/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/7dd7cc75ab654b879b80abe8476907f6)](https://app.codacy.com/gh/Radiationsafety/bssunfold/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![DOI](https://zenodo.org/badge/1122800086.svg)](https://doi.org/10.5281/zenodo.18056376)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Radiationsafety/bssunfold/blob/main/example/01-basic-example.ipynb)
## Overview

**BSSUnfold** is a Python package for neutron spectrum unfolding from measurements obtained with Bonner Sphere Spectrometers (BSS). The package implements several mathematical algorithms for solving the inverse problem of unfolding neutron energy spectra from detector readings, with applications in radiation protection, nuclear physics research, and accelerator facilities.

![logo](docs/images/bssunfold_logo.png)

## Features

- **Multiple Unfolding Algorithms**:
  - Tikhonov regularization with convex optimization (CVXPY)
  - Landweber iterative method
  - Combined approach for improved accuracy

- **Radiation Dose Calculations**:
  - INTERNATIONAL COMMISSION ON RADIOLOGICAL PROTECTION (ICRP), publication 116: conversion coefficients for effective dose

- **Comprehensive Data Management**:
  - Automatic response function processing
  - Uncertainty quantification via Monte Carlo methods

- **Advanced Visualization**:
  - Spectrum plotting with uncertainty bands
  - Detector reading comparisons

## Installation

### Using pip
```bash
pip install bssunfold
```

### Using uv (recommended)
```bash
uv add bssunfold
```

### From Source
```bash
git clone https://github.com/radiationsafety/bssunfold.git
cd bssunfold
pip install -e .
```

## Quick Start

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
detector.plot_spectrum(uncertainty=True)
detector.plot_readings_comparison()

# Calculate and display dose rates
print("Dose rates [pcSv/s]:", result['doserates'])
```

## Input Data Structure

### Response Functions
Response functions must be provided as a CSV file with the following format:
```
E_MeV,0in,2in,3in,5in,8in,10in,12in
1.00E-09,0.001,0.005,0.01,0.02,0.03,0.04,0.05
1.00E-08,0.002,0.006,0.012,0.022,0.032,0.042,0.052
...
```

### Detector Readings
Readings should be provided as a dictionary mapping sphere names to measured values:
```python
readings = {
    'sphere_0in': 150.2,   # Bare detector
    'sphere_2in': 120.5,   # 2-inch polyethylene sphere
    'sphere_3in': 95.7,    # 3-inch polyethylene sphere
    # ... additional spheres
}
```

## Available Methods

### 1. `unfold_cvxpy()`
Tikhonov regularization with convex optimization for stable spectrum reconstruction.

```python
result = detector.unfold_cvxpy(
    readings,
    regularization=1e-4,      # Regularization parameter
    norm=2,                   # L2 norm for regularization
    calculate_errors=True,    # Monte Carlo uncertainty estimation
    save_result=True          # Store result in history
)
```

### 2. `unfold_landweber()`
Iterative Landweber method with convergence control.

```python
result = detector.unfold_landweber(
    readings,
    max_iterations=1000,      # Maximum iterations
    tolerance=1e-6,           # Convergence tolerance
    calculate_errors=True,    # Monte Carlo uncertainty
    save_result=True
)
```

## Output Data

The package provides comprehensive output in standardized formats:

### Spectrum Results
- Energy grid in MeV
- Unfolded neutron spectrum for the grid of energy bins
- Uncertainty estimates (if calculated)

### Dose Calculations
- Effective dose rates for different geometries:
  - AP (Anterior-Posterior)
  - PA (Posterior-Anterior)
  - LLAT (Left Lateral)
  - RLAT (Right Lateral)
  - ROT (Rotational)
  - ISO (Isotropic)

### Quality Metrics
- Residual norms
- Convergence status
- Iteration counts
- Monte Carlo statistics

## Application Areas

### Nuclear Research Facilities
- Neutron spectroscopy at particle accelerators
- Reactor neutron field characterization
- Fusion device diagnostics

### Radiation Protection
- Workplace monitoring at nuclear power plants
- Medical accelerator facilities
- Industrial radiography installations

### Scientific Research
- Space radiation studies
- Cosmic ray neutron measurements
- Nuclear physics experiments

## Advanced Features

### Result Management
```python
# List all saved results
results = detector.list_results()
print(f"Available results: {results}")

# Retrieve specific result
result = detector.get_result('20240115_143022_cvxpy')

# Create comprehensive report
report = detector.create_summary_report(
    save_path='unfolding_report.json'
)

# Clear results history
detector.clear_results()
```

### Custom Uncertainty Analysis
```python
# Custom Monte Carlo parameters
result = detector.unfold_cvxpy(
    readings,
    calculate_errors=True,
    n_montecarlo=500,      # Number of samples
    noise_level=0.02       # 2% measurement noise
)

# Access uncertainty data
uncert_mean = result['spectrum_uncert_mean']
uncert_std = result['spectrum_uncert_std']
percentile_95 = result['spectrum_uncert_percentile_95']
```

## Data Structure

```
bssunfold/
├── CHANGELOG.md
├── CONTRIBUTING.md
├── data
│   └── response_functions
│       └── rf_GSF.csv
├── docs
│   ├── conf.py
│   ├── detector.rst
│   ├── examples.rst
│   ├── images
│   │   └── bssunfold_logo.png
│   ├── index.rst
│   ├── make.bat
│   ├── makefile
│   └── requirements.txt
├── example
│   └── 01-basic-example.ipynb
├── favicon.ico
├── LICENSE
├── pyproject.toml
├── README.md
├── README.pdf
├── requirements.txt
├── src
│   └── bssunfold
│       ├── constants.py
│       ├── detector.py
│       ├── __init__.py
├── tests
│   ├── __init__.py
│   └── test_detector.py
└── uv.lock
```

## Technical Requirements

### Minimum Requirements
- Python 3.11 or higher
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Pandas >= 1.3.0

### Optional Dependencies
- CVXPY >= 1.1.0 (for convex optimization)
- Matplotlib >= 3.5.0 (for visualization)

## Performance

- **Matrix Operations**: Optimized NumPy operations for response matrices
- **Memory Efficient**: Sparse matrix support for large energy grids
- **Parallel Processing**: Monte Carlo simulations can be parallelized
- **Caching**: Response matrices are cached for repeated use

## Citation
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=CtXdf28AAAAJ&hl=en)

If you use BSSUnfold in your research, please cite:
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

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Documentation

Documentation and API reference is available in /docs folder. Theory and methodology in the research paper, example of usage in /examples folder.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Support

For questions, bug reports, or feature requests:

- Open an issue on [GitHub](https://github.com/radiationsafety/bssunfold/issues)
- Contact: kchizhov@jinr.ru

## Acknowledgments

- ICRP and IAEA for data
- Contributors and testers
- Joint Institure for Nuclear Research
- University "Dubna", School of Big Data Analytics

## Publications

1. Чижов К.А., Чижов А.В., Борщев Д.С., Акимочкина М.А. Методы решения обратных задач для обработки результатов измерений на примере восстановления спектра нейтронов, Тридцать третья международная конференция "Математика. Компьютер. Образование, г. дубна, 26 – 31 января 2026 г.
1. Chizhov, K., Chizhov, A. Optimization of the Neutron Spectrum Unfolding Algorithm Using Shifted Legendre Polynomials Based on Weighted Tikhonov Regularization. Phys. Part. Nuclei 56, 1395–1399 (2025). https://doi.org/10.1134/S106377962570056X
2. Chizhov K., Beskrovnaya L., Chizhov A. Neutron spectrum unfolding method based on shifted Legendre polynomials, its application to the IREN facility // Phys. Part. Nucl. Lett. — 2025. — V. 22, no. 2. — P. 337–340. — DOI: https://doi.org/10.1134/S154747712470239X
3. Chizhov K., Beskrovnaya L., Chizhov A. Neutron spectra unfolding from Bonner spectrometer readings by the regularization method using the Legendre polynomials // Phys. Part. Nucl. — 2024. — V. 55. — P. 532–534. — DOI: https://doi.org/10.1134/S1063779624030298
4. Chizhov K., Chizhov A. Optimization approach to neutron spectra unfolding with Bonner multi-sphere spectrometer // Math. Model. — 2024. — V. 7. — P. 89–90.
5. Чижов А. В., Чижов К. А. Восстановление спектров опорных нейтронных полей на Фазотроне (ОИЯИ) на основе показаний многошарового спектрометра Боннера методом усеченного сингулярного разложения Тезисы Трудов LXI Всероссийской конференции по физике РУДН 19 - 23 мая 2025.
6. Chizhov, K., Chizhov, A., TSVD-based neutron spectra unfolding by Bonner multi-sphere spectrometer readings with iteration procedure, proceedings of the International Conference "Distributed Computing and Grid-technologies in Science and Education".
1. Белый А.А., Стариковская М.Д., Чижов К.А. Разработка веб-приложения для эксперимента по восстановлению спектра нейтронов с применением алгоритмов нейронный сетей. Системный анализ в науке и образовании. 2025;(2):49–57. 
1. Starikovskaya MD, Chizhov KA. Neutron spectrum unfolding based on random forest algorithm and generated training sample. In Российский университет дружбы народов им. П. Лумумбы; 2025 [cited 2025 Dec 25]. p. 389–94. Available from: https://www.elibrary.ru/item.asp?id=83014906
1. Chizhov KA, Bely AA, Starikovskaia MD, Volkov EN. Восстановление энергетического спектра потока нейтронного излучения с помощью алгоритма машинного обучения «случайный лес». Современные информационные технологии и ИТ-образование. 2024 Dec 15 [cited 2025 Apr 9];20(4). Available from: http://sitito.cs.msu.ru/index.php/SITITO/article/view/1167

## References
1. Compendium of neutron spectra and detector responses for radiation protection purposes: supplement to technical reports series no. 318. — Vienna: International Atomic Energy Agency, 2001. — Technical reports series no. 403. — STI/DOC/010/403. — ISBN 92-0-102201-8.
2. Diamond, S. and Boyd, S., 2016. CVXPY: A Python-embedded modeling language for convex optimization. Journal of Machine Learning Research, 17(83), pp.1-5.


---

**BSSUnfold** - Professional neutron spectrum unfolding for radiation science and nuclear applications.