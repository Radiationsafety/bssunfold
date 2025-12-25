# BSSunfold - Neutron Spectrum Unfolding Package for Bonner Sphere Spectrometers

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue)](https://bssunfold.readthedocs.io/)

## Overview

**BSSUnfold** is a Python package for neutron spectrum unfolding from measurements obtained with Bonner Sphere Spectrometers (BSS). The package implements several mathematical algorithms for solving the inverse problem of unfolding neutron energy spectra from detector readings, with applications in radiation protection, nuclear physics research, and accelerator facilities.

## Features

- **Multiple Unfolding Algorithms**:
  - Tikhonov regularization with convex optimization (CVXPY)
  - Landweber iterative method
  - Combined approach for improved accuracy

- **Radiation Dose Calculations**:
  - ICRP-116 conversion coefficients for effective dose

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

## Visualization Examples

### Spectrum Plotting
```python
# Basic spectrum plot
detector.plot_spectrum()

# With uncertainty bands
detector.plot_spectrum(
    uncertainty=True,
    log_scale=True,
    save_path='spectrum.png'
)

```

### Readings Comparison
```python
# Compare original vs effective readings
detector.plot_readings_comparison(
    bar_width=0.35,
    title='Detector Readings Comparison',
    save_path='readings_comparison.png'
)
```

### Dose Rate Analysis
```python
# Visualize dose rates by geometry
detector.plot_doserates()
```

## Output Data

The package provides comprehensive output in standardized formats:

### Spectrum Results
- Energy grid in MeV
- Unfolded flux spectrum
- Absolute spectrum values
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
├── data/
│   ├── response_functions/     # Detector response functions
│   ├── conversion_coefficients/ # ICRP-116 and other coefficients
│   └── reference_spectra/      # Reference spectra for comparison
├── docs/
│   ├── user_guide.md          # User manual
│   ├── examples/              # Usage examples
│   └── api_reference/         # API documentation
└── tests/                     # Unit tests
```

## Technical Requirements

### Minimum Requirements
- Python 3.12 or higher
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

Full documentation is available in  /docs folder

- API reference
- Tutorials and examples
- Theory and methodology

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Support

For questions, bug reports, or feature requests:

- Open an issue on [GitHub](https://github.com/radiationsafety/bssunfold/issues)
- Contact: kchizhov@jinr.ru

## Acknowledgments

- ICRP for conversion coefficient data
- Contributors and testers
- Research institutions providing validation data

---

**BSSUnfold** - Professional neutron spectrum unfolding for radiation science and nuclear applications.