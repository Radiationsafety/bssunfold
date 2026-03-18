# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],

and this project adheres to [Semantic Versioning].


## [0.4.1] - 2026-03-17
### Added
 - qpsolvers: smoothness with 1st and 2nd derivatives
 - 11-QP_solvers_smooth.ipynb example for qpsolvers smooth
 - lmfit initial_spectrum

## [0.4.0] - 2026-03-16
### Added
 - Doroshenko iterative method
 - Karcmarz algorithm
 - lmfit package 
 - examples 9-10 for new methods
 - error bar with std for plot_with_uncertainty function

  ### Changed
 - python 3.14 not supported because of proxsuite==0.7.2

## [0.3.0] - 2026-03-11
### Added
 - qpsolvers for QP open source solvers
 - combined algorithm
 - examples 6-8 for combined algorithm, plot with uncertainty, qpsolvers
 - plot_with_uncertainty function
 - save figure with response functions to png, pdf, eps, jpg
 - automatic selection of regularization parameter via pytikhonov package

 ### Changed
 - docs updated

## [0.2.0] - 2026-02-02
### Added
 - mlem algorithm via ODL, with example

## [0.1.3] - 2026-01-15

### Added
 - RF_PTB  in constants (response function for PTB BSS)
 - RF_LANL in constants (response function for LANL BSS)

### Changed
 - numpy 2.0.2 for micropip in marimo


## [0.1.2] - 2026-01-14

### Added
 - conda recipe

### Changed
 - pandas 2.3.3 for micropip in marimo
 - readme.md


## [0.1.1] - 2026-01-12

### Added
 - shields 
 - Citation.cff
 - Codeowners
 - Code of conduct
 - Response functions as a dict to the constants. 
 - github workflows

### Changed
 - 01 basic example


## [0.1.0] - 2025-12-25

- initial release

### Added
- Landweber iterative method
- Tikhonov regularization with CVXPY
- docs
- example
- simple tests

<!-- Links -->
[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

<!-- Versions -->
<!-- [unreleased]: https://github.com/Author/Repository/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/Author/Repository/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/Author/Repository/releases/tag/v0.0.1 -->

<!-- ### Changed

### Deprecated

### Removed

### Fixed

### Security
 -->