Hybrid8 start spectrum for ratio-iteration unfolding
====================================================

The **Hybrid8** spectrum is the standard *a-priori* (start) spectrum of the
MSANDB code family used for cosmic-ray neutron spectrometry with Bonner
spheres at balloon altitudes.  It is the start spectrum that the
MSANDB iterative unfolding applies its measured-to-calculated count-rate
ratios to (G. Simmer, V. Mares, E. Weitzenegger and W. Rühm, *Iterative
unfolding for Bonner sphere spectrometers using the MSANDB code —
Sensitivity analysis and dose calculation*, Radiation Measurements **45**
(2010) 1-9, doi:10.1016/j.radmeas.2009.03.023).

The original Hybrid8 spectrum is a tabulated multi-group spectrum.
:func:`bssunfold.core.hybrid8_start` provides a smooth *parametric analogue*
that follows the qualitative description of the cosmic-ray start spectrum
given by S. Barros, V. Mares, R. Bedogni, M. Reginatto, A. Esposito,
I. F. Gonçalves, P. Vaz and W. Rühm, *Comparison of unfolding codes for
neutron spectrometry with Bonner spheres*, Radiation Protection Dosimetry
**161** (2014) 46-52, doi:10.1093/rpd/nct353: a thermal region, a flat
epithermal 1/E region, an evaporation peak around 2 MeV and a cascade peak
around 100 MeV.

Why a start spectrum matters
----------------------------

Ratio-iteration methods (SAND-II family — see :ref:`math-ratio`) correct a
trial spectrum multiplicatively from the measured-to-calculated ratios.
The trial spectrum is therefore not a mere numerical convenience: it
defines the first iterate, the trajectory of the iteration and — because
the iteration is stopped early to counter semi-convergence — the effective
regularization of the result.  A physically shaped start spectrum keeps
the early iterates in the region of the physically plausible spectra, so
early stopping removes less structure than it does for a flat start.

The start spectrum does **not** change the fixed point of the iteration:
with the same number of iterations and the same stopping rule, SAND-II
converges to the same multiplicative update law; only the path and the
amount of implicit regularization differ.

Mathematical formulation
------------------------

The shape is defined by anchor points :math:`(E_k, S_k)` on the
fluence-per-logarithmic-energy convention :math:`S(E) = E\,\Phi(E)`, and
is interpolated linearly in log-log space,

.. math::

   S(E) \;=\; \exp\!\Big(
      \mathrm{interp}_{\log E}\big[\log S_k\big]\Big),

with the end values clamped outside the anchor range.  The returned
spectrum is the flux-density convention :math:`\Phi(E)`, obtained by
dividing out the energy,

.. math::

   \Phi(E_j) \;=\; \frac{S(E_j)}{E_j},

and is finally normalized over the energy grid so that

.. math::

   \sum_j \Phi_j E_j \;=\; \mathrm{total\_fluence},

i.e. the fluence rate integrated over the grid equals the requested
``total_fluence`` (default 1.0).  The result is strictly positive on any
valid grid, which is what the logarithmic ratio update of the SAND-II
family requires.

Anchor table
------------

The 16 anchors of :data:`bssunfold.core.HYBRID8_ANCHORS` describe the four
Hybrid8 regions — thermal rise below ~0.1 MeV, the epithermal 1/E valley
around 30 MeV, the evaporation peak at 2 MeV and the cascade peak at
100 MeV — plus the stiff power-law decrease towards the GeV range:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - E (MeV)
     - S(E) = E·Φ (relative)
     - Region
   * - 1e-9
     - 0.010
     - thermal rise
   * - 1e-7
     - 0.030
     - thermal rise
   * - 1e-5
     - 0.050
     - thermal plateau
   * - 1e-3
     - 0.100
     - thermal plateau
   * - 0.020
     - 0.300
     - rise towards the evaporation peak
   * - 0.100
     - 0.500
     - rise towards the evaporation peak
   * - 0.500
     - 0.850
     - rise towards the evaporation peak
   * - 2.000
     - 1.000
     - **evaporation peak** (maxwellian)
   * - 5.000
     - 0.550
     - epithermal descent
   * - 10.000
     - 0.350
     - epithermal descent
   * - 30.000
     - 0.220
     - **epithermal valley** (1/E region)
   * - 60.000
     - 0.550
     - rise towards the cascade peak
   * - 100.000
     - 1.000
     - **cascade peak**
   * - 300.000
     - 0.200
     - stiff descent
   * - 1000.000
     - 0.030
     - stiff descent
   * - 2000.000
     - 0.008
     - stiff descent

Usage
-----

The named start spectrum is selected through the ``starting_spectrum``
parameter of the SAND-II solver:

.. code-block:: python

   from bssunfold import Detector

   detector = Detector()

   # flat start (default)
   result = detector.unfold_sandii(readings)

   # Hybrid8 cosmic-ray start spectrum
   result = detector.unfold_sandii(readings, starting_spectrum="hybrid8")

An explicit ``initial_spectrum`` always takes precedence over the named
start, so the Hybrid8 shape can also be rescaled or used as the seed of
another scheme:

.. code-block:: python

   import numpy as np
   from bssunfold.core import hybrid8_start

   # evaluate the Hybrid8 shape on the detector grid (default total_fluence=1.0)
   phi0 = hybrid8_start(detector.E_MeV, total_fluence=2.5)

   # explicit a-priori spectrum wins over the named start
   result = detector.unfold_sandii(
       readings,
       initial_spectrum=phi0,
       starting_spectrum="hybrid8",
   )

Caveats
-------

* The Hybrid8 spectrum describes a **cosmic-ray** environment.  For
  laboratory sources (accelerator-driven spectra, Cf-252, fission
  assemblies) the evaporation/cascade structure of a 2 MeV / 100 MeV
  cosmic-ray spectrum may not be representative; the start spectrum should
  then be supplied explicitly, or the run compared against a flat start.
* The shape is defined by a small set of hand-tuned anchors, not by the
  original multi-group Hybrid8 table; it reproduces the qualitative
  four-region structure, not the tabulated values bin by bin.
* The start spectrum only makes a difference for the ratio-iteration
  family.  Regularized (Tikhonov-type) solvers in this package do not use
  a start spectrum at all.

API reference
-------------

See also :doc:`detector` for the full ``Detector`` API reference.

.. autofunction:: bssunfold.core.unfold_hybrid8.hybrid8_start
   :no-index:

.. autodata:: bssunfold.core.unfold_hybrid8.HYBRID8_ANCHORS
   :no-index:
