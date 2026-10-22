Input and Output Data
=====================

This page describes the data formats accepted by :class:`bssunfold.Detector`,
the structure of the result dictionary returned by every ``unfold_*`` method,
and the advanced features that operate on both.

.. _data-input:

Input Data
----------

Response functions
^^^^^^^^^^^^^^^^^^

The response functions describe the neutron fluence response of every
Bonner sphere of the spectrometer on a common energy grid.  They can be
supplied in four ways, all handled by the :class:`bssunfold.Detector`
constructor:

.. code-block:: python

   import pandas as pd
   from bssunfold import Detector

   # 1) CSV file with an "E_MeV" column and one column per sphere
   detector = Detector(pd.read_csv("response.csv"))

   # 2) pandas DataFrame with an "E_MeV" column and one column per sphere
   detector = Detector(df)

   # 3) dict with an "E_MeV" key and one entry per sphere
   detector = Detector({"E_MeV": energies, "3in": s3, "5in": s5})

   # 4) energy grid + sensitivities, without a response table
   detector = Detector(E_MeV=energies, sensitivities={"3in": s3, "5in": s5})

The standard CSV layout has one row per energy point, a first column named
``E_MeV`` holding the energy in MeV, and one column per detector sphere
whose header is the sphere name used later in the readings dictionary:

.. code-block:: text

   E_MeV,3in,5in,7in,9in,10in,12in,15in,18in
   1.0e-09,1.2e-06,4.5e-07,1.9e-07,9.0e-08,4.1e-08,1.2e-08,2.0e-09,2.6e-10
   1.0e-08,1.2e-05,4.5e-06,1.9e-06,9.0e-07,4.1e-07,1.2e-07,2.0e-08,2.6e-09
   ...

The sensitivities are normalised to the response of one incident neutron per
unit fluence, so the folded count rate is

.. math::

   N_i = \sum_j R_i(E_j)\,\Phi_j\,\mathrm{d}\ln E_j .

The package uses the *midpoint* convention: the tabulated energy is the
centre of its bin and the corresponding bin width :math:`\mathrm{d}\ln E_j`
is derived from the neighbouring points (``Detector.ln_steps``).  The
resulting bin edges are available as ``Detector.energy_bin_edges_MeV``.

Built-in response-function sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Seven response-function sets are shipped with the package and can be
imported directly:

.. code-block:: python

   from bssunfold import Detector, RF_GSF, RF_PTB, RF_LANL, RF_JINR
   from bssunfold import RF_FERMILAB, RF_EURADOS, RF_IHEP

   detector = Detector(RF_JINR)

.. list-table::
   :header-rows: 1
   :widths: 22 14 26 38

   * - Dataset
     - Spheres
     - Energy range
     - Typical use
   * - ``RF_GSF``
     - 10
     - 1e-9 .. 631 MeV
     - GSF Bonner-sphere set; the default of ``Detector()``
   * - ``RF_PTB``
     - 15
     - 1e-9 .. 631 MeV
     - PTB set with the intermediate-thickness spheres
   * - ``RF_LANL``
     - 11
     - 1e-9 .. 631 MeV
     - LANL set with lead-shielded spheres (``12inPb``, ``18inPb``)
   * - ``RF_JINR``
     - 9
     - 1e-9 .. 631 MeV
     - JINR set with cadmium-covered spheres
   * - ``RF_FERMILAB``
     - 8
     - 1e-9 .. 631 MeV
     - Fermilab set
   * - ``RF_EURADOS``
     - 13
     - 1e-9 .. 20 MeV
     - EURADOS inter-comparison set (finer low-energy grid)
   * - ``RF_IHEP``
     - 12
     - 1e-9 .. 2e3 MeV
     - IHEP set extending to the GeV range

Readings
^^^^^^^^

The readings are passed to every ``unfold_*`` method as a plain dictionary
that maps the sphere name to its count rate:

.. code-block:: python

   readings = {"3in": 0.053, "5in": 0.184, "10in": 0.172, "18in": 0.034}

The values are **rates** (counts per second) unless the Poisson noise
model is used together with an explicit counting time, in which case the
expected counts are obtained as :math:`C_i = b_i T`.  Only the spheres that
are present in the dictionary take part in the unfolding; the order of the
readings does not matter.

Dose conversion coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The detector converts the unfolded spectrum into ambient dose equivalent
using a tabulated set of fluence-to-dose conversion coefficients, selected
with the ``cc_type`` argument:

.. code-block:: python

   detector = Detector(RF_GSF, cc_type="ICRP116")

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - ``cc_type``
     - Source
     - Geometries
   * - ``"ICRP116"`` (default)
     - ICRP-116 effective dose coefficients
     - AP, PA, LLAT, RLAT, ROT, ISO
   * - ``"ICRP74_effective"``
     - ICRP-74 effective dose coefficients
     - AP, PA, RLAT, ROT, ISO
   * - ``"NRB99_2009_effective"``
     - NCRP Report No. 99 (2009) coefficients
     - AP, ISO
   * - ``"ICRP74_operational"``
     - ICRP-74 operational quantities
     - ADE, PDE0, PDE45, PDE60, PDE75

The coefficients are interpolated onto the detector energy grid before the
integration, so any energy grid can be used with any coefficient set.

.. _data-output:

Output Data
-----------

All ``unfold_*`` methods return the same dictionary, so the downstream
processing is independent of the method that was used.  The keys that are
present for every method are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Meaning
   * - ``"energy"``
     - Energy grid (MeV) of the unfolded spectrum, one entry per bin.
   * - ``"spectrum"``
     - Unfolded spectrum (see the convention below), non-negative.
   * - ``"spectrum_absolute"``
     - Same as ``"spectrum"``; kept for backward compatibility.
   * - ``"spectrum_definition"``
     - ``"differential_fluence_per_dlnE"`` — the convention of the spectrum.
   * - ``"spectrum_units"``
     - ``"cm^-2 s^-1 (d ln E)^-1"``.
   * - ``"energy_bin_edges_MeV"``
     - Bin edges (MeV), ``n_energy_bins + 1`` values.
   * - ``"integration_rule"``
     - ``"rectangular_midpoint_dlnE"`` — the rule used for the dose integration.
   * - ``"effective_readings"``
     - Count rates calculated from the unfolded spectrum, per sphere.
   * - ``"residual"``, ``"residual_norm"``
     - Difference between the measured and the calculated readings, and its
       Euclidean norm.
   * - ``"method"``
     - Name of the solver that produced the result.
   * - ``"doserates"``
     - Dose rate per irradiation geometry (see below).
   * - ``"dose_coverage_fraction"``
     - Fraction of the coefficient energy range covered by the spectrum.

Method-specific keys are added on top of these — the regularisation
parameters of the Tikhonov-type solvers, the iteration counters of the
iterative ones, the ``"starting_spectrum"`` of the ratio methods, and so on.

Spectrum convention
^^^^^^^^^^^^^^^^^^^

The spectrum is a **differential fluence per unit lethargy**,
:math:`\mathrm{d}\Phi/\mathrm{d}\ln E`, in
:math:`\mathrm{cm^{-2}\,s^{-1}\,(\mathrm{d}\ln E)^{-1}}`.  It is not a
fluence per bin and not a flux density: the fluence of a bin is obtained by
multiplying with the bin width,

.. math::

   F_j = \Phi_j\,\mathrm{d}\ln E_j,
   \qquad
   F_{\text{tot}} = \sum_j F_j .

The same convention is used by the dose integration,
:math:`H = \sum_j h(E_j)\,\Phi_j\,\mathrm{d}\ln E_j`, and by the forward
model, which is why the reconstructed readings match the measured ones.

Dose rates
^^^^^^^^^^

``result["doserates"]`` maps every geometry of the selected coefficient set
to the ambient dose equivalent rate of the unfolded spectrum:

.. code-block:: python

   result = detector.unfold_cvxpy(readings)
   print(result["doserates"]["ISO"])     # effective dose rate, pSv/s
   print(result["doserates"]["AP"])      # anterior-posterior irradiation

The values are in pico-sievert per second (pSv/s) and are the ambient
dose equivalent :math:`H` for the corresponding irradiation geometry.

Uncertainty arrays
^^^^^^^^^^^^^^^^^^

With ``calculate_errors=True`` the result also carries the Monte-Carlo
statistics of the spectrum:

``"spectrum_uncert_mean"``, ``"spectrum_uncert_std"``, ``"spectrum_uncert_min"``,
``"spectrum_uncert_max"``, ``"spectrum_uncert_median"``,
``"spectrum_uncert_percentile_5"``, ``"spectrum_uncert_percentile_95"`` and
``"spectrum_uncert_all"`` (the full ``n_montecarlo x n_energy_bins`` sample
matrix).  See :ref:`data-advanced` for the noise models behind them.

.. _data-advanced:

Advanced Features
-----------------

Maximum energy cutoff
^^^^^^^^^^^^^^^^^^^^^

Every method accepts ``max_neutron_energy``, which forces the fluence above
the given energy (MeV) to zero:

.. code-block:: python

   result = detector.unfold_cvxpy(readings, max_neutron_energy=10.0)

The bins above the cutoff are removed from the problem, the solution is
expanded back to the full grid afterwards, and the dose rates, the
effective readings and the residual are recomputed on the full grid, so
that all derived quantities stay consistent with the returned spectrum.

Result management
^^^^^^^^^^^^^^^^^

With ``save_result=True`` the result is stored in the detector history and
can be retrieved later:

.. code-block:: python

   detector = Detector(RF_GSF)
   result = detector.unfold_cvxpy(readings, save_result=True)

   detector.results_history      # dict of all saved results
   detector.current_result       # the most recent one

Each saved entry receives a ``"timestamp"`` and a ``"saved_key"`` of the
form ``<YYYYmmdd_HHMMSS>_<method>``.

Monte Carlo uncertainty analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The uncertainty of the unfolded spectrum is estimated by perturbing the
readings and re-running the unfolding.  The statistical and the systematic
parts of the perturbation are separated:

.. code-block:: python

   # Poisson counting statistics with an explicit counting time (s)
   result = detector.unfold_cvxpy(
       readings,
       calculate_errors=True,
       n_montecarlo=100,
       noise_model="poisson",
       measurement_time=3600.0,
   )

   # absolute 1-sigma uncertainties per reading
   result = detector.unfold_cvxpy(
       readings,
       calculate_errors=True,
       reading_uncertainties={"3in": 0.004, "5in": 0.006},
   )

   # correlated systematic effects between the readings
   result = detector.unfold_cvxpy(
       readings,
       calculate_errors=True,
       reading_covariance=covariance_matrix,
   )

The statistical component follows the first applicable rule: the Poisson
counting statistics when ``noise_model="poisson"``, the absolute
uncertainties of ``reading_uncertainties`` when given, otherwise the
legacy relative Gaussian noise ``noise_level``.  The covariance matrix
``reading_covariance`` is always added on top of the statistical part, so
that correlated systematics (shared calibration, response-model errors) are
propagated as well.  ``random_state`` makes the sampling reproducible, and
``variance_reduction`` selects the optional ``"antithetic"``, ``"control"``
or ``"both"`` variance-reduction schemes; the parameter is available on the
methods that expose it (``unfold_louhi``, ``unfold_pgd``,
``unfold_mirror_descent``, ``unfold_frank_wolfe``).

Grid-aware regularization
^^^^^^^^^^^^^^^^^^^^^^^^^

The derivative-based penalties of the Tikhonov-type solvers can be built
either on the bin index or on the actual energy grid.  With
``grid_aware=True`` the difference operator acts on :math:`\ln E`, so the
penalty weights the bins by their true logarithmic width and does not
distort a non-uniform grid:

.. code-block:: python

   result = detector.unfold_gks(readings, grid_aware=True)

The same option is available for the Fisher-information solver, which
builds its derivative operator on the ``ln E`` grid when the energy grid is
supplied.

Spectrum comparison
^^^^^^^^^^^^^^^^^^^

Two or more unfolded spectra can be compared with 41 metrics through
``Detector.compare``:

.. code-block:: python

   result = detector.compare(r1, r2)
   print(result["cosine_similarity"], result["mean_squared_error"])

See :doc:`overview` for the complete metric catalogue.

API reference
-------------

See also :doc:`detector` for the full ``Detector`` API reference.
