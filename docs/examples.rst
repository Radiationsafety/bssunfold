Examples
========

Basic Usage
-----------

.. code-block:: python

   import pandas as pd
   from bssunfold import Detector
   
   # Load response functions
   rf_df = pd.read_csv('response_functions.csv')
   
   # Initialize detector
   detector = Detector(rf_df)
   
   # Define readings
   readings = {
       'sphere_1': 150.2,
       'sphere_2': 120.5,
       'sphere_3': 95.7
   }
   
   # Unfold with cvxpy
   result_cvxpy = detector.unfold_cvxpy(
       readings,
       regularization=0.001,
       calculate_errors=True
   )
   
   # Unfold with Landweber
   result_landweber = detector.unfold_landweber(
       readings,
       max_iterations=500,
       tolerance=1e-5,
       calculate_errors=True
   )

   # Unfold with MLEM
    result_mlem = detector.unfold_mlem_odl(
        readings,
        max_iterations=500,
        calculate_errors=True
    )

   # Unfold with GRAVEL (no extra deps)
   result_gravel = detector.unfold_gravel(
       readings,
       max_iterations=200,
       tolerance=1e-6
   )

   # Unfold with MAXED (maximum entropy, no extra deps)
   result_maxed = detector.unfold_maxed(
       readings,
       sigma_factor=0.1
   )

   # Unfold with Bayes (D'Agostini, no extra deps)
   result_bayes = detector.unfold_bayes(
       readings,
       max_iterations=200,
       tolerance=1e-3
   )

   # Unfold with TSVD (truncated SVD, no extra deps)
   result_tsvd = detector.unfold_tsvd(
       readings,
       k=5,
       method='l_curve'
   )

   # Unfold with StatReg (Turchin, no extra deps)
   result_statreg = detector.unfold_statreg(
       readings,
       unfoldermethod='EmpiricalBayes'
   )


Using Spectral Bases
--------------------

The ``basis`` parameter lets you choose the representation space for the
spectrum.  Available bases: ``BinBasis`` (default), ``LegendreBasis``,
``FourierBasis``.

.. code-block:: python

   from bssunfold import Detector
   from bssunfold.core import LegendreBasis, FourierBasis

   detector = Detector()
   readings = {name: 100.0 for name in detector.detector_names[:3]}

   # Legendre polynomial basis (15 polynomials)
   result = detector.unfold_cvxpy(
       readings,
       basis=LegendreBasis(n_polynomials=15),
       regularization=1e-3,
   )

   # Fourier sin/cos basis (10 terms)
   result = detector.unfold_landweber(
       readings,
       basis=FourierBasis(n_terms=10),
       max_iterations=200,
   )

   # QP solver with Legendre + smoothness penalty in coefficient space
   result = detector.unfold_qpsolvers(
       readings,
       basis=LegendreBasis(n_polynomials=12),
       regularization=0.01,
       smoothness_order=2,
   )

   # Convert between representations
   from bssunfold.core import LegendreBasis
   basis = LegendreBasis(n_polynomials=15)
   E = detector.E_MeV
   spectrum = result['spectrum']
   coeffs = basis.to_coeffs(spectrum, E)        # spectrum -> coefficients
   spectrum_back = basis.to_spectrum(coeffs, E)  # coefficients -> spectrum