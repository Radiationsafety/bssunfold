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