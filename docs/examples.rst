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

   # Unfold with MAXED (maximum entropy)
   result_maxed = detector.unfold_maxed(
       readings,
       sigma_factor=0.01,
       omega=1.0
   )

   # Unfold with GRAVEL
   result_gravel = detector.unfold_gravel(
       readings,
       max_iterations=500,
       tolerance=1e-6
   )
