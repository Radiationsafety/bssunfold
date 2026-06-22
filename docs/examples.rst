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

Parametric Unfolding
--------------------

The parametric methods model the neutron spectrum as a weighted sum of
thermal, epithermal, and fast components (FRUIT-style model). They are
especially useful when the spectrum shape can be approximated by these
three components.

.. code-block:: python

   import pandas as pd
   from bssunfold import Detector

   detector = Detector(pd.read_csv('response_functions.csv'))
   readings = {"0in": 0.0003, "2in": 0.0099, "3in": 0.0536, "5in": 0.1841}

   # Simple lmfit-based parametric unfolding
   result_param = detector.unfold_parametric(
       readings,
       parametric_method='thermal+epithermal+fast',
       optimizer='lmfit',
       calculate_errors=True,
   )

   # SQP via cvxpy (requires initial_params or auto-scan)
   result_cvxpy = detector.unfold_parametric(
       readings,
       parametric_method='thermal+epithermal+fast',
       optimizer='cvxpy',
       solver_backend='cvxpy:ECOS',
       calculate_errors=True,
   )

   # Combined: lmfit first, then QP refinement
   result_combined = detector.unfold_parametric(
       readings,
       parametric_method='thermal+epithermal+fast',
       optimizer='combined',
       solver_backend='cvxpy',
       calculate_errors=True,
   )

    # Standalone parametric solver with custom initial guess
    from bssunfold.core.unfold_parametric import solve_parametric_cvxpy

    result = solve_parametric_cvxpy(
        A_matrix=detector.response_functions.values,
        b_readings=np.array([readings[k] for k in detector.sphere_names]),
        E=detector.energy_bins,
        parametric_method='thermal+epithermal+fast',
        initial_params={'P_th': 1e5, 'P_epi': 1e5, 'P_f': 1e5,
                        'b': 0.5, 'beta_prime': 0.5, 'alpha': 2.0, 'beta': 0.5},
        max_iter=20,
        tolerance=1e-6,
        solver_backend='auto',
    )

BON95 Parametric Unfolding
--------------------------

The BON95 method models the lethargy spectrum E*Phi(E) as a linear
combination of four components (thermal, epithermal, intermediate, fast)
with shape parameters found by grid search and linear coefficients solved
by weighted NLS. After parametric fitting, the result is refined by
directed-divergence (I-divergence) iterations.

.. code-block:: python

   import pandas as pd
   from bssunfold import Detector

   detector = Detector(pd.read_csv('response_functions.csv'))
   readings = {"0in": 0.0003, "2in": 0.0099, "3in": 0.0536, "5in": 0.1841}

   # BON95 parametric unfolding (grid search + directed-divergence)
   result_bon95 = detector.unfold_parametric2(
       readings,
       b_range=(0.5, 2.0, 5),       # epithermal exponent grid
       Tf_range=(0.5, 10.0, 5),     # fast peak energy grid (MeV)
       c_range=(0.5, 3.0, 4),       # fast peak width grid
       noise_level=0.05,             # 5% measurement uncertainty
       calculate_errors=True,
   )

   # Compare with FRUIT parametric
   result_fruit = detector.unfold_parametric(readings, optimizer='lmfit')

   print("BON95 spectrum shape:", result_bon95['spectrum'].shape)
   print("FRUIT spectrum shape:", result_fruit['spectrum'].shape)

   # Standalone BON95 solver
   from bssunfold.core.unfold_parametric2 import solve_parametric2
   import numpy as np

   E = detector.E_MeV
   ln_steps = np.zeros(len(E))
   log_e = np.log10(E + 1e-15)
   ln_steps[0] = log_e[1] - log_e[0]
   ln_steps[-1] = log_e[-1] - log_e[-2]
   ln_steps[1:-1] = (log_e[2:] - log_e[:-2]) / 2.0
   ln_steps *= np.log(10)

   A = np.array([detector.sensitivities[n] for n in readings])
   b = np.array([readings[n] for n in readings])

   spectrum, success, msg, nfev = solve_parametric2(A, b, E, ln_steps)
   print(f"Converged: {success}, message: {msg}")