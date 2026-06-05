Detector Class
==============

.. autoclass:: bssunfold.Detector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __str__, __repr__
   
Unfold Methods
==============

The following unfolding methods are available through the Detector class:

.. autofunction:: bssunfold.core.unfold_cvxpy.unfold_cvxpy

.. autofunction:: bssunfold.core.unfold_landweber.unfold_landweber

.. autofunction:: bssunfold.core.unfold_mlem.unfold_mlem

.. autofunction:: bssunfold.core.unfold_qpsolvers.unfold_qpsolvers

.. autofunction:: bssunfold.core.unfold_doroshenko.unfold_doroshenko

.. autofunction:: bssunfold.core.unfold_kaczmarz.unfold_kaczmarz

.. autofunction:: bssunfold.core.unfold_lmfit.unfold_lmfit

.. autofunction:: bssunfold.core.unfold_mlem_odl.unfold_mlem_odl

.. autofunction:: bssunfold.core.unfold_combined.unfold_combined

.. autofunction:: bssunfold.core.unfold_gravel.unfold_gravel

.. autofunction:: bssunfold.core.unfold_maxed.unfold_maxed

.. autofunction:: bssunfold.core.unfold_tikhonov_legendre.unfold_tikhonov_legendre

.. autofunction:: bssunfold.core.unfold_bayes.unfold_bayes

.. autofunction:: bssunfold.core.unfold_bayes_spline_regularization.unfold_bayes_spline_regularization

.. autofunction:: bssunfold.core.unfold_statreg.unfold_statreg

.. autofunction:: bssunfold.core.unfold_scipy_direct_method.unfold_scipy_direct_method

.. autofunction:: bssunfold.core.unfold_tsvd.unfold_tsvd

Core Functions
==============

Underlying solver functions:

.. autofunction:: bssunfold.core.unfold_cvxpy.solve_cvxpy

.. autofunction:: bssunfold.core.unfold_landweber.solve_landweber

.. autofunction:: bssunfold.core.unfold_mlem.solve_mlem

.. autofunction:: bssunfold.core.unfold_qpsolvers.solve_qpsolvers

.. autofunction:: bssunfold.core.unfold_doroshenko.solve_doroshenko

.. autofunction:: bssunfold.core.unfold_kaczmarz.solve_kaczmarz

.. autofunction:: bssunfold.core.unfold_lmfit.solve_lmfit

.. autofunction:: bssunfold.core.unfold_gravel.solve_gravel

.. autofunction:: bssunfold.core.unfold_maxed.solve_maxed

.. autofunction:: bssunfold.core.unfold_tikhonov_legendre.solve_tikhonov_legendre

.. autofunction:: bssunfold.core.unfold_bayes.solve_bayes

.. autofunction:: bssunfold.core.unfold_bayes_spline_regularization.solve_bayes_spline

.. autofunction:: bssunfold.core.unfold_statreg.solve_statreg

.. autofunction:: bssunfold.core.unfold_scipy_direct_method.solve_scipy_direct

.. autofunction:: bssunfold.core.unfold_tsvd.solve_tsvd

Regularization Selection
========================

.. autofunction:: bssunfold.core.regularization.select_regularization_parameter

.. autofunction:: bssunfold.core.regularization.lcurve_selection

.. autofunction:: bssunfold.core.regularization.gcv_selection

.. autofunction:: bssunfold.core.regularization.discrepancy_principle_selection

.. autofunction:: bssunfold.core.regularization.cosine_similarity_selection
   
