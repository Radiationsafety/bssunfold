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

Core Functions
==============

Underlying solver functions:

.. autofunction:: bssunfold.core.unfolding_methods.solve_cvxpy

.. autofunction:: bssunfold.core.unfolding_methods.solve_landweber

.. autofunction:: bssunfold.core.unfolding_methods.solve_mlem

.. autofunction:: bssunfold.core.unfolding_methods.solve_qpsolvers

.. autofunction:: bssunfold.core.unfolding_methods.solve_doroshenko

.. autofunction:: bssunfold.core.unfolding_methods.solve_kaczmarz

.. autofunction:: bssunfold.core.unfolding_methods.solve_lmfit

Regularization Selection
========================

.. autofunction:: bssunfold.core.regularization.select_regularization_parameter

.. autofunction:: bssunfold.core.regularization.lcurve_selection

.. autofunction:: bssunfold.core.regularization.gcv_selection

.. autofunction:: bssunfold.core.regularization.discrepancy_principle_selection

.. autofunction:: bssunfold.core.regularization.cosine_similarity_selection
