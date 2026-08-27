Interpreting Unfolding Results
==============================

Neutron spectrum unfolding is an *ill-posed inverse problem*: the number of
energy groups far exceeds the number of detector readings, so many spectra fit
the measurements equally well. A single ``spectrum`` vector is therefore not
enough to trust a result — you need to know *how* the solution was obtained,
*which* constraints and measurements drive it, and *how stable* it is.

bssunfold provides exactly this through the **interpretation** API, built on the
optional `pyoptexplain <https://pypi.org/project/pyoptexplain/>`_ package
(post-optimality analysis of quadratic programs). Install it with:

.. code-block:: bash

   pip install bssunfold[interpret]

The interpretation solves the *same* unfolding QP used by
``unfold_qpsolvers`` / ``unfold_cvxpy`` (``min 1/2 xᵀQx + cᵀx`` with ``x ≥ 0``)
but keeps the solver internals, dual variables and perturbation results, and
turns them into a human-readable report plus quantitative metrics.

What the interpretation answers
-------------------------------

.. mermaid::

   graph TD
       A["Interpretation (pyoptexplain)"] --> B["Solve report"]
       A --> C["Shadow prices (duals)"]
       A --> D["Robustness"]
       A --> E["Detector sensitivity"]
       A --> F["Regularization sweep"]
       A --> G["Non-negativity trust"]
       A --> H["Scenarios"]

       B --> B1["solver status, objective"]
       B --> B2["active / zeroed energy groups"]
       C --> C1["price of each x ≥ 0 bound"]
       C --> C2["price of sum(x) == norm"]
       D --> D1["spectrum change for ±1…5% readings"]
       E --> E1["importance per detector sphere"]
       F --> F1["solution across a grid of α"]
       G --> G1["effect of allowing small negatives"]
       H --> H1["structured what-if cases"]

       style A fill:#4a90d9,color:#fff

* **Solve report** — solver status, objective value, condition number and which
  energy groups are *active* (zeroed at the bound ``x = 0``).
* **Shadow prices (duals)** — how much the objective would change if a zeroed
  group were allowed to become positive. Large prices flag energy ranges the
  data "wants" but the non-negativity constraint keeps at zero.
* **Robustness** — how the unfolded spectrum moves when the readings are
  perturbed by ±1…5%. A stable solution barely changes.
* **Detector sensitivity / importance** — how the spectrum changes when *one*
  detector reading at a time is perturbed. This ranks the spheres by how much
  information they carry about the solution.
* **Regularization sweep** — how the solution and residual change across a grid
  of the regularization parameter α, revealing whether the result is
  α-sensitive.
* **Non-negativity trust** — what happens if small negative values are allowed,
  i.e. whether ``x ≥ 0`` is forcing the solution into an artificial corner.
* **Scenarios** — structured what-if cases over a custom scenario space.

API overview
------------

.. list-table::
   :header-rows: 1
   :widths: 8 14 25

   * - Level
     - Callable
     - Purpose
   * - High-level
     - ``Detector.unfold_interpret``
     - Unfold the readings and append ``report`` and ``interpretation_metrics``
       to the standard result dict.
   * - Analysis-only
     - ``Detector.interpret_result``
     - Run the interpretation directly on readings without the unfolding
       bookkeeping; returns ``report``, ``metrics`` and ``tables``.
   * - Low-level
     - ``bssunfold.core.unfold_interpret.interpret_qp``
     - Full QP interpretation from a response matrix ``A`` and readings ``b``.
   * - Low-level
     - ``bssunfold.core.unfold_interpret.build_interpretation_qp``
     - Build the QP (matrix ``Q``, vector ``c``, bounds) without solving.
   * - Low-level
     - ``bssunfold.core.unfold_interpret.solve_interpret``
     - Solve a QP and collect solver/duals/perturbation diagnostics.

Basic usage
-----------

.. code-block:: python

   from bssunfold import Detector, RF_LANL

   detector = Detector(RF_LANL)
   readings = {"3in": 0.5, "5in": 1.2, "8in": 2.1, "12in": 3.4}

   # 1) Unfold and interpret in one call
   result = detector.unfold_interpret(readings, tolerance=1e-5)

   spectrum  = result["spectrum"]              # unfolded flux per energy bin
   report    = result["report"]                # full Markdown report
   metrics   = result["interpretation_metrics"]  # JSON-friendly diagnostics

   # 2) Analysis-only: same interpretation without the unfolding wrapper
   ir = detector.interpret_result(readings, tolerance=1e-5)
   tables = ir["tables"]  # pandas DataFrames: summary, duals, detectors, ...

.. note::

   **About ``tolerance``:** pyoptexplain's backend can report ``iteration_limit``
   on the full 11-sphere ``RF_LANL`` problem at the strictest default tolerance
   (``1e-8``). Relaxing the feasibility/optimality tolerance to ``1e-5`` is
   enough for an ``optimal`` status with a residual below 0.3 % — a good
   compromise for analysis-grade work. See the worked example in
   ``examples/24-interpret.ipynb``.

How to use it to interpret a spectrum
-------------------------------------

Once you have ``result["report"]``, read it in this order:

1. **Check the solve status and residual.** If the solver is not ``optimal``,
   or the residual is much larger than the measurement noise, the response
   matrix and the readings are inconsistent — check the detector set and the
   readings before trusting any spectrum shape.

2. **Look at the active (zeroed) groups.** Unfolding is sparse by nature: many
   energy groups end up pinned at ``x = 0``. This tells you where the solution
   is *determined* by the data and where it is only constrained by the
   non-negativity bound. A spectrum that is concentrated in a few groups is a
   sparse, low-information solution; a "washed-out" spectrum with everything
   active usually means the data cannot resolve the energy structure.

3. **Read the shadow prices.** A large dual on a zeroed group means the data
   would prefer a *positive* flux there but the constraint forbids it. If the
   largest duals cluster in one energy range (e.g. a fast peak or a thermal
   component), that range is the least constrained part of the spectrum — the
   measurement set has little resolving power there, so the value is an
   artifact of the regularization, not of the data.

4. **Check robustness.** Compare the relative spectrum change under ±1…5 %
   reading perturbations. If the solution moves by more than the perturbation
   level, the unfolding is unstable for this detector set and you should add
   spheres (or a prior) before drawing quantitative conclusions.

5. **Check non-negativity trust.** If allowing 5 % negative values moves the
   spectrum only slightly, the ``x ≥ 0`` bound is not distorting the solution.
   A large change means the physical spectrum actually needs negative
   components — typically a sign that the response functions or readings are
   miscalibrated.

How to use it to choose the detector sphere set
-----------------------------------------------

The interpretation is also a **sphere-selection tool**. The
``detector_importance`` table ranks the spheres by how much the unfolded
spectrum changes when each reading is perturbed on its own:

* **High-importance spheres** dominate the solution — the spectrum is tightly
  pinned by their readings. In the ``RF_LANL`` Cf-252 example the lead-shielded
  and mid-size spheres (``9inPb``, ``12inPb``, ``8in``) carry most of the
  information.
* **Low-importance spheres** (e.g. the small ``3in`` sphere on a hard fast
  spectrum) barely influence the solution. Their readings are consistent with
  the rest of the set but add almost no information — they can be removed
  without changing the unfolded spectrum, saving measurement time.
* **High-residual spheres** (from the per-detector residual table) disagree
  with the reconstructed spectrum. Re-examine their calibration or response
  function before trusting the result.

The **regularization sweep** complements this: if the residual stays flat over
a wide range of α, the sphere set already provides enough information and the
choice of regularization is not critical. If the residual changes sharply with
α, the set is under-determined and more (or better chosen) spheres are needed.

Worked example
--------------

Run the notebook ``examples/24-interpret.ipynb`` for a complete, reproducible
walkthrough on the built-in ``RF_LANL`` response functions with the ISO Cf-252
reference spectrum. It produces the full Markdown report, the quantitative
metrics tables and the key diagnostic plots (unfolded spectrum vs. reference,
per-detector fit, detector importance, sensitivity and non-negativity).
