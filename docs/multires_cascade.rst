Multi-resolution cascades
=========================

What was implemented
--------------------

`unfold_cascade` (and the ``Detector.unfold_cascade`` wrapper) now supports
true coarse-to-fine multi-resolution through two complementary mechanisms:

* **Per-stage coarse grid.** A :class:`CascadeStage` may set ``coarse=True``
  (and optionally ``coarse_bins``). That stage is then solved on a reduced
  energy grid built from the original detector, and its solution is
  prolongated back to the fine grid to seed the next stage.

* **One-flag activation.** Passing ``multi_resolution=True`` to
  ``unfold_cascade`` / ``unfold_adaptive_cascade`` marks the first stage as
  coarse automatically (``coarse_bins`` defaults to
  ``max(8, n_energy_bins // 8)``).

The coarse detector is produced by :func:`bssunfold.core._multires.build_coarse_detector`:
adjacent response-matrix columns are summed
(``A_coarse[i, k] = sum_{j in bin k} A[i, j]``) so a coarse spectrum of bin
totals reproduces the same detector readings as the fine one, and the coarse
energy grid is the geometric mean of each group. The prolongation
(:func:`bssunfold.core._multires.prolongate_spectrum`) spreads each coarse
bin total uniformly across its fine bins, preserving total fluence, and is
used as the ``initial_spectrum`` for the following fine-grid stage.

Why it can help (assessment)
----------------------------

Multi-resolution / coarse-to-fine strategies are a recurring theme in
unfolding literature because the high-resolution inverse problem is
ill-conditioned: noise is amplified most strongly at the finest scales, so a
direct fine-grid solve can be unstable. Resolving the low-frequency shape on
a coarse grid first, then refining, is reported to stabilise the solution
and reduce sensitivity to noise and to the choice of starting guess:

* Reginatto et al. — sequential / Bayesian approaches where one method's
  result informs the next (prior / initial guess transfer).
* Vega-Carrillo et al. — hybrid methods chaining a fast approximate method
  with a more accurate one.
* Milian et al. — multi-resolution ideas in Bonner-sphere spectrometry
  (coarse grid first, interpolate, then refine).
* Garcia et al. — cascaded optimisation for radiation-field reconstruction.

These references are also cited in the module docstring of
``unfold_cascade.py``.

Evidence basis and limitations
------------------------------

* This assessment is based on the cited literature and on the numerical
  behaviour of the implemented mechanism. A live literature/web
  corroboration pass was **not** performed in this change set, so the
  references above should be re-checked against primary sources before any
  strong quantitative claim is made.
* The implementation is verified by unit tests
  (``tests/test_cascade.py``): the coarse detector's response matrix equals
  the column-sum coarsening of the fine one, prolongation preserves fluence,
  and a coarse-first cascade returns a finite fine-grid spectrum.
* Multi-resolution is a *stabilising prior*, not a guarantee of improved
  accuracy for every spectrum. Its benefit is expected to be largest for
  noisy inputs and for methods that are sensitive to the starting point
  (e.g. iterative / gradient-based refiners). The coarse pre-solve is used
  only as an initial guess for the fine stage; the fine stage still solves
  the full-resolution problem, so no physical information is discarded.

How to use it
-------------

.. code-block:: python

    # Cascade with a coarse first stage seeding the fine stages.
    result = detector.unfold_cascade(
        readings,
        cascade_stages=create_default_cascade("general"),
        multi_resolution=True,
    )

    # Or configure a stage explicitly.
    from bssunfold.core.unfold_cascade import CascadeStage
    stages = [
        CascadeStage(method="tsvd", use_as_initial=False, coarse=True, coarse_bins=10),
        CascadeStage(method="landweber", params={"max_iterations": 50}, use_as_initial=True),
    ]
    result = detector.unfold_cascade(readings, cascade_stages=stages)
