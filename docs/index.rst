bauer
=====

**Bayesian Estimation of Perceptual, Numerical and Risky Choice.**

*bauer* is a PyMC-based Python library for fitting hierarchical Bayesian cognitive
models to behavioural decision-making data.  It covers magnitude comparison,
psychometric functions, and risky choice — from simple Weber's-law models to
flexible noise curves and prospect-theory variants.

Key features
------------

- **Ready-to-use model classes** for magnitude comparison, psychometric functions,
  and risky choice — no need to hand-code PyMC models.
- **Hierarchical fitting by default**: group mean + between-subject SD inferred
  jointly with subject-level parameters.  Essential for typical trial counts
  (100–250 per condition).
- **Regression support** via patsy formulas: e.g. ``regressors={'nu': 'C(condition)'}``
  to let any parameter vary by experimental condition.
- **Posterior predictive checks** with ``model.ppc(data, idata)``.
- **Full ArviZ integration**: trace diagnostics, HDI plots, ELPD model comparison.
- **Included datasets**: Garcia et al. (2022) magnitude/risk, de Hollander et al.
  (2024) dot-cloud and symbolic gambles.

Quick start
-----------

.. code-block:: python

   from bauer.models import MagnitudeComparisonModel
   from bauer.utils.data import load_garcia2022

   data = load_garcia2022(task='magnitude')
   model = MagnitudeComparisonModel(paradigm=data, fit_seperate_evidence_sd=True)
   model.build_estimation_model(data=data, hierarchical=True)
   idata = model.sample(draws=1000, tune=1000)

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   concepts

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorial/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api_reference

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
