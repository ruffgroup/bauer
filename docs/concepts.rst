Architecture & Concepts
=======================

bauer is built around a composable class hierarchy that separates the probabilistic
scaffolding (PyMC model construction, MCMC, parameter extraction) from the
model-specific likelihood (how stimuli map to choice probabilities).

Class hierarchy
---------------

:class:`~bauer.core.BaseModel` is the abstract base for every model.  It handles:

- Building PyMC hierarchical or flat priors from a ``free_parameters`` dict
- MCMC sampling via :meth:`~bauer.core.BaseModel.sample`
- MAP estimation via :meth:`~bauer.core.BaseModel.fit_map`
- Simulating choices via :meth:`~bauer.core.BaseModel.simulate`
- Posterior predictive checks via :meth:`~bauer.core.BaseModel.ppc`
- Extracting subject- and group-level parameter posteriors

Two mixins extend ``BaseModel``:

:class:`~bauer.core.LapseModel`
    Adds a ``p_lapse`` parameter so that on a fraction of trials the agent
    responds randomly.  The choice probability becomes
    :math:`p \cdot (1 - p_\text{lapse}) + 0.5 \cdot p_\text{lapse}`.

:class:`~bauer.core.RegressionModel`
    Adds `patsy <https://patsy.readthedocs.io>`_ formula support so that any
    free parameter can be a linear function of trial-level covariates.  Pass a
    ``regressors`` dict mapping parameter names to formula strings when
    constructing the model.

Concrete models are created via multiple inheritance, e.g.::

    class MagnitudeComparisonLapseRegressionModel(
        LapseModel, MagnitudeComparisonRegressionModel
    ): ...

Model families
--------------

**Psychometric models** (``PsychometricModel`` and variants)
    Two-alternative forced choice with a sensitivity parameter ``nu`` and a
    bias ``bias``.  Requires columns ``x1``, ``x2``, ``choice``.

**Magnitude comparison models** (``MagnitudeComparisonModel``, ``FlexibleNoiseComparisonModel``)
    Bayesian observer models for choosing between two numerical quantities.
    Stimuli are represented in log space, corrupted by Gaussian noise, and
    compared against a prior over the stimulus distribution.  Requires
    columns ``n1``, ``n2``, ``choice``.

**Risky choice models** (``RiskModel``, ``ProspectTheoryModel``, ``LossAversionModel``, ``RNPModel``, ``FlexibleNoiseRiskModel``, ``ExpectedUtilityRiskModel``)
    Models for decisions between monetary lotteries defined by outcomes and
    probabilities.  Requires columns ``n1``, ``n2``, ``p1``, ``p2``, ``choice``
    (or ``gain``, ``loss``, ``prob_gain`` for Prospect Theory).

Parameter transforms
--------------------

Each free parameter has an associated transform that maps it from an unbounded
normal prior to a valid range:

.. list-table::
   :header-rows: 1

   * - Transform
     - Range
     - Typical use
   * - ``identity``
     - :math:`(-\infty, \infty)`
     - bias, prior mean
   * - ``softplus``
     - :math:`(0, \infty)`
     - noise SDs
   * - ``logistic``
     - :math:`(0, 1)`
     - lapse rate

Data conventions
----------------

- The ``choice`` column must be **boolean** (``True`` = chose option 2 / second alternative).
- For hierarchical models, ``subject`` must appear either as an index level or as a column.
- All paradigm-specific columns (``n1``, ``n2``, ``p1``, ``p2``, …) must be present
  and named exactly as the model expects.

Typical workflow
----------------

.. code-block:: python

    from bauer.models import MagnitudeComparisonModel
    from bauer.utils.data import load_garcia2022

    data = load_garcia2022(task='magnitude')

    model = MagnitudeComparisonModel(paradigm=data)
    model.build_estimation_model(data=data, hierarchical=True)

    # Quick MAP estimate
    pars = model.fit_map()

    # Full MCMC
    idata = model.sample(draws=1000, tune=1000)

    # Parameter summaries
    subj_pars = model.get_subjectwise_parameter_estimates(idata)
    group_pars = model.get_groupwise_parameter_estimates(idata)

    # Posterior predictive check
    ppc = model.ppc(data, idata)
