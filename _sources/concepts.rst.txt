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

**Psychophysical models** (``PsychophysicalModel`` and variants)
    Two-alternative forced choice fitting a **psychometric function** with a
    sensitivity parameter ``nu`` and a bias ``bias``.  Requires columns
    ``x1``, ``x2``, ``choice``.

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
- All paradigm-specific columns must be present and named exactly: magnitude
  ``n1``/``n2``; risky choice ``n1``/``n2``/``p1``/``p2``.
- **For DDM/RDM (reaction-time) models**, also include ``rt`` in **seconds**
  (> 0). Always **drop implausibly fast trials** (e.g. ``df = df[df['rt'] >= 0.20]``):
  the WFPT likelihood has a flat, zero-gradient region when the non-decision
  time exceeds the fastest RT, and the sampler gets stuck there.

Using your own data
-------------------

Point a model at any DataFrame that follows the conventions above:

.. code-block:: python

    import pandas as pd
    from bauer.models import RiskModel              # or DDMRiskModel, etc.

    df = pd.read_csv('my_choices.tsv', sep='\t')    # your data
    df['choice'] = df['chose_second'].astype(bool)  # boolean: True = option 2
    df = df.set_index('subject')                    # subject as index (hierarchical)
    # df must have n1, n2, p1, p2  (+ rt in seconds for DDM/RDM models)

    model = RiskModel(paradigm=df)
    model.build_estimation_model(data=df, hierarchical=True)

Typical workflow
----------------

.. code-block:: python

    from bauer.models import MagnitudeComparisonModel
    from bauer.utils.data import load_garcia2022

    data = load_garcia2022(task='magnitude')

    model = MagnitudeComparisonModel(paradigm=data)
    model.build_estimation_model(data=data, hierarchical=True)

    # Full MCMC. backend='numpyro' is much faster than the default 'pymc'
    # (and uses the GPU automatically if jax[cuda12] is installed) — see below.
    idata = model.sample(draws=1000, tune=1000, backend='numpyro')

    # ALWAYS check convergence before interpreting anything:
    import arviz as az
    print(az.summary(idata, kind='diagnostics'))    # want r̂ ≤ 1.01, ESS ≥ ~400

    # Parameter summaries
    subj_pars = model.get_subjectwise_parameter_estimates(idata)
    group_pars = model.get_groupwise_parameter_estimates(idata)

    # Posterior predictive check
    ppc = model.ppc(data, idata)

Sampler backends and speed
--------------------------

``model.sample(backend=...)`` chooses the NUTS engine:

- ``'pymc'`` (**default**) — reliable but **slow**.
- ``'numpyro'`` — JAX-backed, **much faster** (~3–10× on CPU, ~5–30× on GPU),
  and it parallelises the chains for you (``chain_method`` is auto-set to
  ``'vectorized'``).

Use ``backend='numpyro'`` for any non-trivial fit. **GPU** needs no code change
— install ``jax[cuda12]`` (see :doc:`installation`) and the same call uses the
GPU. For DDM/RDM fits use ``tune=2000`` (not the ``tune=1000`` default — too low
for those), keep the default starting-point finder on (next section), and check
r̂.


Sampling: starting points (initial values)
------------------------------------------

DDM and regression posteriors are long, curved ridges, and *where the chains
start* strongly affects whether they converge. With a generic initialization
this is effectively a lottery: the same model and settings can give
:math:`\hat r \approx 1.0` on one random seed and :math:`\hat r > 3` on the
next.

bauer handles this with a **starting-point finder**, enabled by default on the
DDM/race models (:attr:`recommended_init = 'mapjitter'`). It starts each chain
at a **data-informed plausible value** — the posterior mode from
:meth:`~bauer.core.BaseModel.fit_map`, falling back to the prior-central point —
and **disperses the chains around it by a fraction of each parameter's prior
SD**. Chains sit around the typical set rather than all at the mode, so
:math:`\hat r` stays meaningful. This is the same idea HSSM uses (curated
initial values plus a small jitter). It works for *every* parameter
automatically — core DDM, the Bayesian-observer front-end, and flexible
(B-spline) noise coefficients — with no per-parameter tuning.

You normally do nothing; it is on by default. To override::

    idata = model.sample(draws=1000, tune=2000, find_init=False)   # disable
    idata = model.sample(draws=1000, tune=2000, initvals=my_initvals)  # your own

It is built by :meth:`~bauer.core.BaseModel.get_initial_points`.

.. note::

   For large hierarchical fits a single chain can still occasionally wander off
   and get stuck (chance) — inflating :math:`\hat r` across many parameters at
   once. Check the **group-level** (``*_mu``) :math:`\hat r` and, if one chain
   is stuck, simply re-run with a different ``random_seed`` (and/or a longer
   ``tune``). A handful of weakly-identified subject-level parameters with
   mildly elevated :math:`\hat r` is normal and fine — that is honest
   uncertainty, not a failed fit.
