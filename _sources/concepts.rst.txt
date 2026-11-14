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

.. _input-scaling:

Scale your inputs to order 1
----------------------------

bauer's default priors assume that every free parameter is **of order 1**:
group means get ``Normal(mu, 0.5–10)`` priors, group SDs a
``HalfNormal(0.25)`` prior, and regression slopes ``N(0, 1)``. With a
typical sample (10–30 subjects) the **group-SD prior is not swamped by the
data**, so its scale effectively sets how much subjects are shrunk towards
the group mean. If your inputs are in units where the parameters are far
from 1, that scale is wrong by orders of magnitude:

- **Inputs too large** (e.g. stimulus differences of ~1000 ms, Hz, CHF):
  ``nu`` and ``bias`` are ~1000, so a group SD of ~0.25 forces **near-total
  shrinkage** — every subject collapses onto the group mean.
- **Inputs too small** (e.g. stimulus differences of ~10⁻³): ``bias`` is
  ~10⁻³, the group-SD prior is ~250× wider than any plausible spread, and there is
  **effectively no shrinkage** — the hierarchy does nothing. (``nu`` is
  partly protected here: below ~1 the softplus is close to ``exp``, so its
  group SD acts roughly on ``log(nu)``. Above ~1 softplus is close to the
  identity, which is why large units hit ``nu`` hard.)

The fix is to **standardize the inputs** so that the quantity a parameter
lives in is of order 1, fit, and rescale the parameters back:

.. code-block:: python

    scale = (df['x2'] - df['x1']).std()       # or a meaningful unit (e.g. 100 ms)
    df_fit = df.assign(x1=df['x1'] / scale, x2=df['x2'] / scale)
    model = PsychophysicalModel(paradigm=df_fit)
    model.build_estimation_model(data=df_fit)
    idata = model.sample()
    # nu and bias are in units of `scale`: multiply back for reporting
    nu_raw = idata.posterior['nu'] * scale

Only the *scale* matters, not the location (the psychophysical model depends
on ``x2 - x1``). When you compare several tasks, standardize each task
separately (or fit them in a common, order-1 unit) so the priors mean the same
thing in each.

Which models are affected:

- **Linear-unit models** — ``Psychophysical*`` (``x1``/``x2``): ``nu`` and
  ``bias`` are in the units of ``x``. ``ProspectTheoryModel`` and
  ``LossAversionModel`` also take raw amounts and have priors / grids written
  for amounts of roughly 1–100; check they match your units.
- **Continuous regressors** in any ``*RegressionModel``: a covariate with SD
  300 implies a slope of ~1/300 under an ``N(0, 1)`` prior. z-score it (or use
  a meaningful unit). Binary / contrast-coded factors are fine as they are.
- **Not affected**: the magnitude and risky-choice Bayesian-observer models
  (``MagnitudeComparison*``, ``Risk*``, flex / power-law, DDM / race
  variants) work on ``log(n)``, so a change of units is a constant shift that
  the prior mean absorbs; noise is a Weber fraction.

bauer raises a ``UserWarning`` from :meth:`build_estimation_model` when the
typical stimulus difference (psychophysics) or a continuous regressor's SD
falls outside ``model.input_scale_range`` (default ``(0.2, 5)``). Silence it
with ``model.warn_on_input_scale = False`` if the units are deliberate (e.g.
you also passed matching custom priors).

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
    from bauer.utils import convergence_report
    convergence_report(idata)       # r̂ ≤ 1.01, bulk ESS ≥ 400, no divergences → OK

    # Parameter summaries: posterior mean + 95% HDI from long-format draws
    from bauer.utils import posterior_draws, summarize_posterior
    group = summarize_posterior(posterior_draws(idata, ['n1_evidence_sd_mu', 'n2_evidence_sd_mu']),
                                'parameter')
    subjects = summarize_posterior(posterior_draws(idata, ['n1_evidence_sd', 'n2_evidence_sd']),
                                   ['parameter', 'subject'])

    # Posterior predictive check: one row per trial x posterior sample
    ppc = model.ppc(data, idata)

See :ref:`summarising-posteriors` for regression models, contrasts, forest plots and
group- and subject-level posterior predictive checks.

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
