Estimation tasks (continuous reports)
=====================================

``bauer.models.EstimationTaskModel`` is a hierarchical Bayesian observer for
tasks in which a participant sees a magnitude ``n`` and *reports* it on a
bounded scale (a slider, a typed number).  It is built from three components.

=================  ===========================================================
Component          Options
=================  ===========================================================
Encoding space     ``LogSpace()`` (Weber), ``LinearSpace()`` (constant noise),
                   ``PowerSpace()`` (Box-Cox exponent ``alpha`` fitted per
                   participant; ``PowerSpace(a)`` fixes it)
Prior              ``UniformPrior(lo, hi)`` (e.g. the task's range) or
                   ``GaussianPrior()`` (Gaussian in the encoding space:
                   log-normal for log space; centre and width fitted)
Lapse              uniform over the response options (``lapse=True``)
=================  ===========================================================

The readout is the posterior mean of ``n``.  It is rounded to the response
options and censored at the scale ends (``response_range``).  The free noise
``nu`` is expressed in log units at ``n_ref`` (a Weber fraction) for every
encoding space.

Quick start
-----------

.. code-block:: python

   from bauer.models import EstimationTaskModel, LogSpace, UniformPrior

   # data: columns n (presented) and response (integer), index level 'subject'
   model = EstimationTaskModel(data, encoding_space=LogSpace(),
                               prior=UniformPrior(4.5, 24.5), response_range=(5, 25))
   model.build_estimation_model(data)
   idata = model.sample(draws=1500, tune=1500, target_accept=0.95)

   ppc = model.posterior_predictive(idata, n_posterior_samples=200)  # simulated datasets
   sim = model.simulate(data, parameters_by_subject)                 # synthetic data
   table, _ = model.parameter_recovery(data, parameters_by_subject)  # simulate -> refit

Because the likelihood is an observed ``pm.Categorical`` over the response
options, ``pm.compute_log_likelihood(idata, model=model.estimation_model)``
gives pointwise log-likelihoods for ``az.loo`` / ``az.compare``.  Subject-level
covariates on a parameter's group mean: ``covariates={'nu': 'column'}``.

How the likelihood is computed
------------------------------

*Closed form* (log space + Gaussian prior, i.e. log-normal): the posterior
over ``log n`` is Gaussian, the posterior-mean estimate is log-normal in ``n``,
and the probability of each response option is a difference of normal CDFs.

*Grid* (all other combinations, e.g. the uniform prior or a free ``alpha``):
for every participant the estimate ``n_hat(m)`` is computed on a measurement
grid (posterior mean over a hypothesis grid) and inverted at the edges of the
response options.  Because ``n_hat`` is monotone in ``m``, the probability of
each option is again a difference of normal CDFs of the measurement.  This is
exact up to linear interpolation, and smooth in the parameters, so NUTS gets
usable gradients.  The two paths agree to < 2e-3 per option probability (tested).

Efficient coding and power spaces
---------------------------------

An efficient (infomax) encoder for a stimulus density ``p_enc`` uses the CDF of
``p_enc`` as its internal space, with constant noise there (Wei & Stocker,
2015).  For a power-law density ``p_enc ~ n^-kappa`` that is ``PowerSpace`` with
``alpha = 1 - kappa``.  So the log space is efficient for a ``1/n`` ("natural")
prior, and the linear space is efficient for a uniform prior.  A fitted
``alpha`` can be read as the exponent of the stimulus statistics the encoding is
adapted to.  It is related to, but not the same as, Stevens' exponent of the
*reports*: the Bayesian readout undoes most of the transform and adds
compression toward the prior.
