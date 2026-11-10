"""Bayesian observer for magnitude reproduction (e.g. interval timing).

The observer works in log space, where Weber's law makes the noise constant:

1. Measurement  ``m ~ N(log x, nu^2)`` (optionally ``+ bias``)
2. Prior        ``log x ~ N(mu_p, sigma_p^2)``; ``prior_mu`` (mu_p) is fitted
                by default, ``prior_sd`` (sigma_p) optionally; otherwise both
                are the mean and SD of the presented log stimuli
3. Posterior mean ``m_hat = w m + (1 - w) mu_p``, ``w = sigma_p^2 / (sigma_p^2 + nu^2)``
4. Response     ``log r = m_hat``
5. Lapses       with probability ``p_lapse`` a response uniform over the
                (log) response range

The trial-to-trial spread of reproductions of a fixed stimulus is the spread
of the **posterior mean across measurements**, ``SD[log r | x] = w * nu``. It
is NOT the posterior SD on a single trial (``sqrt(w) * nu``), which is the
observer's within-trial uncertainty, and not the measurement noise ``nu``.
Using either of those as the response SD predicts reproductions that are too
variable (and biases ``nu`` down to compensate).

Likelihood::

    log r ~ (1 - p_lapse) N(w log x + (1 - w) mu_p, (w nu)^2)
            + p_lapse Uniform(log r_min, log r_max)

What the data identify: the slope of mean log reproduction against log
stimulus is ``w``, the intercept is ``(1 - w) mu_p``, and the spread around the
line is ``w * nu``. So ``mu_p`` is identified (as long as w < 1, i.e. there is
some central tendency), and with ``fit_prior_sd=True`` both ``nu`` (= spread /
slope) and ``sigma_p`` (from w and nu) are identified too. A measurement bias
(``fit_bias=True``, "clock speed") only shifts the intercept as well
(``w * bias + (1 - w) * mu_p``), so fit either ``prior_mu`` or ``bias``, not
both.

In natural units the reproductions of a fixed stimulus are log-normal with
log-SD ``w * nu``: their SD grows in proportion to the stimulus (constant
coefficient of variation, the scalar property) and their skewness,
``(e^{s^2} + 2) sqrt(e^{s^2} - 1)`` with ``s = w nu``, is the same at every
magnitude. Skew changing with magnitude is evidence for a noise source that
is not constant in log space (e.g. additive motor noise).
Optional ``fit_motor_noise=True`` adds log-space motor noise in quadrature,
``SD = sqrt((w nu)^2 + sigma_motor^2)``; that uses up the spread, so it should
not be combined with ``fit_prior_sd=True`` (nu, sigma_p and sigma_motor would
trade off).
"""
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from ..core import BaseModel, RegressionModel
from ..utils.math import inverse_softplus_np, logit_np


class ReproductionModel(BaseModel):
    """Log-space Bayesian observer for reproduction tasks.

    Data columns: ``stimulus`` (positive, e.g. seconds), ``response`` (same
    units), and ``subject`` in the index for hierarchical fits.

    Parameters
    ----------
    paradigm : pd.DataFrame
    response_range : (float, float), optional
        Range of possible responses (natural units), used for the uniform
        lapse density. Default: observed range ± 20%.
    prior_mu, prior_sd : float, optional
        Observer prior on log stimulus. Default: mean and SD of
        ``log(stimulus)`` in ``paradigm`` (the presented distribution). When
        fitted, these are the starting points (prior centres) instead.
    fit_prior_mu : bool
        Fit the prior mean ``prior_mu`` per subject (default True). Its group
        prior is centred on the presented mean.
    fit_prior_sd : bool
        Fit the prior width ``prior_sd`` per subject (default False: fixed at
        the presented SD). Identified from slope + spread when
        ``fit_motor_noise`` is False.
    fit_bias : bool
        Fit an additive measurement bias in log space (default False).
        Mutually exclusive with ``fit_prior_mu`` (same intercept).
    fit_lapse : bool
        Include the uniform lapse component (default True).
    fit_motor_noise : bool
        Add log-space motor noise ``sigma_motor`` (default False).
    stimulus_col, response_col : str
        Column names (defaults ``'stimulus'`` / ``'response'``).
    """

    def __init__(self, paradigm, response_range=None, prior_mu=None, prior_sd=None,
                 fit_prior_mu=True, fit_prior_sd=False, fit_bias=False,
                 fit_lapse=True, fit_motor_noise=False,
                 stimulus_col='stimulus', response_col='response'):
        self.stimulus_col = stimulus_col
        self.response_col = response_col
        self.paradigm_keys = []
        log_x = np.log(paradigm[stimulus_col].values.astype(float))
        self.prior_mu = float(log_x.mean()) if prior_mu is None else float(prior_mu)
        self.prior_sd = float(log_x.std()) if prior_sd is None else float(prior_sd)
        if response_range is None:
            r = paradigm[response_col].values.astype(float)
            response_range = (r.min() / 1.2, r.max() * 1.2)
        self.log_response_range = tuple(np.log(np.asarray(response_range, dtype=float)))
        self.fit_lapse = fit_lapse
        self.fit_motor_noise = fit_motor_noise
        self.fit_prior_sd = fit_prior_sd
        self.fit_prior_mu = fit_prior_mu
        self.fit_bias = fit_bias
        if fit_prior_mu and fit_bias:
            raise ValueError('fit_prior_mu and fit_bias both only shift the intercept '
                             '(w*bias + (1-w)*prior_mu): fit one of them.')
        if fit_prior_sd and fit_motor_noise:
            import warnings
            warnings.warn('fit_prior_sd together with fit_motor_noise is weakly identified: '
                          'nu, prior_sd and sigma_motor trade off.', UserWarning)
        super().__init__(paradigm)

    def get_free_parameters(self):
        pars = {
            'nu': {'mu_intercept': inverse_softplus_np(0.3), 'sigma_intercept': 0.5,
                   'transform': 'softplus'},
        }
        if self.fit_prior_mu:
            pars['prior_mu'] = {'mu_intercept': self.prior_mu, 'sigma_intercept': 0.5,
                                'transform': 'identity'}
        if self.fit_bias:
            pars['bias'] = {'mu_intercept': 0.0, 'sigma_intercept': 0.3, 'transform': 'identity'}
        if self.fit_prior_sd:
            pars['prior_sd'] = {'mu_intercept': inverse_softplus_np(self.prior_sd),
                                'sigma_intercept': 0.5, 'transform': 'softplus'}
        if self.fit_motor_noise:
            pars['sigma_motor'] = {'mu_intercept': inverse_softplus_np(0.1),
                                   'sigma_intercept': 0.5, 'transform': 'softplus'}
        if self.fit_lapse:
            # wide between-subject spread: lapse rates differ a lot between people
            pars['p_lapse'] = {'mu_intercept': logit_np(0.02), 'sigma_intercept': 1.0,
                               'cauchy_sigma_intercept': 1.0, 'transform': 'logistic'}
        return pars

    def _get_paradigm(self, paradigm=None, subject_mapping=None):
        p = super()._get_paradigm(paradigm, subject_mapping=subject_mapping)
        p['log_stimulus'] = np.log(paradigm[self.stimulus_col].values.astype(float))
        if self.response_col in paradigm.columns:
            p['log_response'] = np.log(paradigm[self.response_col].values.astype(float))
        return p

    def get_model_inputs(self, parameters):
        return parameters

    def predicted_log_response(self, log_stimulus, parameters):
        """Mean and SD of log reproductions (excluding lapses)."""
        nu = parameters['nu']
        bias = parameters['bias'] if self.fit_bias else 0.0
        prior_mu = parameters['prior_mu'] if self.fit_prior_mu else self.prior_mu
        prior_sd = parameters['prior_sd'] if self.fit_prior_sd else self.prior_sd
        w = prior_sd ** 2 / (prior_sd ** 2 + nu ** 2)
        mean = w * (log_stimulus + bias) + (1 - w) * prior_mu
        sd = w * nu                                   # SD of the posterior MEAN
        if self.fit_motor_noise:
            sd = pt.sqrt(sd ** 2 + parameters['sigma_motor'] ** 2)
        return mean, sd

    def build_likelihood(self, parameters, save_p_choice=False):
        model = pm.Model.get_context()
        mean, sd = self.predicted_log_response(model['log_stimulus'], parameters)
        if not self.fit_lapse:
            pm.Normal('ll', mean, sd, observed=model['log_response'])
            return
        lapse = parameters['p_lapse']
        n = mean.shape[0]
        lo, hi = self.log_response_range
        pm.Mixture('ll', w=pt.stack([1 - lapse, lapse], axis=-1),
                   comp_dists=[pm.Normal.dist(mean, sd),
                               pm.Uniform.dist(pt.full((n,), lo), pt.full((n,), hi))],
                   observed=model['log_response'])

    def ppc(self, paradigm, idata, n_posterior_samples=200, random_seed=None,
            progressbar=False):
        """Posterior-predictive reproductions for the fitted data.

        Returns a DataFrame indexed by ``paradigm.index`` + ``ppc_sample`` with
        column ``simulated_response`` (natural units), like the choice models'
        ``ppc``. Uses ``n_posterior_samples`` evenly thinned posterior draws.
        """
        post = idata.posterior
        n_total = post.sizes['chain'] * post.sizes['draw']
        step = max(1, n_total // n_posterior_samples)
        stacked = post.stack(sample=('chain', 'draw'))
        thinned = stacked.isel(sample=slice(None, None, step)).isel(
            sample=slice(0, n_posterior_samples))
        thinned = thinned.drop_vars(['sample', 'chain', 'draw'], errors='ignore')
        thinned = thinned.rename({'sample': 'draw'}).expand_dims('chain')
        with self.estimation_model:
            pp = pm.sample_posterior_predictive(thinned, var_names=['ll'],
                                                random_seed=random_seed,
                                                progressbar=progressbar)
        arr = np.exp(pp.posterior_predictive['ll'].values[0])    # (samples, trials)
        assert arr.shape[1] == len(paradigm), 'PPC trial count does not match paradigm'
        out = pd.DataFrame(arr.T, index=paradigm.index,
                           columns=pd.Index(np.arange(arr.shape[0]), name='ppc_sample'))
        return out.stack().rename('simulated_response').to_frame()


class ReproductionRegressionModel(RegressionModel, ReproductionModel):
    """:class:`ReproductionModel` with patsy formulas on any parameter.

    ``fixed_regressors``: population design per parameter;
    ``random_regressors``: which terms also vary per subject (default:
    random intercept only). E.g. for a within-subject drug design::

        f = "C(condition, Treatment('placebo')) + session_c"
        ReproductionRegressionModel(df, fixed_regressors={'nu': f, 'bias': f},
                                    random_regressors={'nu': f, 'bias': f})
    """

    def __init__(self, paradigm, regressors=None, fixed_regressors=None,
                 random_regressors=None, **kwargs):
        RegressionModel.__init__(self, regressors, fixed_regressors=fixed_regressors,
                                 random_regressors=random_regressors)
        ReproductionModel.__init__(self, paradigm, **kwargs)
