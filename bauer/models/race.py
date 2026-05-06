"""Generalized Bayesian race-diffusion (RDM) variants of bauer's choice models.

Each stimulus drives its own Wiener accumulator with drift

    μ_i = w_0 + w_d·(tilde_i - tilde_j) + w_s·(tilde_i + tilde_j)
                                          (default, ``advantage=True``;
                                           van Ravenzwaaij 2020)
    μ_i = w_0 + (μ_post,i - μ_p)          (ablation, ``advantage=False``)

racing to a common threshold ``a`` with diffusion noise σ = 1. The first-
passage time per accumulator is inverse Gaussian (analytical, no LANs).
Likelihood combines the winner's IG density with the losers' survival
functions. Reference: Tillman, Van Zandt & Logan (2020).

Conceptual reading (sequential evidence stream):
    The within-trial Wiener noise σ *is* the per-unit-time sensory noise; the
    accumulator state at time t is the agent's running estimate of log s_i
    given evidence so far. Across-trial drift variability is **not** a
    separate parameter — sensory uncertainty is fully expressed through the
    within-trial diffusion. Adding s_v on top would double-count.
    See Bogacz et al. (2006), Drugowitsch et al. (2012).

Magnitude effect on RT (faster RTs at larger stakes) emerges from the Bayesian
front-end (larger μ_post → larger drift → faster race) without any explicit
RT-magnitude parameter — its curvature is determined by σ_sens and σ_p, both
already constrained by the choice data.

HSSM convention for the observed array: column 0 = |rt|, column 1 = signed
response (+1 = accumulator 2 wins / choice=True, -1 = accumulator 1 wins).
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .magnitude import (
    MagnitudeComparisonModel, FlexibleNoiseComparisonModel,
    PowerLawNoiseComparisonModel, PowerLawNoiseComparisonRegressionModel,
)
from .risky_choice import (
    RiskModel, FlexibleNoiseRiskModel, FlexibleNoiseRiskRegressionModel,
    PowerLawNoiseRiskModel, PowerLawNoiseRiskRegressionModel,
)
from ..utils.bayes import get_posterior
from ..utils.math import inverse_softplus_np


_LOG_2PI = float(np.log(2 * np.pi))
_SQRT_2 = float(np.sqrt(2.0))


def _ndtr(x):
    """Standard normal CDF via erf, for pytensor."""
    return 0.5 * (1.0 + pt.erf(x / _SQRT_2))


def logp_race_diffusion_2(data, v1, v2, sigma1, sigma2, a, t):
    """Analytical log-likelihood for a 2-accumulator Wald (Wiener) race
    with per-accumulator drift AND noise.

    Each accumulator k follows ``dx_k = v_k dt + sigma_k dW`` until it hits
    absorbing barrier ``a`` (assumed common). First-passage time is inverse
    Gaussian with mean ``mu_k = a / v_k`` and shape ``lambda_k = a^2 / sigma_k^2``.

    Parameters
    ----------
    data : (n, 2) tensor
        Column 0: response time (positive). Column 1: signed response in
        ``{-1, +1}`` (+1 = accumulator 2 wins, HSSM convention).
    v1, v2 : positive scalar or per-trial vector pytensor
        Drift rates. Must be positive for the race to terminate.
    sigma1, sigma2 : positive scalar or per-trial vector pytensor
        Per-accumulator diffusion SDs.
    a : positive scalar or per-trial pytensor
        Common absorbing threshold.
    t : non-negative scalar or per-trial pytensor
        Non-decision time (in same units as rt, e.g. seconds).
    """
    data = pt.reshape(data, (-1, 2))
    rt = pt.abs(data[:, 0])
    response = data[:, 1]

    decision_time = pt.maximum(rt - t, 1e-12)

    mu1 = a / v1
    mu2 = a / v2
    lam1 = a ** 2 / sigma1 ** 2
    lam2 = a ** 2 / sigma2 ** 2

    def logpdf_ig(td, mu, lam):
        return (0.5 * pt.log(lam) - 0.5 * _LOG_2PI - 1.5 * pt.log(td)
                - lam * (td - mu) ** 2 / (2.0 * mu ** 2 * td))

    def logsf_ig(td, mu, lam):
        # Survival = Phi(-z1) - exp(2*lam/mu) * Phi(-z2)
        # Both terms can over/underflow individually. Compute in log space:
        #   log SF = log(Phi(-z1)) + log1p(-exp(2*lam/mu + log Phi(-z2) - log Phi(-z1)))
        # Asymptotically the two terms inside log1p cancel, so log1p(-1) = -inf and
        # log SF = -inf in the deep tails. Clip the diff strictly < 0 for safety.
        sqrt_lam_t = pt.sqrt(lam / td)
        z1 = sqrt_lam_t * (td / mu - 1.0)
        z2 = sqrt_lam_t * (td / mu + 1.0)
        log_sf_z1 = pt.log(pt.maximum(_ndtr(-z1), 1e-300))
        log_sf_z2 = pt.log(pt.maximum(_ndtr(-z2), 1e-300))
        diff = 2.0 * lam / mu + log_sf_z2 - log_sf_z1
        diff_clip = pt.minimum(diff, -1e-12)
        return log_sf_z1 + pt.log1p(-pt.exp(diff_clip))

    logp_winner1 = logpdf_ig(decision_time, mu1, lam1) + logsf_ig(decision_time, mu2, lam2)
    logp_winner2 = logpdf_ig(decision_time, mu2, lam2) + logsf_ig(decision_time, mu1, lam1)

    return pt.where(response > 0, logp_winner2, logp_winner1)


def _sample_wald_race_2(v1, v2, sigma1, sigma2, a, t0, n_samples=1, rng=None):
    """Draw (rt, choice) pairs from a 2-accumulator Wald race.

    Uses ``numpy.random.Generator.wald`` to sample independent inverse-Gaussian
    first-passage times for each accumulator, then takes the minimum (race).
    Returns rts with shape (n_trials, n_samples) and choices in {0, 1} where 1
    means accumulator 2 won.
    """
    if rng is None:
        rng = np.random.default_rng()
    v1 = np.asarray(v1, dtype=float).ravel()
    v2 = np.asarray(v2, dtype=float).ravel()
    sigma1 = np.asarray(sigma1, dtype=float).ravel()
    sigma2 = np.asarray(sigma2, dtype=float).ravel()
    a = np.asarray(a, dtype=float).ravel()
    t0 = np.asarray(t0, dtype=float).ravel()

    mu1 = a / v1
    mu2 = a / v2
    lam1 = a ** 2 / sigma1 ** 2
    lam2 = a ** 2 / sigma2 ** 2

    n_trials = len(v1)
    t1 = rng.wald(mu1[:, None], lam1[:, None], size=(n_trials, n_samples))
    t2 = rng.wald(mu2[:, None], lam2[:, None], size=(n_trials, n_samples))

    rts = np.minimum(t1, t2) + t0[:, None]
    choices = (t2 < t1).astype(int)
    return rts, choices


class RaceMixin:
    """Generalized Bayesian race-diffusion model.

    Subclasses provide the cognitive front-end (posterior mean, prior); this
    mixin adds the race likelihood and ``w_0, a, t0`` (and optionally
    ``w_d, w_s`` when ``advantage=True``).
    """
    advantage = True

    def get_free_parameters(self):
        pars = super().get_free_parameters()
        pars['w_0'] = {'mu_intercept': inverse_softplus_np(2.5),
                       'sigma_intercept': 0.5, 'transform': 'softplus'}
        if self.advantage:
            pars['w_d'] = {'mu_intercept': inverse_softplus_np(0.5),
                           'sigma_intercept': 0.5, 'transform': 'softplus'}
            pars['w_s'] = {'mu_intercept': 0.0,
                           'sigma_intercept': 0.5, 'transform': 'identity'}
        pars['a'] = {'mu_intercept': inverse_softplus_np(1.0),
                     'sigma_intercept': 0.3, 'transform': 'softplus'}
        pars['t0'] = {'mu_intercept': inverse_softplus_np(0.2),
                      'sigma_intercept': 0.5, 'transform': 'softplus'}
        return pars

    def _get_paradigm(self, paradigm=None, subject_mapping=None):
        try:
            p = super()._get_paradigm(paradigm, subject_mapping=subject_mapping)
        except TypeError:
            p = super()._get_paradigm(paradigm)
        if 'rt' in paradigm.columns:
            rt = np.asarray(paradigm['rt'].values, dtype=float)
            if np.any(rt <= 0) or np.any(np.isnan(rt)):
                n_bad = int((rt <= 0).sum() + np.isnan(rt).sum())
                raise ValueError(
                    f"Found {n_bad} trial(s) with rt <= 0 or NaN. "
                    "Filter non-responses and convert RT to seconds before fitting."
                )
            p['rt'] = rt
            if 'choice' in paradigm.columns:
                signed = np.where(paradigm['choice'].astype(bool).values, 1.0, -1.0)
                p['_rt_choice_data'] = np.column_stack([np.abs(rt), signed])
            # Per-trial hard upper bound on t0: 0.95 * min(rt) within subject.
            # Non-decision time cannot exceed the fastest observed RT — that's
            # a physical constraint, not a modeling preference. Eliminates the
            # a→0/t0→long collapse mode without needing a hard floor on `a`.
            subject_ix = np.asarray(p['subject_ix'], dtype=int)
            min_rt_per_subj = pd.Series(rt).groupby(subject_ix).min().sort_index().values
            p['_t0_cap'] = (min_rt_per_subj[subject_ix] * 0.95).astype(float)
        return p

    def build_likelihood(self, parameters, save_p_choice=False):
        model = pm.Model.get_context()
        if '_rt_choice_data' not in model.named_vars:
            raise ValueError("Race models require 'rt' and 'choice' columns in the paradigm.")
        model_inputs = self.get_model_inputs(parameters)

        v1, v2, sigma1, sigma2 = self._get_drifts(model_inputs, parameters)
        if save_p_choice:
            pm.Deterministic('drift_1', v1)
            pm.Deterministic('drift_2', v2)
            pm.Deterministic('sigma_1', sigma1)
            pm.Deterministic('sigma_2', sigma2)

        a = parameters['a']
        t0 = pt.minimum(parameters['t0'], model['_t0_cap'])
        if save_p_choice:
            pm.Deterministic('t0_eff', t0)
        observed = model['_rt_choice_data'].get_value()
        pm.CustomDist('ll', v1, v2, sigma1, sigma2, a, t0,
                      logp=lambda value, v1_, v2_, s1_, s2_, a_, t_:
                          logp_race_diffusion_2(value, v1_, v2_, s1_, s2_, a_, t_),
                      observed=observed)

    def build_prediction_model(self, paradigm, parameters):
        if isinstance(parameters, pd.DataFrame):
            parameter_subjects = (parameters['subject'] if 'subject' in parameters
                                  else parameters.index.get_level_values('subject'))
            if 'subject' in paradigm.index.names:
                assert np.array_equal(paradigm.index.unique(level='subject'),
                                      parameter_subjects)
            elif 'subject' in paradigm.columns:
                assert np.array_equal(paradigm.subject.unique(), parameter_subjects)
            parameters = parameters.to_dict(orient='list')

        with pm.Model() as self.prediction_model:
            paradigm_ = self._get_paradigm(paradigm=paradigm)
            self.set_paradigm(paradigm_)
            for key, value in parameters.items():
                pm.Data(key, value)
            params = self.get_parameter_values()
            model_inputs = self.get_model_inputs(params)
            v1, v2, sigma1, sigma2 = self._get_drifts(model_inputs, params)
            pm.Deterministic('drift_1', v1)
            pm.Deterministic('drift_2', v2)
            pm.Deterministic('sigma_1', sigma1)
            pm.Deterministic('sigma_2', sigma2)
            pm.Deterministic('a_t', params['a'])
            t0_eff = pt.minimum(params['t0'], pm.Model.get_context()['_t0_cap'])
            pm.Deterministic('t0_t', t0_eff)

    def simulate(self, paradigm, parameters, n_samples=1, random_seed=None):
        self.build_prediction_model(paradigm, parameters)
        v1 = self.prediction_model['drift_1'].eval()
        v2 = self.prediction_model['drift_2'].eval()
        s1 = self.prediction_model['sigma_1'].eval()
        s2 = self.prediction_model['sigma_2'].eval()
        a = self.prediction_model['a_t'].eval()
        t0 = self.prediction_model['t0_t'].eval()
        rng = np.random.default_rng(random_seed)
        rts, choices = _sample_wald_race_2(v1, v2, s1, s2, a, t0,
                                            n_samples=n_samples, rng=rng)

        if not paradigm.index.name:
            paradigm.index.name = 'trial'
        rt_df = pd.DataFrame(
            rts, index=paradigm.index,
            columns=pd.Index(np.arange(n_samples) + 1, name='sample'),
        ).stack().to_frame('simulated_rt')
        choice_df = pd.DataFrame(
            choices, index=paradigm.index,
            columns=pd.Index(np.arange(n_samples) + 1, name='sample'),
        ).stack().to_frame('simulated_choice')
        choice_df['simulated_choice'] = choice_df['simulated_choice'] > 0
        out_df = rt_df.join(choice_df).join(paradigm)
        if 'subject' in paradigm.columns:
            out_df = out_df.set_index('subject', append=True)
            out_df = out_df.reorder_levels(['subject'] + list(out_df.index.names)[:-1])
        return out_df

    def ppc(self, paradigm, idata, n_posterior_samples=200, inner_samples=1,
            random_seed=None, progressbar=True):
        rng = np.random.default_rng(random_seed)
        post = idata.posterior
        n_chain, n_draw = post.sizes['chain'], post.sizes['draw']
        flat_idx = rng.choice(n_chain * n_draw, n_posterior_samples, replace=False)
        chain_idx = flat_idx // n_draw
        draw_idx = flat_idx % n_draw
        param_names = list(self.free_parameters.keys())
        hierarchical = 'subject' in post[param_names[0]].dims

        results = []
        iterator = range(n_posterior_samples)
        if progressbar:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator, desc='race PPC')
            except ImportError:
                pass

        for k in iterator:
            ci, di = int(chain_idx[k]), int(draw_idx[k])
            if hierarchical:
                subjects = post.coords['subject'].values
                par_dict = {name: post[name].isel(chain=ci, draw=di).values
                            for name in param_names}
                pars_df = pd.DataFrame(par_dict, index=pd.Index(subjects, name='subject'))
                sim = self.simulate(paradigm, pars_df, n_samples=inner_samples,
                                    random_seed=int(rng.integers(0, 2**31 - 1)))
            else:
                par_dict = {name: float(post[name].isel(chain=ci, draw=di).values)
                            for name in param_names}
                sim = self.simulate(paradigm, par_dict, n_samples=inner_samples,
                                    random_seed=int(rng.integers(0, 2**31 - 1)))
            trial_levels = [n for n in sim.index.names if n != 'sample']
            sim = sim.groupby(level=trial_levels, observed=True)[
                ['simulated_rt', 'simulated_choice']
            ].mean()
            sim['ppc_sample'] = k
            results.append(sim[['simulated_rt', 'simulated_choice', 'ppc_sample']])

        out = pd.concat(results)
        out = out.set_index('ppc_sample', append=True)
        return out

    def _get_drifts(self, model_inputs, parameters):  # noqa: ARG002
        raise NotImplementedError(
            "Race subclasses must implement _get_drifts(model_inputs, parameters) "
            "returning (v1, v2, sigma1, sigma2)."
        )

    def build_loglik_model(self, paradigm, parameters):
        """Build a PyMC model that exposes per-trial log-likelihood as a
        Deterministic. Used by ``compute_log_likelihood`` for post-hoc model
        comparison via PSIS-LOO / WAIC.
        """
        if isinstance(parameters, pd.DataFrame):
            parameters = parameters.to_dict(orient='list')
        with pm.Model() as self.loglik_model:
            paradigm_ = self._get_paradigm(paradigm=paradigm)
            self.set_paradigm(paradigm_)
            for key, value in parameters.items():
                pm.Data(key, value)
            params = self.get_parameter_values()
            model_inputs = self.get_model_inputs(params)
            v1, v2, sigma1, sigma2 = self._get_drifts(model_inputs, params)
            mc = pm.Model.get_context()
            a = params['a']
            t0 = pt.minimum(params['t0'], mc['_t0_cap'])
            signed = pt.switch(mc['choice'], 1.0, -1.0)
            data = pt.stack([mc['rt'], signed], axis=1)
            per_trial = logp_race_diffusion_2(data, v1, v2, sigma1, sigma2, a, t0)
            pm.Deterministic('per_trial_ll', per_trial)

    def compute_log_likelihood(self, idata, paradigm=None, var_name='ll_race'):
        from .ddm import _attach_log_likelihood
        return _attach_log_likelihood(self, idata, paradigm=paradigm, var_name=var_name)


class RaceDiffusionMagnitudeComparisonModel(RaceMixin, MagnitudeComparisonModel):
    """Generalized Bayesian race-diffusion model for magnitude comparison.

    Paradigm columns required: ``n1``, ``n2``, ``choice`` (bool), ``rt`` (seconds).
    """

    def __init__(self, paradigm=None, fit_prior=True,
                 fit_seperate_evidence_sd=True, memory_model='independent',
                 save_trialwise_n_estimates=False, advantage=True):
        self.advantage = advantage
        super().__init__(paradigm=paradigm, fit_prior=fit_prior,
                         fit_seperate_evidence_sd=fit_seperate_evidence_sd,
                         memory_model=memory_model,
                         save_trialwise_n_estimates=save_trialwise_n_estimates)

    def _get_drifts(self, model_inputs, parameters):
        return _drifts_from_post_and_prior(model_inputs, parameters,
                                            advantage=self.advantage)


def _drifts_from_post_and_prior(model_inputs, parameters, advantage=True):
    """Per-accumulator drifts under the generalized Bayesian race model.

    For risk front-ends (``p1, p2`` in model_inputs) the race operates on
    ``log(EU_k) = post_log_n_k + log(p_k)``, with the centering reference
    shifted so the prior baseline sits on log(EU) too.
    """
    post_n1_mu, _ = get_posterior(
        model_inputs['n1_prior_mu'], model_inputs['n1_prior_sd'],
        model_inputs['n1_evidence_mu'], model_inputs['n1_evidence_sd'])
    post_n2_mu, _ = get_posterior(
        model_inputs['n2_prior_mu'], model_inputs['n2_prior_sd'],
        model_inputs['n2_evidence_mu'], model_inputs['n2_evidence_sd'])
    mu_p1 = model_inputs['n1_prior_mu']
    mu_p2 = model_inputs['n2_prior_mu']
    if 'p1' in model_inputs and 'p2' in model_inputs:
        log_p1, log_p2 = pt.log(model_inputs['p1']), pt.log(model_inputs['p2'])
        post_n1_mu = post_n1_mu + log_p1
        post_n2_mu = post_n2_mu + log_p2
        mu_p1 = mu_p1 + log_p1
        mu_p2 = mu_p2 + log_p2

    tilde_1 = post_n1_mu - mu_p1
    tilde_2 = post_n2_mu - mu_p2
    w0 = parameters['w_0']
    if advantage:
        wd, ws = parameters['w_d'], parameters['w_s']
        diff = tilde_1 - tilde_2
        summ = tilde_1 + tilde_2
        v1 = w0 + wd * diff + ws * summ
        v2 = w0 - wd * diff + ws * summ
    else:
        v1 = w0 + tilde_1
        v2 = w0 + tilde_2
    # Positivity guard (Wald requires v > 0; informative w_0 prior keeps this slack).
    v1 = pt.maximum(v1, 1e-3)
    v2 = pt.maximum(v2, 1e-3)
    return v1, v2, pt.ones_like(v1), pt.ones_like(v2)


class RaceDiffusionFlexibleNoiseComparisonModel(RaceMixin, FlexibleNoiseComparisonModel):
    """Race-diffusion variant with B-spline stimulus-dependent encoding noise.

    Paradigm columns required: ``n1``, ``n2``, ``choice`` (bool), ``rt`` (seconds).
    """

    def __init__(self, paradigm, fit_seperate_evidence_sd=True,
                 fit_prior=True, spline_order=5,
                 memory_model='independent', advantage=True):
        self.advantage = advantage
        FlexibleNoiseComparisonModel.__init__(
            self, paradigm,
            fit_seperate_evidence_sd=fit_seperate_evidence_sd,
            fit_prior=fit_prior,
            spline_order=spline_order,
            memory_model=memory_model,
        )

    def _get_drifts(self, model_inputs, parameters):
        return _drifts_from_post_and_prior(model_inputs, parameters,
                                            advantage=self.advantage)


class RaceDiffusionRiskModel(RaceMixin, RiskModel):
    """Race-diffusion variant for risky choice (race on log(EU_k)).

    Paradigm columns required: ``n1``, ``n2``, ``p1``, ``p2``, ``choice`` (bool),
    ``rt`` (seconds).
    """

    def __init__(self, paradigm=None, prior_estimate='objective',
                 fit_seperate_evidence_sd=True,
                 save_trialwise_n_estimates=False, memory_model='independent',
                 advantage=True):
        self.advantage = advantage
        super().__init__(
            paradigm=paradigm, prior_estimate=prior_estimate,
            fit_seperate_evidence_sd=fit_seperate_evidence_sd,
            save_trialwise_n_estimates=save_trialwise_n_estimates,
            memory_model=memory_model,
        )

    def _get_drifts(self, model_inputs, parameters):
        return _drifts_from_post_and_prior(model_inputs, parameters,
                                            advantage=self.advantage)


class RaceDiffusionFlexibleNoiseRiskModel(RaceMixin, FlexibleNoiseRiskModel):
    """Race-diffusion variant of :class:`FlexibleNoiseRiskModel` for risky
    choice with stimulus-dependent (B-spline) encoding noise.

    Race operates on log(EU_k) (carried in via FlexibleNoiseRiskModel's
    cognitive front-end), and σ_k(n) is a per-trial spline rather than a
    scalar. With ``advantage=True`` (default) drifts decompose into
    difference + summary terms; with ``False`` they're w_0 + tilde_μ_k*.

    Paradigm columns required: ``n1``, ``n2``, ``p1``, ``p2``, ``choice`` (bool),
    ``rt`` (seconds).
    """

    def __init__(self, paradigm, prior_estimate='full',
                 fit_seperate_evidence_sd=True,
                 save_trialwise_n_estimates=False, spline_order=5,
                 representational_noise='payoff',
                 memory_model='independent',
                 advantage=True):
        self.advantage = advantage
        FlexibleNoiseRiskModel.__init__(
            self, paradigm,
            prior_estimate=prior_estimate,
            fit_seperate_evidence_sd=fit_seperate_evidence_sd,
            save_trialwise_n_estimates=save_trialwise_n_estimates,
            spline_order=spline_order,
            representational_noise=representational_noise,
            memory_model=memory_model,
        )

    def _get_drifts(self, model_inputs, parameters):
        return _drifts_from_post_and_prior(model_inputs, parameters,
                                            advantage=self.advantage)


class RaceDiffusionFlexibleNoiseRiskRegressionModel(
        RaceMixin, FlexibleNoiseRiskRegressionModel):
    """Race-diffusion variant of :class:`FlexibleNoiseRiskRegressionModel`.

    Patsy-formula regression on noise spline coefficients (auto-expanded if
    you target ``n1_evidence_sd`` etc.) plus on accumulator parameters
    (``a``, ``t0``, ``w_0``, ``w_d``, ``w_s``). Use for TMS analyses where
    the noise function should differ across stimulation conditions.

    Example::

        regressors = {
            'n1_evidence_sd': 'stimulation_condition',
            'n2_evidence_sd': 'stimulation_condition',
            'a': 'stimulation_condition',
        }
    """

    def __init__(self, paradigm, regressors, prior_estimate='full',
                 fit_seperate_evidence_sd=True,
                 save_trialwise_n_estimates=False, spline_order=5,
                 representational_noise='payoff',
                 memory_model='independent',
                 advantage=True):
        self.advantage = advantage
        FlexibleNoiseRiskRegressionModel.__init__(
            self, paradigm, regressors,
            prior_estimate=prior_estimate,
            fit_seperate_evidence_sd=fit_seperate_evidence_sd,
            save_trialwise_n_estimates=save_trialwise_n_estimates,
            spline_order=spline_order,
            representational_noise=representational_noise,
            memory_model=memory_model,
        )

    def _get_drifts(self, model_inputs, parameters):
        return _drifts_from_post_and_prior(model_inputs, parameters,
                                            advantage=self.advantage)


# ============================================================
# Race-Diffusion × PowerLawNoise variants
# ============================================================

class RaceDiffusionPowerLawNoiseComparisonModel(RaceMixin, PowerLawNoiseComparisonModel):
    """Race-diffusion variant of :class:`PowerLawNoiseComparisonModel`.

    Race accumulators with σ_k(n) = exp(log_sd_k) · n^noise_exponent. The
    noise_exponent is shared across n1/n2; together with separate log-SD
    intercepts this lets you recover the Stevens compression exponent
    (alpha = 1 - noise_exponent) jointly with choice + RT.
    """

    def __init__(self, paradigm, fit_seperate_evidence_sd=True,
                 fit_prior=False, memory_model='independent',
                 advantage=True, flat_observer_prior=False):
        self.advantage = advantage
        PowerLawNoiseComparisonModel.__init__(
            self, paradigm,
            fit_seperate_evidence_sd=fit_seperate_evidence_sd,
            fit_prior=fit_prior, memory_model=memory_model,
            flat_observer_prior=flat_observer_prior,
        )

    def _get_drifts(self, model_inputs, parameters):
        return _drifts_from_post_and_prior(model_inputs, parameters,
                                            advantage=self.advantage)


class RaceDiffusionPowerLawNoiseComparisonRegressionModel(
        RaceMixin, PowerLawNoiseComparisonRegressionModel):
    """Race-diffusion + power-law noise + patsy-formula regression.

    Same multi-inheritance pattern as the flex versions. Use to let
    ``noise_exponent`` (Stevens compression) vary with experimental
    condition.
    """

    def __init__(self, paradigm, regressors,
                 fit_seperate_evidence_sd=True,
                 fit_prior=False, memory_model='independent',
                 advantage=True):
        self.advantage = advantage
        PowerLawNoiseComparisonRegressionModel.__init__(
            self, paradigm, regressors,
            fit_seperate_evidence_sd=fit_seperate_evidence_sd,
            fit_prior=fit_prior, memory_model=memory_model,
        )

    def _get_drifts(self, model_inputs, parameters):
        return _drifts_from_post_and_prior(model_inputs, parameters,
                                            advantage=self.advantage)


class RaceDiffusionPowerLawNoiseRiskModel(RaceMixin, PowerLawNoiseRiskModel):
    """Race-diffusion + power-law-noise risky choice."""

    def __init__(self, paradigm, prior_estimate='full',
                 fit_seperate_evidence_sd=True,
                 save_trialwise_n_estimates=False,
                 memory_model='independent',
                 advantage=True):
        self.advantage = advantage
        PowerLawNoiseRiskModel.__init__(
            self, paradigm,
            prior_estimate=prior_estimate,
            fit_seperate_evidence_sd=fit_seperate_evidence_sd,
            save_trialwise_n_estimates=save_trialwise_n_estimates,
            memory_model=memory_model,
        )

    def _get_drifts(self, model_inputs, parameters):
        return _drifts_from_post_and_prior(model_inputs, parameters,
                                            advantage=self.advantage)


class RaceDiffusionPowerLawNoiseRiskRegressionModel(
        RaceMixin, PowerLawNoiseRiskRegressionModel):
    """Race-diffusion + power-law-noise risky choice + regression."""

    def __init__(self, paradigm, regressors, prior_estimate='full',
                 fit_seperate_evidence_sd=True,
                 save_trialwise_n_estimates=False,
                 memory_model='independent',
                 advantage=True):
        self.advantage = advantage
        PowerLawNoiseRiskRegressionModel.__init__(
            self, paradigm, regressors,
            prior_estimate=prior_estimate,
            fit_seperate_evidence_sd=fit_seperate_evidence_sd,
            save_trialwise_n_estimates=save_trialwise_n_estimates,
            memory_model=memory_model,
        )

    def _get_drifts(self, model_inputs, parameters):
        return _drifts_from_post_and_prior(model_inputs, parameters,
                                            advantage=self.advantage)
