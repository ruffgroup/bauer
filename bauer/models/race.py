"""Racing diffusion model (RDM) variants of bauer's choice models.

Each stimulus drives its own Wiener accumulator racing to a common threshold.
The first-passage time of accumulator k with drift v_k and noise sigma_k to
barrier a is inverse Gaussian with mean a/v_k and shape a^2/sigma_k^2 — fully
analytical, no LANs needed. Likelihood combines the winner's IG density with
the survival functions of the losing accumulators. Standard reference is the
"racing diffusion model" of Tillman, Van Zandt & Logan (2020).

Cognitive interpretation: each accumulator k integrates noisy evidence about
its corresponding stimulus n_k. Drift = log(n_k) (stimulus-driven), diffusion
noise = the *encoding* SD per stimulus (``n_k_evidence_sd``, i.e. nu_k in the
Khaw, Li & Woodford 2020 framework). Encoding noise can differ per accumulator
(e.g. n1 carries memory noise added on top of perceptual noise), so the two
accumulators race with different first-passage distributions. The Khaw-Woodford
prior is *not* applied to drift here — applying it would make drifts collapse
toward the prior mean for tight priors and discard discriminative information
that's actually present in the encoding samples.

Predicts the size effect: larger n_k → larger drift on both accumulators →
faster RTs regardless of which one wins.

HSSM convention for ``data``: column 0 = |rt|, column 1 = signed response
(+1 = accumulator 2 wins / choice = True, -1 = accumulator 1 wins).
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .magnitude import MagnitudeComparisonModel, FlexibleNoiseComparisonModel
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
    """Swaps the static cumulative-normal likelihood for an analytical Wald-race
    likelihood with per-accumulator drift AND noise.

    Subclasses implement ``_get_drifts(model_inputs, parameters)`` returning
    ``(v1, v2, sigma1, sigma2)`` — both must be positive per-trial pytensor
    vectors.

    Free parameters added: ``a`` (threshold), ``t0`` (non-decision time),
    optionally ``v_scale`` (drift coefficient). No ``z`` — single-boundary
    accumulators have no analogue of biased starting point in the DDM sense
    (asymmetric *thresholds* per accumulator would be the analogue and could
    be added later).
    """
    fit_v_scale = False

    def get_free_parameters(self):
        pars = super().get_free_parameters()
        if self.fit_v_scale:
            pars['v_scale'] = {'mu_intercept': 1.0, 'sigma_intercept': 1.0,
                               'transform': 'identity'}
        pars['a'] = {'mu_intercept': inverse_softplus_np(1.0),
                     'sigma_intercept': 0.5, 'transform': 'softplus'}
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
        return p

    def build_likelihood(self, parameters, save_p_choice=False):
        model = pm.Model.get_context()
        if 'rt' not in [v.name for v in model.value_vars + list(model.named_vars.values())]:
            raise ValueError("Race models require an 'rt' column in the paradigm for fitting.")
        model_inputs = self.get_model_inputs(parameters)

        v1, v2, sigma1, sigma2 = self._get_drifts(model_inputs, parameters)
        if save_p_choice:
            pm.Deterministic('drift_1', v1)
            pm.Deterministic('drift_2', v2)
            pm.Deterministic('sigma_1', sigma1)
            pm.Deterministic('sigma_2', sigma2)

        a = parameters['a']
        t0 = parameters['t0']
        signed = pt.switch(model['choice'], 1.0, -1.0)
        data = pt.stack([model['rt'], signed], axis=1)

        pm.Potential('ll_race', logp_race_diffusion_2(data, v1, v2, sigma1, sigma2, a, t0))

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
            pm.Deterministic('t0_t', params['t0'])

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
            a, t0 = params['a'], params['t0']
            mc = pm.Model.get_context()
            signed = pt.switch(mc['choice'], 1.0, -1.0)
            data = pt.stack([mc['rt'], signed], axis=1)
            per_trial = logp_race_diffusion_2(data, v1, v2, sigma1, sigma2, a, t0)
            pm.Deterministic('per_trial_ll', per_trial)

    def compute_log_likelihood(self, idata, paradigm=None, var_name='ll_race'):
        from .ddm import _attach_log_likelihood
        return _attach_log_likelihood(self, idata, paradigm=paradigm, var_name=var_name)


class RaceDiffusionMagnitudeComparisonModel(RaceMixin, MagnitudeComparisonModel):
    """Racing diffusion model (RDM) for magnitude comparison.

    Each stimulus n_k drives its own Wiener accumulator with:

    - **drift** $v_k = \\mu_{post,k}$ — the Bayesian posterior mean of $\\log n_k$.
      With asymmetric encoding noise ($\\nu_1 \\ne \\nu_2$), the prior pulls the
      two accumulator drifts by different amounts ($\\beta_k = \\sigma_p^2 /
      (\\sigma_p^2 + \\nu_k^2)$), reproducing the order-effect mechanism that
      the static cumnorm model uses.
    - **diffusion noise** $\\sigma_k$ = ``n_k_evidence_sd`` — the *encoding* SD
      per accumulator (what Khaw, Li & Woodford 2020 call $\\nu_k$). Distinct
      per accumulator so a noisier $r_k$ also gives more variable race times.
      Note this is *not* the posterior width $\\sigma_{post}^2 = \\beta \\nu^2$.

    Per-accumulator first-passage to threshold $a$ is inverse Gaussian with
    mean $a/v_k$ and shape $a^2/\\sigma_k^2$.

    Predicts the size effect: larger $n_k$ → larger drift on both accumulators
    → faster RTs regardless of which one wins. And the order effect:
    $\\nu_1 > \\nu_2$ slows accumulator 1 in two ways — directly via larger
    $\\sigma_1$, and indirectly via more prior pulling on $v_1$.

    Tight-prior limit: $\\beta_k \\to 0$, drifts collapse to $\\mu_{prior}$,
    race becomes 50/50 — consistent with the static cumnorm model in the same
    limit (the Bayesian observer ignores noisy samples when the prior is sharp).

    Paradigm columns required: ``n1``, ``n2``, ``choice`` (bool), ``rt`` (seconds).

    Parameters
    ----------
    fit_v_scale : bool
        If True, fit a multiplicative ``v_scale`` on drifts. Default False —
        the ``a`` parameter absorbs the scale and removes the degeneracy.
    """

    def __init__(self, paradigm=None, fit_prior=False,
                 fit_seperate_evidence_sd=True, memory_model='independent',
                 save_trialwise_n_estimates=False, fit_v_scale=False):
        self.fit_v_scale = fit_v_scale
        super().__init__(paradigm=paradigm, fit_prior=fit_prior,
                         fit_seperate_evidence_sd=fit_seperate_evidence_sd,
                         memory_model=memory_model,
                         save_trialwise_n_estimates=save_trialwise_n_estimates)

    def _get_drifts(self, model_inputs, parameters):
        return _drifts_from_post_and_prior(model_inputs, self.fit_v_scale, parameters)


def _drifts_from_post_and_prior(model_inputs, fit_v_scale, parameters):
    """Shared drift/noise computation for race-diffusion magnitude variants.

    Drift = Bayesian posterior mean. Diffusion noise = SD of posterior mean
    conditional on a fixed objective numerosity = beta_k * nu_k. Used by both
    the scalar and flexible-noise race-diffusion magnitude models.
    """
    nu1 = model_inputs['n1_evidence_sd']
    nu2 = model_inputs['n2_evidence_sd']
    sp1 = model_inputs['n1_prior_sd']
    sp2 = model_inputs['n2_prior_sd']
    post_n1_mu, _ = get_posterior(model_inputs['n1_prior_mu'], sp1,
                                   model_inputs['n1_evidence_mu'], nu1)
    post_n2_mu, _ = get_posterior(model_inputs['n2_prior_mu'], sp2,
                                   model_inputs['n2_evidence_mu'], nu2)
    beta1 = sp1 ** 2 / (sp1 ** 2 + nu1 ** 2)
    beta2 = sp2 ** 2 / (sp2 ** 2 + nu2 ** 2)
    v1, v2 = post_n1_mu, post_n2_mu
    sigma1, sigma2 = beta1 * nu1, beta2 * nu2
    if fit_v_scale:
        v1 = parameters['v_scale'] * v1
        v2 = parameters['v_scale'] * v2
    return v1, v2, sigma1, sigma2


class RaceDiffusionFlexibleNoiseComparisonModel(RaceMixin, FlexibleNoiseComparisonModel):
    """Racing diffusion model with stimulus-dependent (spline) encoding noise.

    Same structure as :class:`RaceDiffusionMagnitudeComparisonModel`, but
    ``n1_evidence_sd`` and ``n2_evidence_sd`` are smooth functions of stimulus
    magnitude (B-spline of order ``polynomial_order``) rather than scalar
    parameters per subject. The race accumulators inherit asymmetric, magnitude-
    dependent noise from the spline.

    Drift = posterior mean per accumulator (Bayesian-pulled by the stimulus-
    dependent encoding noise). Diffusion noise per accumulator is again
    $\\sigma_k = \\beta_k \\nu_k(n_k)$ — the SD of the posterior mean conditional
    on a fixed $n_k$.

    Paradigm columns required: ``n1``, ``n2``, ``choice`` (bool), ``rt`` (seconds).
    """

    def __init__(self, paradigm, fit_seperate_evidence_sd=True,
                 fit_prior=False, polynomial_order=5,
                 memory_model='independent', fit_v_scale=False):
        self.fit_v_scale = fit_v_scale
        FlexibleNoiseComparisonModel.__init__(
            self, paradigm,
            fit_seperate_evidence_sd=fit_seperate_evidence_sd,
            fit_prior=fit_prior,
            polynomial_order=polynomial_order,
            memory_model=memory_model,
        )

    def _get_drifts(self, model_inputs, parameters):
        return _drifts_from_post_and_prior(model_inputs, self.fit_v_scale, parameters)
