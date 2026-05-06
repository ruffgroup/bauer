"""Drift-diffusion-model variants of bauer's choice models.

These models replace the static cumulative-normal choice rule with a Wiener
first-passage-time likelihood (HSSM's analytic Navarro-Fuss implementation),
so they fit reaction times jointly with choices.

The cognitive front-end (Bayesian observer with priors, asymmetric noise,
memory model, etc.) is reused unchanged from the static models. The same
``(diff_mu, diff_sd)`` that the static model feeds to the cumulative normal
becomes the DDM drift signal: ``v = v_scale * diff_mu / diff_sd``. This is
the subjective signal-to-noise ratio of the perceived evidence — order
effects from asymmetric n1/n2 noise propagate into drift via differential
prior pulling, exactly as in the static model.

HSSM convention (used here):
    a   half boundary separation (>0)
    z   normalized starting point in [0, 1]; 0.5 = unbiased
    t0  non-decision time, in seconds
    rt  reaction time, in seconds — must be > 0
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .magnitude import MagnitudeComparisonModel, FlexibleNoiseComparisonModel
from .risky_choice import (
    RiskModel, FlexibleNoiseRiskModel, FlexibleNoiseRiskRegressionModel,
)
from ..utils.bayes import get_posterior
from ..utils.math import inverse_softplus_np

try:
    from hssm.likelihoods import logp_ddm
except ImportError:
    logp_ddm = None

try:
    from ssms.basic_simulators import simulator as _ssms_simulator
except ImportError:
    _ssms_simulator = None


def _attach_log_likelihood(model_obj, idata, paradigm=None, var_name='ll'):
    """Compute per-trial log-likelihood for a fitted DDMMixin/RaceMixin model
    and attach to ``idata.log_likelihood``. Reuses ``build_loglik_model`` to
    construct a graph with parameters as ``pm.Data`` and the per-trial logp as
    a ``Deterministic``, then loops over posterior samples.
    """
    import pytensor
    import xarray as xr
    if paradigm is None:
        paradigm = model_obj.paradigm

    n_chain = idata.posterior.sizes['chain']
    n_draw = idata.posterior.sizes['draw']
    param_names = list(model_obj.free_parameters.keys())

    # Determine shape of each parameter (per-subject vs scalar)
    hierarchical = 'subject' in idata.posterior[param_names[0]].dims
    if hierarchical:
        n_subjects = idata.posterior.sizes['subject']
        placeholder = {name: np.zeros(n_subjects) for name in param_names}
    else:
        placeholder = {name: 0.0 for name in param_names}

    model_obj.build_loglik_model(paradigm, placeholder)
    pmodel = model_obj.loglik_model
    per_trial_var = pmodel['per_trial_ll']
    fn = pytensor.function([], per_trial_var, on_unused_input='ignore')

    n_trials = len(paradigm)
    out = np.zeros((n_chain, n_draw, n_trials), dtype=np.float64)
    for c in range(n_chain):
        for d in range(n_draw):
            for name in param_names:
                vals = idata.posterior[name].isel(chain=c, draw=d).values
                if hierarchical:
                    pmodel[name].set_value(np.asarray(vals, dtype=float))
                else:
                    pmodel[name].set_value(float(vals))
            out[c, d, :] = fn()

    da = xr.DataArray(
        out, dims=['chain', 'draw', 'observation'],
        coords={'chain': idata.posterior.chain.values,
                'draw': idata.posterior.draw.values,
                'observation': np.arange(n_trials)},
    )
    ll_ds = xr.Dataset({var_name: da})
    if 'log_likelihood' in idata:
        # arviz won't replace cleanly; build new InferenceData group
        idata.log_likelihood = ll_ds
    else:
        idata.add_groups(log_likelihood=ll_ds)
    return idata


class DDMMixin:
    """Swaps the static cumulative-normal likelihood for an HSSM Wiener WFPT.

    Subclasses implement ``_get_drift(model_inputs, parameters)`` returning a
    per-trial drift signal computed from the cognitive model.

    Adds three free DDM parameters: ``a`` (half boundary separation), ``z``
    (normalized start point), ``t0`` (non-decision time). Optionally adds
    ``v_scale``, the drift coefficient, when ``fit_v_scale=True`` (default).

    Note on ``v_scale`` identifiability: the drift formula is
    ``v = v_scale * (post_n2_mu - post_n1_mu) / sqrt(sd1^2 + sd2^2)``. With a
    flat prior, scaling all ``evidence_sd`` by ``c`` is exactly equivalent to
    scaling ``v_scale`` by ``1/c`` — perfect degeneracy. With an informative
    prior, scaling ``evidence_sd`` also changes prior-pulling weights, which
    breaks the degeneracy non-linearly. RT shape adds further information
    (drift sets accumulation speed). Empirically the two parameters fit cleanly
    in our tests, but they remain strongly correlated. Set ``fit_v_scale=False``
    to fix ``v_scale=1`` and let ``evidence_sd`` absorb the drift scale.
    """

    fit_v_scale = False
    fix_z = True

    def get_free_parameters(self):
        pars = super().get_free_parameters()
        if self.fit_v_scale:
            pars['v_scale'] = {'mu_intercept': 1.0, 'sigma_intercept': 1.0,
                               'transform': 'identity'}
        # Hard lower bound on `a` to block the a→0 collapse mode. With
        # min_value=0.3 and mu_intercept=0, softplus(0)=0.69 → a ≈ 0.99.
        pars['a'] = {'mu_intercept': 0.0, 'sigma_intercept': 0.5,
                     'transform': 'softplus', 'min_value': 0.3}
        if not self.fix_z:
            pars['z'] = {'mu_intercept': 0.0, 'sigma_intercept': 0.5,
                         'transform': 'logistic'}
        pars['t0'] = {'mu_intercept': inverse_softplus_np(0.2),
                      'sigma_intercept': 0.5, 'transform': 'softplus'}
        return pars

    def _get_paradigm(self, paradigm=None, subject_mapping=None):
        # MagnitudeComparisonModel._get_paradigm doesn't accept subject_mapping;
        # try with the kwarg, fall back to positional for older overrides.
        try:
            p = super()._get_paradigm(paradigm, subject_mapping=subject_mapping)
        except TypeError:
            p = super()._get_paradigm(paradigm)
        # rt is required for fitting but optional for prediction/simulation.
        if 'rt' in paradigm.columns:
            rt = np.asarray(paradigm['rt'].values, dtype=float)
            if np.any(rt <= 0) or np.any(np.isnan(rt)):
                n_bad = int((rt <= 0).sum() + np.isnan(rt).sum())
                raise ValueError(
                    f"Found {n_bad} trial(s) with rt <= 0 or NaN. "
                    "Filter non-responses and convert RT to seconds before fitting."
                )
            p['rt'] = rt
            # Pre-compute the (n, 2) HSSM-format data array for the WFPT/race
            # likelihood: column 0 = |rt|, column 1 = signed response (+1 = upper).
            if 'choice' in paradigm.columns:
                signed = np.where(paradigm['choice'].astype(bool).values, 1.0, -1.0)
                p['_rt_choice_data'] = np.column_stack([np.abs(rt), signed])
        return p

    def build_likelihood(self, parameters, save_p_choice=False):
        if logp_ddm is None:
            raise ImportError(
                "DDM models require hssm. Install with: pip install bauer[ddm]"
            )
        model = pm.Model.get_context()
        if '_rt_choice_data' not in model.named_vars:
            raise ValueError(
                "DDM models require 'rt' and 'choice' columns in the paradigm."
            )
        model_inputs = self.get_model_inputs(parameters)

        v = self._get_drift(model_inputs, parameters)
        if save_p_choice:
            pm.Deterministic('drift', v)

        a = parameters['a']
        z = pt.constant(0.5) if self.fix_z else parameters['z']
        t0 = parameters['t0']

        # CustomDist with the (rt, signed) data array as observed: gives clean
        # PyMC observed-RV semantics, so pm.compute_log_likelihood works for
        # PSIS-LOO model comparison without any post-hoc helper.
        observed = model['_rt_choice_data'].get_value()
        pm.CustomDist('ll', v, a, z, t0,
                      logp=lambda value, v_, a_, z_, t_: logp_ddm(value, v_, a_, z_, t_),
                      observed=observed)

    def build_prediction_model(self, paradigm, parameters):
        """Build a PyMC model that exposes per-trial drift + (a, z, t0) as
        Deterministic nodes — no likelihood. Used by ``simulate`` and ``predict``.
        """
        if isinstance(parameters, pd.DataFrame):
            parameter_subjects = (parameters['subject'] if 'subject' in parameters
                                  else parameters.index.get_level_values('subject'))
            if 'subject' in paradigm.index.names:
                assert np.array_equal(paradigm.index.unique(level='subject'),
                                      parameter_subjects), \
                    "Subjects in paradigm don't match subjects in parameters."
            elif 'subject' in paradigm.columns:
                assert np.array_equal(paradigm.subject.unique(), parameter_subjects), \
                    "Subjects in paradigm don't match subjects in parameters."
            parameters = parameters.to_dict(orient='list')

        with pm.Model() as self.prediction_model:
            paradigm_ = self._get_paradigm(paradigm=paradigm)
            self.set_paradigm(paradigm_)
            for key, value in parameters.items():
                pm.Data(key, value)
            params = self.get_parameter_values()
            model_inputs = self.get_model_inputs(params)
            v = self._get_drift(model_inputs, params)
            pm.Deterministic('drift', v)
            # Also expose (a, z, t0) tiled to per-trial shape for simulate.
            pm.Deterministic('a_t', params['a'])
            z_per_trial = pt.full_like(v, 0.5) if self.fix_z else params['z']
            pm.Deterministic('z_t', z_per_trial)
            pm.Deterministic('t0_t', params['t0'])

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
            v = self._get_drift(model_inputs, params)
            a, t0 = params['a'], params['t0']
            z = pt.constant(0.5) if self.fix_z else params['z']
            mc = pm.Model.get_context()
            signed = pt.switch(mc['choice'], 1.0, -1.0)
            data = pt.stack([mc['rt'], signed], axis=1)
            per_trial = logp_ddm(data, v, a, z, t0)
            pm.Deterministic('per_trial_ll', per_trial)

    def compute_log_likelihood(self, idata, paradigm=None, var_name='ll_ddm'):
        """Compute per-trial log-likelihood post-hoc and attach to
        ``idata.log_likelihood`` so ``az.compare`` works for model selection.

        Returns the modified idata in-place. Stored variable is named ``var_name``
        (default ``ll_ddm``).
        """
        return _attach_log_likelihood(self, idata, paradigm=paradigm, var_name=var_name)

    def predict(self, paradigm, parameters):
        """Return per-trial drift, P(upper boundary), and mean RT — all analytical.

        For DDM with drift ``v``, half-boundary ``a``, normalized start ``z``,
        non-decision time ``t0`` (HSSM convention; full boundary = 2a):

            P(upper) = (1 - exp(-4*v*a*z)) / (1 - exp(-4*v*a))
            E[RT]    = t0 + (2a / v) * (P(upper) - z)        (v ≠ 0)
            E[RT]    = t0 + 4*a^2 * z * (1-z)                 (v = 0)

        Median RT has no clean closed form; use ``simulate`` and take the
        median across draws if you need it.
        """
        self.build_prediction_model(paradigm, parameters)
        v = self.prediction_model['drift'].eval()
        a = self.prediction_model['a_t'].eval()
        z = self.prediction_model['z_t'].eval()
        t0 = self.prediction_model['t0_t'].eval()

        with np.errstate(over='ignore', invalid='ignore'):
            num = 1.0 - np.exp(-4.0 * v * a * z)
            den = 1.0 - np.exp(-4.0 * v * a)
            p_upper = np.where(np.abs(v) < 1e-9, z, num / den)
            mean_rt = np.where(
                np.abs(v) < 1e-9,
                t0 + 4.0 * a * a * z * (1.0 - z),
                t0 + (2.0 * a / np.where(np.abs(v) < 1e-9, 1.0, v)) * (p_upper - z),
            )

        out = pd.DataFrame({'drift': v, 'p_upper': p_upper, 'mean_rt': mean_rt},
                           index=paradigm.index)
        return out.join(paradigm)

    def simulate(self, paradigm, parameters, n_samples=1, random_seed=None):
        """Simulate (rt, choice) draws via ssm-simulators.

        Parameters
        ----------
        paradigm : DataFrame
            Trial-level paradigm. Must have stimulus columns; ``rt`` is ignored.
        parameters : dict or DataFrame
            Parameter values (per-subject DataFrame for hierarchical, dict for
            single-subject).
        n_samples : int
            Number of independent simulated datasets per trial.
        random_seed : int or None
            Forwarded to ``ssms.basic_simulators.simulator``.

        Returns
        -------
        pd.DataFrame
            Indexed like ``paradigm`` (with a ``sample`` level), columns
            ``simulated_rt`` and ``simulated_choice`` (bool, ``True`` = upper).
        """
        if _ssms_simulator is None:
            raise ImportError(
                "DDM simulate requires ssm-simulators. "
                "Install with: pip install bauer[ddm]"
            )
        self.build_prediction_model(paradigm, parameters)
        with self.prediction_model:
            v = self.prediction_model['drift'].eval()
            a = self.prediction_model['a_t'].eval()
            z = self.prediction_model['z_t'].eval()
            t0 = self.prediction_model['t0_t'].eval()
        # Per-trial theta = [v, a, z, t]
        theta = np.column_stack([v, a, z, t0])
        out = _ssms_simulator.simulator(
            theta=theta, model='ddm', n_samples=n_samples,
            random_state=random_seed,
        )
        # ssms output shape: (n_trials, 1) when n_samples=1; otherwise
        # (n_samples, n_trials, 1). Squeeze and reshape to (n_trials, n_samples).
        rts = np.asarray(out['rts']).squeeze(-1)
        choices = np.asarray(out['choices']).squeeze(-1)
        if n_samples > 1:
            rts = rts.T          # (n_trials, n_samples)
            choices = choices.T

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
        # ssms convention: +1 = upper, -1 = lower; map to bool (True = upper)
        choice_df['simulated_choice'] = choice_df['simulated_choice'] > 0
        out_df = rt_df.join(choice_df).join(paradigm)

        if 'subject' in paradigm.columns:
            out_df = out_df.set_index('subject', append=True)
            out_df = out_df.reorder_levels(['subject'] + list(out_df.index.names)[:-1])

        return out_df

    def ppc(self, paradigm, idata, n_posterior_samples=200, inner_samples=1,
            random_seed=None, progressbar=True):
        """Posterior predictive simulation: draw (rt, choice) per trial for
        ``n_posterior_samples`` posterior draws.

        ``inner_samples`` controls how many independent (rt, choice) draws are
        simulated per real trial per posterior sample. ``1`` is the canonical
        choice (one fake dataset per posterior draw); higher values average out
        the per-trial Bernoulli/RT-noise within each posterior sample, giving
        a tighter PPC band that more cleanly reflects parameter uncertainty.

        Returns a long-format DataFrame indexed by trial × ppc_sample with
        columns ``simulated_rt`` and ``simulated_choice``.
        """
        if _ssms_simulator is None:
            raise ImportError(
                "DDM ppc requires ssm-simulators. Install with: pip install bauer[ddm]"
            )
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
                iterator = tqdm(iterator, desc='PPC')
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
            # Average over the 'sample' level so each posterior draw produces
            # one row per real trial.
            trial_levels = [n for n in sim.index.names if n != 'sample']
            sim = sim.groupby(level=trial_levels, observed=True)[
                ['simulated_rt', 'simulated_choice']
            ].mean()
            sim['ppc_sample'] = k
            results.append(sim[['simulated_rt', 'simulated_choice', 'ppc_sample']])

        out = pd.concat(results)
        out = out.set_index('ppc_sample', append=True)
        return out

    def _get_drift(self, model_inputs, parameters):  # noqa: ARG002
        raise NotImplementedError(
            "DDM subclasses must implement _get_drift(model_inputs, parameters)."
        )


def _drift_from_snr(model_inputs, v_scale=None):
    """Drift = ((post_n2_mu - post_n1_mu) + threshold) / sqrt(sd1^2 + sd2^2).

    Shared by every DDM model whose cognitive front-end produces the standard
    ``n{1,2}_prior_mu/sd`` and ``n{1,2}_evidence_mu/sd`` keys in model_inputs.
    For magnitude-comparison front-ends, ``threshold = 0`` (set by the model's
    ``get_model_inputs``) so the formula reduces to plain SNR. For
    :class:`RiskModel`-based front-ends, ``threshold = log(p2/p1)`` so drift
    carries the EU comparison directly.
    """
    post_n1_mu, _ = get_posterior(
        model_inputs['n1_prior_mu'], model_inputs['n1_prior_sd'],
        model_inputs['n1_evidence_mu'], model_inputs['n1_evidence_sd'],
    )
    post_n2_mu, _ = get_posterior(
        model_inputs['n2_prior_mu'], model_inputs['n2_prior_sd'],
        model_inputs['n2_evidence_mu'], model_inputs['n2_evidence_sd'],
    )
    threshold = model_inputs.get('threshold', 0.0)
    diff_mu = (post_n2_mu - post_n1_mu) + threshold
    diff_sd = pt.sqrt(model_inputs['n1_evidence_sd'] ** 2 +
                      model_inputs['n2_evidence_sd'] ** 2)
    v = diff_mu / diff_sd
    if v_scale is not None:
        v = v_scale * v
    return v


class DDMMagnitudeComparisonModel(DDMMixin, MagnitudeComparisonModel):
    """DDM variant of MagnitudeComparisonModel.

    Drift is the subjective signal-to-noise of the perceived log-magnitude
    difference: ``v = v_scale * (post_n2_mu - post_n1_mu) / sqrt(sd1^2 + sd2^2)``.
    Positive drift drives the upper boundary, which corresponds to ``choice=True``
    (i.e. choosing option 2). All cognitive parameters
    (``n1_evidence_sd``, ``n2_evidence_sd``, ``prior_mu``, ``prior_sd``,
    memory model) are inherited unchanged.

    Paradigm columns required: ``n1``, ``n2``, ``choice`` (bool), ``rt`` (seconds).

    Parameters
    ----------
    fit_v_scale : bool
        If True, fit ``v_scale`` as a free parameter. Default ``False`` fixes
        ``v_scale = 1`` and lets ``evidence_sd`` absorb the drift scale (see
        DDMMixin docstring for the identifiability discussion). The default is
        ``False`` because ``v_scale`` and ``evidence_sd`` are strongly correlated;
        fixing one removes a degeneracy that bloats posterior intervals.
    """

    def __init__(self, paradigm=None, fit_prior=False,
                 fit_seperate_evidence_sd=True, memory_model='independent',
                 save_trialwise_n_estimates=False, fit_v_scale=False,
                 fix_z=True):
        self.fit_v_scale = fit_v_scale
        self.fix_z = fix_z
        super().__init__(paradigm=paradigm, fit_prior=fit_prior,
                         fit_seperate_evidence_sd=fit_seperate_evidence_sd,
                         memory_model=memory_model,
                         save_trialwise_n_estimates=save_trialwise_n_estimates)

    def _get_drift(self, model_inputs, parameters):
        v_scale = parameters['v_scale'] if self.fit_v_scale else None
        return _drift_from_snr(model_inputs, v_scale=v_scale)


class DDMFlexibleNoiseComparisonModel(DDMMixin, FlexibleNoiseComparisonModel):
    """DDM variant of FlexibleNoiseComparisonModel.

    Drift uses the same SNR-of-perceived-difference formula as
    :class:`DDMMagnitudeComparisonModel`, but ``n1_evidence_sd`` and
    ``n2_evidence_sd`` are stimulus-dependent splines (polynomial B-spline of
    log-magnitude) rather than scalar parameters per subject. This lets
    discrimination noise vary smoothly with stimulus size.

    Paradigm columns required: ``n1``, ``n2``, ``choice`` (bool), ``rt`` (seconds).

    Parameters
    ----------
    fit_v_scale : bool
        See :class:`DDMMagnitudeComparisonModel`. Default ``False``.
    spline_order, fit_seperate_evidence_sd, fit_prior, memory_model :
        Forwarded to :class:`FlexibleNoiseComparisonModel`.
    """

    def __init__(self, paradigm, fit_seperate_evidence_sd=True,
                 fit_prior=False, spline_order=5,
                 memory_model='independent', fit_v_scale=False,
                 fix_z=True):
        self.fit_v_scale = fit_v_scale
        self.fix_z = fix_z
        FlexibleNoiseComparisonModel.__init__(
            self, paradigm,
            fit_seperate_evidence_sd=fit_seperate_evidence_sd,
            fit_prior=fit_prior,
            spline_order=spline_order,
            memory_model=memory_model,
        )

    def _get_drift(self, model_inputs, parameters):
        v_scale = parameters['v_scale'] if self.fit_v_scale else None
        return _drift_from_snr(model_inputs, v_scale=v_scale)


class DDMRiskModel(DDMMixin, RiskModel):
    """DDM variant of :class:`RiskModel` for risky-choice tasks.

    Drift is the SNR of the perceived log-EU difference:

        v = v_scale * ((post_log_n_2 - post_log_n_1) + log(p_2/p_1)) / sqrt(sd_1^2 + sd_2^2)

    Equivalently, with ``threshold = log(p_2/p_1)`` carried through from
    ``RiskModel.get_model_inputs``, this is just :func:`_drift_from_snr` —
    probabilities enter as a deterministic shift of the perceived
    log-magnitude difference. The Bayesian observer infers ``log(n_k)`` only;
    probabilities are observed precisely. Positive drift drives the upper
    boundary, which corresponds to ``choice=True`` (option 2 chosen).

    Paradigm columns required: ``n1``, ``n2``, ``p1``, ``p2``, ``choice`` (bool),
    ``rt`` (seconds).

    Parameters forwarded to :class:`RiskModel`. ``fit_v_scale`` and ``fix_z``
    follow the same conventions as :class:`DDMMagnitudeComparisonModel`.
    """

    def __init__(self, paradigm=None, prior_estimate='objective',
                 fit_seperate_evidence_sd=True,
                 save_trialwise_n_estimates=False, memory_model='independent',
                 fit_v_scale=False, fix_z=True):
        self.fit_v_scale = fit_v_scale
        self.fix_z = fix_z
        super().__init__(
            paradigm=paradigm, prior_estimate=prior_estimate,
            fit_seperate_evidence_sd=fit_seperate_evidence_sd,
            save_trialwise_n_estimates=save_trialwise_n_estimates,
            memory_model=memory_model,
        )

    def _get_drift(self, model_inputs, parameters):
        v_scale = parameters['v_scale'] if self.fit_v_scale else None
        return _drift_from_snr(model_inputs, v_scale=v_scale)


class DDMFlexibleNoiseRiskModel(DDMMixin, FlexibleNoiseRiskModel):
    """DDM variant of :class:`FlexibleNoiseRiskModel` for risky choice with
    stimulus-dependent (B-spline) encoding noise.

    Drift uses the same SNR-of-perceived-log-EU formula as
    :class:`DDMRiskModel`, but ``σ_k(n)`` is now a per-trial spline rather
    than a scalar.

    Paradigm columns required: ``n1``, ``n2``, ``p1``, ``p2``, ``choice`` (bool),
    ``rt`` (seconds).
    """

    def __init__(self, paradigm, prior_estimate='full',
                 fit_seperate_evidence_sd=True,
                 save_trialwise_n_estimates=False, spline_order=5,
                 representational_noise='payoff',
                 memory_model='independent',
                 fit_v_scale=False, fix_z=True):
        self.fit_v_scale = fit_v_scale
        self.fix_z = fix_z
        FlexibleNoiseRiskModel.__init__(
            self, paradigm,
            prior_estimate=prior_estimate,
            fit_seperate_evidence_sd=fit_seperate_evidence_sd,
            save_trialwise_n_estimates=save_trialwise_n_estimates,
            spline_order=spline_order,
            representational_noise=representational_noise,
            memory_model=memory_model,
        )

    def _get_drift(self, model_inputs, parameters):
        v_scale = parameters['v_scale'] if self.fit_v_scale else None
        return _drift_from_snr(model_inputs, v_scale=v_scale)


class DDMFlexibleNoiseRiskRegressionModel(DDMMixin, FlexibleNoiseRiskRegressionModel):
    """DDM variant of :class:`FlexibleNoiseRiskRegressionModel`.

    Patsy-formula regression on noise spline coefficients (auto-expanded if
    you target ``n1_evidence_sd`` etc.) plus on accumulator params
    (``a``, ``t0``, ``v_scale`` if ``fit_v_scale``). Use for TMS analyses
    where the noise function should differ across stimulation conditions.

    Example:

        regressors = {
            'n1_evidence_sd': 'stimulation_condition',  # auto-expands per spline
            'n2_evidence_sd': 'stimulation_condition',
            'a': 'stimulation_condition',
        }
    """

    def __init__(self, paradigm, regressors, prior_estimate='full',
                 fit_seperate_evidence_sd=True,
                 save_trialwise_n_estimates=False, spline_order=5,
                 representational_noise='payoff',
                 memory_model='independent',
                 fit_v_scale=False, fix_z=True):
        self.fit_v_scale = fit_v_scale
        self.fix_z = fix_z
        FlexibleNoiseRiskRegressionModel.__init__(
            self, paradigm, regressors,
            prior_estimate=prior_estimate,
            fit_seperate_evidence_sd=fit_seperate_evidence_sd,
            save_trialwise_n_estimates=save_trialwise_n_estimates,
            spline_order=spline_order,
            representational_noise=representational_noise,
            memory_model=memory_model,
        )

    def _get_drift(self, model_inputs, parameters):
        v_scale = parameters['v_scale'] if self.fit_v_scale else None
        return _drift_from_snr(model_inputs, v_scale=v_scale)
