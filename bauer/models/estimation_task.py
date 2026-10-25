"""
Bayesian-observer model of magnitude estimation on a bounded response scale.

A participant sees a magnitude ``n`` (e.g. a dot array) and reports it on a
bounded, discrete scale (e.g. an integer slider from 5 to 25).  The model has
three pluggable parts:

**Encoding space** -- ``PowerSpace(alpha)``.  The internal measurement is
Gaussian in a Box-Cox / power-transformed space,

    m ~ N(f_alpha(n), nu_f^2),      f_alpha(n) = (n^alpha - 1) / alpha,

with ``alpha -> 0`` the log space (Weber's law: noise in items grows in
proportion to n) and ``alpha = 1`` the linear space (constant noise in items).
In between, the noise in items grows as ``n^(1 - alpha)`` -- a generalised Weber
law.  ``LogSpace()`` and ``LinearSpace()`` are the two fixed ends;
``PowerSpace()`` (no argument) makes alpha a free, hierarchical parameter.

Efficient coding (Wei & Stocker 2015; Morais & Pillow 2018) is a special case
of the same family: an encoder whose internal space is the CDF of an encoding
prior ``p_enc`` with constant noise there is a power space whenever ``p_enc``
is a power law.  Efficient for a ``1/n`` ("natural") prior = log space;
efficient for the task's uniform prior = linear space; ``p_enc ~ n^-kappa``
gives ``alpha = 1 - kappa``.  So a fitted alpha can be read as the exponent of
the stimulus distribution the encoding is adapted to.

The free noise parameter ``nu`` is the noise **in log units at n_ref**
(a Weber fraction), whatever alpha is; internally ``nu_f = nu * n_ref^alpha``.
This keeps ``nu`` comparable across alpha and removes most of the alpha / noise
trade-off that a raw ``nu_f`` would have.

**Prior** -- the observer's belief about ``n`` used to decode.

* ``UniformPrior(lo, hi)`` -- e.g. the true task distribution (no free
  parameters).  For integer stimuli 5..24 use ``UniformPrior(4.5, 24.5)``.
* ``GaussianPrior()`` -- Gaussian in the *same* encoding space,
  ``N(f_alpha(mu), sd_f^2)``: log-normal for alpha = 0, Gaussian in items for
  alpha = 1.  Free parameters ``prior_mu`` (its centre, in items) and
  ``prior_sd`` (its width in log units at the centre; ``sd_f = prior_sd *
  prior_mu^alpha``).

**Lapse** -- with probability ``p_lapse`` the response is uniform over the
response options (on by default).

**Readout** is the posterior mean of ``n`` (in items); the estimate is reported
directly, rounded to the nearest response option and censored at the scale
ends.  There is no separate motor noise by default (``motor_noise=False``): in a
Bayesian observer the encoding noise already sets both the compression and the
report scatter, and on the round-1 retinonumeral data an extra Gaussian motor
noise was not separately recoverable from the encoding noise (see
``tests/test_estimation_task.py``).  ``motor_noise=True`` adds a Gaussian
``sigma_r`` (items) for users who want it.

Likelihood -- two paths, chosen automatically
----------------------------------------------
* **Closed form** (``LogSpace`` + ``GaussianPrior``, no motor noise): the
  posterior over ``log n`` is Gaussian, so ``log n_hat = w m + (1 - w) mu +
  w nu^2 / 2`` with ``w = sd^2 / (sd^2 + nu^2)``, the report is log-normal,
  ``log r ~ N(w log n + (1 - w) mu + w nu^2 / 2, (w nu)^2)``, and the
  probability of each response option is a difference of normal CDFs at the
  option edges.
* **Grid** (everything else, e.g. the uniform prior or a free alpha): for each
  participant the estimate ``n_hat(m)`` is computed on a measurement grid (the
  posterior mean over a hypothesis grid), then *inverted* at the response-option
  edges (``n_hat`` is monotone in ``m``), so P(option) is again a difference of
  normal CDFs of the measurement -- exact up to the grid's interpolation and
  smooth in the parameters (usable gradients for NUTS).  The measurement grid
  moves with alpha.  The two paths agree to < 1e-3 per option probability on
  the cases where both apply (tested).

The per-trial likelihood is an observed ``pm.Categorical`` over the response
options, so ``pm.compute_log_likelihood`` (LOO / ``az.compare``) and
posterior-predictive sampling work directly.

Example
-------
>>> from bauer.models import EstimationTaskModel, LogSpace, UniformPrior
>>> model = EstimationTaskModel(data, encoding_space=LogSpace(),
...                             prior=UniformPrior(4.5, 24.5), response_range=(5, 25))
>>> model.build_estimation_model(data); idata = model.sample()

``data`` has columns ``n`` (presented magnitude) and ``response`` (integer
report) and a ``subject`` index level.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from ..estimation import EstimationBaseModel


def _norm_cdf(x):
    return 0.5 * pt.erfc(-x / np.sqrt(2.0))


# ============================================================================
# Components
# ============================================================================

class PowerSpace:
    """Encoding space f_alpha(n) = (n^alpha - 1) / alpha (Box-Cox).

    Parameters
    ----------
    alpha : float or None
        Fixed exponent, or ``None`` for a free hierarchical parameter
        ``alpha`` (default prior N(0.5, 0.5) on the group mean, i.e. 95% mass
        on [-0.5, 1.5]).
    """

    def __init__(self, alpha=None):
        self.alpha = alpha

    @property
    def free(self):
        return self.alpha is None

    @property
    def is_log(self):
        return (not self.free) and self.alpha == 0.0

    def transform(self, n, alpha):
        """f_alpha(n); ``alpha`` scalar or tensor broadcastable to ``n``."""
        if not self.free and self.alpha == 0.0:
            return pt.log(n)
        # expm1 form; |alpha| floored at 1e-4 (error ~1e-4 * log(n)^2 / 2)
        a = pt.switch(pt.lt(pt.abs(alpha), 1e-4), 1e-4, alpha)
        return pt.expm1(a * pt.log(n)) / a

    def transform_np(self, n, alpha):
        if alpha == 0.0:
            return np.log(n)
        return np.expm1(alpha * np.log(n)) / alpha

    def __repr__(self):
        return f'PowerSpace(alpha={self.alpha})'


def LogSpace():
    """Log encoding (Weber's law): ``PowerSpace(0)``."""
    return PowerSpace(0.0)


def LinearSpace():
    """Linear encoding (constant noise in items): ``PowerSpace(1)``."""
    return PowerSpace(1.0)


class UniformPrior:
    """Uniform prior on [lo, hi] (items); no free parameters."""
    free_parameters = ()

    def __init__(self, lo, hi):
        self.lo, self.hi = float(lo), float(hi)

    def __repr__(self):
        return f'UniformPrior({self.lo}, {self.hi})'


class GaussianPrior:
    """Gaussian prior in the encoding space (log-normal for log space).

    Free parameters: ``prior_mu`` (centre, items) and ``prior_sd`` (width in
    log units at the centre).
    """
    free_parameters = ('prior_mu', 'prior_sd')

    def __repr__(self):
        return 'GaussianPrior()'


# Alias in the spirit of the docs (log-normal when the space is log)
LogNormalPrior = GaussianPrior


# ============================================================================
# Model
# ============================================================================

class EstimationTaskModel(EstimationBaseModel):
    """Hierarchical Bayesian observer for magnitude estimation (see module doc).

    Parameters
    ----------
    paradigm : pd.DataFrame
        ``n`` (presented magnitude, > 0), ``response`` (integer); ``subject``
        index level for hierarchical fits.
    encoding_space : PowerSpace
    prior : UniformPrior or GaussianPrior
    lapse : bool
    motor_noise : bool
    response_range : (int, int)
        Smallest and largest response option (inclusive integers, step 1).
    n_ref : float
        Magnitude at which ``nu`` is expressed (log units).
    n_measurement_grid, n_hypothesis_grid : int
    hypothesis_range : (float, float)
        Support of the observer's hypothesis grid (items).
    """

    def __init__(self, paradigm=None, encoding_space=None, prior=None, lapse=True,
                 motor_noise=False, response_range=(5, 25), n_ref=12.0,
                 n_measurement_grid=200, n_hypothesis_grid=246,
                 hypothesis_range=(1.0, 50.0), **kwargs):
        self.encoding_space = encoding_space if encoding_space is not None else LogSpace()
        self.prior = prior if prior is not None else UniformPrior(4.5, 24.5)
        self.lapse = lapse
        self.motor_noise = motor_noise
        self.response_range = tuple(int(x) for x in response_range)
        self.n_ref = float(n_ref)
        self.n_measurement_grid = int(n_measurement_grid)
        self.n_hypothesis_grid = int(n_hypothesis_grid)
        self.hypothesis_range = tuple(float(x) for x in hypothesis_range)
        super().__init__(paradigm, lapse_rate=0.0, **kwargs)
        self._setup_grids()

    @property
    def closed_form(self):
        """True when the likelihood is computed analytically (no grid)."""
        return (self.encoding_space.is_log and isinstance(self.prior, GaussianPrior)
                and not self.motor_noise)

    # ------------------------------------------------------------------
    def get_free_parameters(self):
        pars = {'nu': dict(mu_intercept=-1.0, sigma_intercept=0.75,
                           cauchy_sigma_intercept=0.5, transform='softplus')}
        if self.encoding_space.free:
            pars['alpha'] = dict(mu_intercept=0.5, sigma_intercept=0.5,
                                 cauchy_sigma_intercept=0.3, transform='identity')
        if isinstance(self.prior, GaussianPrior):
            pars['prior_mu'] = dict(mu_intercept=12.0, sigma_intercept=4.0,
                                    cauchy_sigma_intercept=2.0, transform='softplus')
            pars['prior_sd'] = dict(mu_intercept=-0.5, sigma_intercept=0.75,
                                    cauchy_sigma_intercept=0.5, transform='softplus')
        if self.motor_noise:
            pars['sigma_r'] = dict(mu_intercept=1.0, sigma_intercept=1.0,
                                   cauchy_sigma_intercept=0.75, transform='softplus')
        if self.lapse:
            pars['p_lapse'] = dict(transform='beta', beta_mu_mean=0.05, beta_mu_kappa=8.0)
        return pars

    def get_model_inputs(self, parameters):
        return {key: self.subjectwise(key) for key in self.free_parameters}

    # ------------------------------------------------------------------
    def _setup_grids(self, paradigm=None):
        rlo, rhi = self.response_range
        self.response_options = np.arange(rlo, rhi + 1, dtype=float)
        # finite outer edges (censoring); infinite ones give NaN gradients
        self.response_edges = np.concatenate(
            [[1e-3], self.response_options[:-1] + 0.5, [1e4]])
        h0, h1 = self.hypothesis_range
        self.hyp_grid = np.linspace(h0, h1, self.n_hypothesis_grid)
        self.d_hyp = self.hyp_grid[1] - self.hyp_grid[0]
        # measurement grid in units of the hypothesis range in f-space
        self.meas_t = np.linspace(-0.5, 1.5, self.n_measurement_grid)
        if isinstance(self.prior, UniformPrior):
            a, b = self.prior.lo, self.prior.hi
            left = np.clip(self.hyp_grid - self.d_hyp / 2, a, b)
            right = np.clip(self.hyp_grid + self.d_hyp / 2, a, b)
            w = right - left
            self.uniform_prior_mass = w / w.sum()

    def _get_paradigm(self, paradigm=None, subject_mapping=None):
        if paradigm is None:
            paradigm = self.paradigm
        rlo, rhi = self.response_range
        paradigm_ = {'_data_n': len(paradigm)}
        if 'subject' in paradigm.index.names:
            paradigm_['subject_ix'], _ = pd.factorize(paradigm.index.get_level_values('subject'))
        elif 'subject' in paradigm.columns:
            paradigm_['subject_ix'], _ = pd.factorize(paradigm['subject'])
        else:
            paradigm_['subject_ix'] = np.zeros(len(paradigm), dtype=int)
        paradigm_['n'] = paradigm['n'].values.astype(float)
        if 'response' in paradigm.columns:
            r = paradigm['response'].values.astype(float)
            if np.any(np.isnan(r)):
                raise ValueError('response has NaNs; drop no-response trials first')
            r = np.clip(np.round(r), rlo, rhi)
            paradigm_['response_ix'] = (r - rlo).astype(int)
        else:
            paradigm_['response_ix'] = np.zeros(len(paradigm), dtype=int)
        return paradigm_

    # ------------------------------------------------------------------
    # Observer
    # ------------------------------------------------------------------
    def _alpha(self, inputs, S):
        if self.encoding_space.free:
            return inputs['alpha']
        return pt.ones((S,)) * self.encoding_space.alpha

    def _f(self, x, alpha):
        """f_alpha(x) for x (P,) and alpha (S,) -> (S, P)."""
        # (+ 0 * alpha keeps the subject axis when the space ignores alpha)
        return self.encoding_space.transform(x[None, :], alpha[:, None]) + 0.0 * alpha[:, None]

    def _nu_f(self, inputs, alpha):
        return inputs['nu'] * self.n_ref ** alpha                       # (S,)

    def _option_probs_closed(self, inputs, n, subject_ix):
        """(n_trials, R) for log space + Gaussian (log-normal) prior."""
        nu = inputs['nu'][subject_ix]
        mu = pt.log(inputs['prior_mu'][subject_ix])
        sd = inputs['prior_sd'][subject_ix]
        w = sd ** 2 / (sd ** 2 + nu ** 2)
        loc = w * pt.log(n) + (1 - w) * mu + w * nu ** 2 / 2
        scale = w * nu
        edges = pt.log(pt.as_tensor_variable(self.response_edges))
        cdf = _norm_cdf((edges[None, :] - loc[:, None]) / scale[:, None])
        return cdf[:, 1:] - cdf[:, :-1]

    def estimate_on_grid(self, inputs, S):
        """Measurement grid m (S, M) and posterior-mean estimate n_hat(m) (S, M)."""
        alpha = self._alpha(inputs, S)
        nu_f = self._nu_f(inputs, alpha)
        h = pt.as_tensor_variable(self.hyp_grid)
        fh = self._f(h, alpha)                                           # (S, H)
        lo = fh[:, 0][:, None]
        hi = fh[:, -1][:, None]
        m = lo + pt.as_tensor_variable(self.meas_t)[None, :] * (hi - lo)  # (S, M)
        loglik = -0.5 * ((m[:, :, None] - fh[:, None, :]) / nu_f[:, None, None]) ** 2
        if isinstance(self.prior, UniformPrior):
            logprior = pt.log(pt.as_tensor_variable(self.uniform_prior_mass) + 1e-300)[None, :]
        else:
            mu_f = self.encoding_space.transform(inputs['prior_mu'], alpha)          # (S,)
            sd_f = inputs['prior_sd'] * inputs['prior_mu'] ** alpha
            # density in n: N(f(h); mu_f, sd_f) * f'(h),  f'(h) = h^(alpha - 1)
            logprior = (-0.5 * ((fh - mu_f[:, None]) / sd_f[:, None]) ** 2
                        + (alpha[:, None] - 1) * pt.log(h)[None, :])
        logpost = loglik + logprior[:, None, :]
        logpost = logpost - pt.max(logpost, axis=2, keepdims=True)
        post = pt.exp(logpost)
        post = post / pt.sum(post, axis=2, keepdims=True)
        n_hat = pt.sum(post * h[None, None, :], axis=2)                  # (S, M)
        return m, n_hat, alpha, nu_f

    def _invert(self, m, n_hat, values):
        """m at which the monotone n_hat(m) crosses each of ``values`` (R+1,).

        Linear interpolation per participant; values outside the range of
        n_hat map to the grid ends (where the measurement density is ~0).
        """
        M = self.n_measurement_grid
        v = pt.as_tensor_variable(values)
        below = pt.sum(pt.lt(n_hat[:, :, None], v[None, None, :]), axis=1)    # (S, V)
        j = pt.clip(below - 1, 0, M - 2)
        grid_ix = pt.arange(M)
        oh0 = pt.eq(grid_ix[None, None, :], j[:, :, None])                    # (S, V, M)
        oh1 = pt.eq(grid_ix[None, None, :], (j + 1)[:, :, None])
        n0 = pt.sum(oh0 * n_hat[:, None, :], axis=2)
        n1 = pt.sum(oh1 * n_hat[:, None, :], axis=2)
        m0 = pt.sum(oh0 * m[:, None, :], axis=2)
        m1 = pt.sum(oh1 * m[:, None, :], axis=2)
        frac = pt.clip((v[None, :] - n0) / (n1 - n0 + 1e-12), 0.0, 1.0)
        out = m0 + frac * (m1 - m0)
        # entirely below / above the estimate's range
        out = pt.switch(pt.eq(below, 0), m[:, :1] - 1e3, out)
        out = pt.switch(pt.eq(below, M), m[:, -1:] + 1e3, out)
        return out                                                           # (S, V)

    def _option_probs_grid(self, inputs, n, subject_ix, S):
        m, n_hat, alpha, nu_f = self.estimate_on_grid(inputs, S)
        fn = self.encoding_space.transform(n, alpha[subject_ix])            # (T,)
        nu_t = nu_f[subject_ix]
        if not self.motor_noise:
            m_edges = self._invert(m, n_hat, self.response_edges)          # (S, R+1)
            z = (m_edges[subject_ix] - fn[:, None]) / nu_t[:, None]
            cdf = _norm_cdf(z)
            return cdf[:, 1:] - cdf[:, :-1]
        # motor noise: integrate the measurement density over the grid
        dens = pt.exp(-0.5 * ((m[subject_ix] - fn[:, None]) / nu_t[:, None]) ** 2)
        dens = dens / pt.sum(dens, axis=1, keepdims=True)                   # (T, M)
        edges = pt.as_tensor_variable(self.response_edges)
        sig = inputs['sigma_r'][:, None, None]
        cdf = _norm_cdf((edges[None, None, :] - n_hat[:, :, None]) / sig)   # (S, M, R+1)
        kern = cdf[:, :, 1:] - cdf[:, :, :-1]                               # (S, M, R)
        return pt.sum(dens[:, :, None] * kern[subject_ix], axis=1)

    def trial_probabilities(self, inputs, n, subject_ix, S):
        if self.closed_form:
            p = self._option_probs_closed(inputs, n, subject_ix)
        else:
            p = self._option_probs_grid(inputs, n, subject_ix, S)
        if self.lapse:
            lam = inputs['p_lapse'][subject_ix][:, None]
            p = (1 - lam) * p + lam / len(self.response_options)
        p = pt.clip(p, 1e-10, 1.0)
        return p / pt.sum(p, axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # PyMC graph
    # ------------------------------------------------------------------
    def build_estimation_model(self, data=None, coords=None, hierarchical=True,
                               save_p_choice=False, flat_prior=False, paradigm=None):
        if data is None:
            data = paradigm if paradigm is not None else self.paradigm
        self.paradigm = data
        if hierarchical and coords is None:
            coords = {'subject': data.index.unique(level='subject')}
        with pm.Model(coords=coords) as self.estimation_model:
            paradigm_ = self._get_paradigm(paradigm=data)
            self.set_paradigm(paradigm_)
            self.build_priors(hierarchical=hierarchical, flat_prior=flat_prior)
            self.build_likelihood(None, save_p_choice=save_p_choice)

    def build_likelihood(self, parameters, save_p_choice=False):
        model = pm.Model.get_context()
        inputs = self.get_model_inputs(parameters)
        S = len(model.coords['subject']) if 'subject' in model.coords else 1
        p = self.trial_probabilities(inputs, model['n'], model['subject_ix'], S)
        if save_p_choice:
            p = pm.Deterministic('p_response', p)
        pm.Categorical('response', p=p, observed=model['response_ix'])

    # ------------------------------------------------------------------
    # Prediction / simulation / PPC / recovery
    # ------------------------------------------------------------------
    def _subjects(self, paradigm):
        if 'subject' in paradigm.index.names:
            subj = paradigm.index.get_level_values('subject')
        elif 'subject' in paradigm.columns:
            subj = paradigm['subject'].values
        else:
            subj = np.zeros(len(paradigm), dtype=int)
        return pd.factorize(subj)

    def predict_response_distribution(self, paradigm, parameters):
        """Per-trial probabilities over the response options (DataFrame).

        ``parameters``: DataFrame indexed by subject, or a dict of scalars.
        """
        codes, uniques = self._subjects(paradigm)
        if isinstance(parameters, pd.DataFrame):
            pars = parameters.loc[list(uniques)]
            arrs = {k: pars[k].values.astype(float) for k in self.free_parameters}
        else:
            arrs = {k: np.broadcast_to(np.atleast_1d(np.asarray(parameters[k], float)),
                                       (len(uniques),)).copy() for k in self.free_parameters}
        inputs = {k: pt.as_tensor_variable(v) for k, v in arrs.items()}
        p = self.trial_probabilities(inputs, pt.as_tensor_variable(paradigm['n'].values.astype(float)),
                                     pt.as_tensor_variable(codes), len(uniques)).eval()
        return pd.DataFrame(p, index=paradigm.index,
                            columns=pd.Index(self.response_options.astype(int), name='response'))

    def predict(self, paradigm, parameters):
        """Predicted mean and s.d. of the report per trial."""
        p = self.predict_response_distribution(paradigm, parameters).values
        r = self.response_options
        mean = p @ r
        out = paradigm.copy()
        out['predicted_mean'] = mean
        out['predicted_sd'] = np.sqrt(np.maximum(p @ r ** 2 - mean ** 2, 0))
        return out

    def simulate(self, paradigm, parameters, n_samples=1, rng=None):
        """Simulated integer reports (lapses and censoring included).

        Long DataFrame with a ``sample`` index level and ``simulated_response``.
        """
        rng = np.random.default_rng(rng)
        p = self.predict_response_distribution(paradigm, parameters).values
        cdf = np.cumsum(p, axis=1)
        u = rng.random((n_samples, len(p), 1))
        ix = np.minimum((u > cdf[None]).sum(-1), p.shape[1] - 1)
        sim = self.response_options[ix].T
        par = paradigm.drop(columns=['response'], errors='ignore')
        out = pd.concat([par.assign(sample=k + 1, simulated_response=sim[:, k])
                         for k in range(n_samples)])
        return out.set_index('sample', append=True)

    def posterior_predictive(self, idata=None, n_posterior_samples=200, random_seed=None):
        """Simulated datasets (same trials as the fit), one per posterior draw.

        Long DataFrame: index = fitted paradigm's index + ``ppc_sample``,
        column ``simulated_response``.
        """
        import arviz as az
        idata = self._get_idata(idata)
        post = idata.posterior.stack(sample=('chain', 'draw'))
        rng = np.random.default_rng(random_seed)
        sel = rng.choice(post.sizes['sample'], size=min(n_posterior_samples, post.sizes['sample']),
                         replace=False)
        thin = post.isel(sample=sel).drop_vars(['sample', 'chain', 'draw'])
        thin = thin.assign_coords(sample=np.arange(len(sel))).rename(sample='draw')
        thin = thin.expand_dims('chain').transpose('chain', 'draw', ...)
        with self.estimation_model:
            pp = pm.sample_posterior_predictive(az.InferenceData(posterior=thin),
                                                var_names=['response'],
                                                random_seed=random_seed, progressbar=False)
        arr = pp.posterior_predictive['response'].values[0]
        sim = self.response_options[arr].T
        out = pd.DataFrame(sim, index=self.paradigm.index,
                           columns=pd.Index(np.arange(len(sel)), name='ppc_sample'))
        return out.stack().rename('simulated_response').to_frame()

    ppc = posterior_predictive

    def parameter_recovery(self, paradigm, parameters, rng=None, **sample_kwargs):
        """Simulate one dataset from ``parameters`` (DataFrame by subject) on
        ``paradigm``'s trials, refit a fresh copy of this model hierarchically,
        and return true vs recovered (posterior mean) parameters per subject."""
        sim = self.simulate(paradigm, parameters, n_samples=1, rng=rng)
        data = sim.droplevel('sample').rename(columns={'simulated_response': 'response'})
        fit = EstimationTaskModel(data, encoding_space=self.encoding_space, prior=self.prior,
                                  lapse=self.lapse, motor_noise=self.motor_noise,
                                  response_range=self.response_range, n_ref=self.n_ref,
                                  n_measurement_grid=self.n_measurement_grid,
                                  n_hypothesis_grid=self.n_hypothesis_grid,
                                  hypothesis_range=self.hypothesis_range)
        fit.build_estimation_model(data, hierarchical=True)
        idata = fit.sample(**sample_kwargs)
        rec = pd.DataFrame({k: idata.posterior[k].mean(('chain', 'draw')).to_series()
                            for k in self.free_parameters})
        return pd.concat([parameters[list(self.free_parameters)], rec.loc[parameters.index]],
                         axis=1, keys=['true', 'recovered']), idata
