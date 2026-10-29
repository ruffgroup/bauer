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
ends.  ``motor_noise=True`` adds Gaussian response noise ``sigma_r`` (items)
around the estimate before rounding.  It is separable from the encoding noise
because the encoding noise also sets the compression (through the prior) and
motor noise does not: on a 19-participant, ~500-trial-per-participant design
with a uniform prior, both recovered at r > .93.  With ``motor_noise=False`` the
encoding noise alone sets the scatter, and the closed-form path is available
for log space + log-normal prior.

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

Censored (unfinished) responses
-------------------------------
A ``censor`` column (-1 / 0 / +1) marks reports that were cut short, e.g. a slider
still moving away from its start when the response window closed: +1 means the
intended report is at least the final position, -1 at most. Such a trial's
likelihood is the summed probability of the response options beyond (and
including) the final one, under the same model; no extra parameters. Without the
column (or with all zeros) the likelihood is unchanged. Posterior-predictive
draws are the intended (uncensored) reports.

The per-trial likelihood is an observed ``pm.Categorical`` (``pm.CustomDist``
with censoring) over the response options, so ``pm.compute_log_likelihood`` (LOO / ``az.compare``) and
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
    """Uniform prior on [lo, hi] (items); no free parameters.

    ``discrete=True``: equal mass on the integers lo, lo + 1, ..., hi (the
    observer believes only in the magnitudes that can occur, e.g.
    ``UniformPrior(5, 24, discrete=True)``).  Otherwise a continuous uniform
    density on [lo, hi] (for integers 5..24 use ``UniformPrior(4.5, 24.5)``).
    """
    free_parameters = ()
    truncated = True

    def __init__(self, lo, hi, discrete=False):
        self.lo, self.hi, self.discrete = float(lo), float(hi), bool(discrete)

    def __repr__(self):
        return f'UniformPrior({self.lo}, {self.hi}, discrete={self.discrete})'


class GaussianPrior:
    """Gaussian prior in the encoding space (log-normal for log space).

    Free parameters: ``prior_mu`` (centre, items) and ``prior_sd`` (width in
    log units at the centre).  Its support is the whole hypothesis grid
    (``hypothesis_range``), so it does not know the task's bounds.
    """
    free_parameters = ('prior_mu', 'prior_sd')
    truncated = False
    discrete = False

    def __repr__(self):
        return 'GaussianPrior()'


class TruncatedGaussianPrior(GaussianPrior):
    """``GaussianPrior`` restricted to [lo, hi] (the bounds participants know).

    ``discrete=True`` (default): mass only on the integers lo..hi, with weights
    proportional to the prior's density in n at each integer,
    ``N(f(k); mu_f, sd_f) * f'(k)`` (the same density-in-n convention as the
    continuous prior, evaluated at the integers).  ``space='n'``: Gaussian in
    numerosity itself, mass on k proportional to N(k; prior_mu, prior_sd) with both
    in ITEMS, so the prior does not depend on alpha.  ``discrete=False``: the
    continuous density with the mass outside [lo, hi] zeroed (e.g. lo = 4.5,
    hi = 24.5).  A narrow prior centred mid-range reproduces ``GaussianPrior``.
    A very wide one becomes uniform in the encoding space: ``UniformPrior``
    with the same support for the linear space, a 1/n prior for the log space.
    """
    truncated = True

    def __init__(self, lo, hi, discrete=True, space='internal'):
        if space not in ('internal', 'n'):
            raise ValueError("space must be 'internal' or 'n'")
        self.lo, self.hi, self.discrete, self.space = float(lo), float(hi), bool(discrete), space

    def __repr__(self):
        return (f'TruncatedGaussianPrior({self.lo}, {self.hi}, discrete={self.discrete}, '
                f'space={self.space!r})')


# Alias in the spirit of the docs (log-normal when the space is log)
LogNormalPrior = GaussianPrior


def _censored_logp(R):
    """Per-trial log-likelihood with optional censoring (see build_likelihood)."""
    def logp(value, p, censor):
        value = pt.cast(value, 'int64')
        oh = pt.eq(pt.arange(R)[None, :], value[:, None])                  # (T, R)
        p_obs = pt.sum(p * oh, axis=1)
        cum = pt.cumsum(p, axis=1)
        p_le = pt.sum(cum * oh, axis=1)                                     # P(option <= final)
        p_ge = 1.0 - p_le + p_obs                                           # P(option >= final)
        lik = pt.switch(pt.eq(censor, 0), p_obs, pt.switch(pt.gt(censor, 0), p_ge, p_le))
        return pt.log(pt.clip(lik, 1e-12, 1.0))
    return logp


def _censored_random(p, censor, rng=None, size=None):
    """Uncensored draws (the report the observer intended)."""
    p = np.asarray(p)
    u = rng.random(p.shape[:-1])[..., None]
    return np.minimum((u > np.cumsum(p, axis=-1)).sum(-1), p.shape[-1] - 1)


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
                 n_measurement_grid=None, n_hypothesis_grid=246,
                 hypothesis_range=(1.0, 50.0), covariates=None, condition=None,
                 condition_parameters=(), measurement_shift=None, noise_modulator=None,
                 n_covariate_bins=7, leak=None, prior_leak=None, **kwargs):
        self.covariates = dict(covariates or {})
        # leak: column with a second (uncued) magnitude n_u; the measurement is centred on
        # (1 - w) f(n) + w f(n_u), the observer decodes it as if it came from n alone
        self.leak = leak
        # prior_leak: column with n_u; the prior centre on that trial is mu + k (n_u - mu)
        # (parameter prior_leak_k) and the observer decodes with that prior. With leak
        # as well (same column), both act.
        if leak and prior_leak and leak != prior_leak:
            raise ValueError('leak and prior_leak must name the same column')
        self.prior_leak = prior_leak
        self.measurement_shift = measurement_shift
        self.noise_modulator = noise_modulator
        self.n_covariate_bins = int(n_covariate_bins)
        self.condition = condition
        self.condition_parameters = tuple(condition_parameters)
        self.encoding_space = encoding_space if encoding_space is not None else LogSpace()
        self.prior = prior if prior is not None else UniformPrior(4.5, 24.5)
        self.lapse = lapse
        self.motor_noise = motor_noise
        self.response_range = tuple(int(x) for x in response_range)
        self.n_ref = float(n_ref)
        # Accuracy (checked against M = 4000 over a 3.5x wider range, at fitted
        # parameters): with motor noise 200 points give |dP| < 1e-4; without it the
        # inversion of n_hat(m) needs ~600 for |dP| < 1e-3.
        if n_measurement_grid is None:
            n_measurement_grid = 200 if motor_noise else 600
        self.n_measurement_grid = int(n_measurement_grid)
        self.n_hypothesis_grid = int(n_hypothesis_grid)
        self.hypothesis_range = tuple(float(x) for x in hypothesis_range)
        super().__init__(paradigm, lapse_rate=0.0, **kwargs)
        self._setup_grids()

    @property
    def closed_form(self):
        """True when the likelihood is computed analytically (no grid)."""
        return (self.encoding_space.is_log and type(self.prior) is GaussianPrior
                and not self.motor_noise and not getattr(self, 'measurement_shift', None)
                and not getattr(self, 'noise_modulator', None)
                and not getattr(self, 'leak', None)
                and not getattr(self, 'prior_leak', None))

    # ------------------------------------------------------------------
    def get_free_parameters(self):
        pars = {'nu': dict(mu_intercept=-1.0, sigma_intercept=0.75,
                           cauchy_sigma_intercept=0.5, transform='softplus')}
        if self.encoding_space.free:
            pars['alpha'] = dict(mu_intercept=0.5, sigma_intercept=0.5,
                                 cauchy_sigma_intercept=0.3, transform='identity')
        if isinstance(self.prior, GaussianPrior) and getattr(self.prior, 'space', 'internal') == 'n':
            pars['prior_mu'] = dict(mu_intercept=14.5, sigma_intercept=4.0,
                                    cauchy_sigma_intercept=2.0, transform='softplus')
            pars['prior_sd'] = dict(mu_intercept=6.0, sigma_intercept=3.0,
                                    cauchy_sigma_intercept=1.5, transform='softplus')
        elif isinstance(self.prior, GaussianPrior):
            pars['prior_mu'] = dict(mu_intercept=12.0, sigma_intercept=4.0,
                                    cauchy_sigma_intercept=2.0, transform='softplus')
            pars['prior_sd'] = dict(mu_intercept=-0.5, sigma_intercept=0.75,
                                    cauchy_sigma_intercept=0.5, transform='softplus')
        if self.motor_noise:
            pars['sigma_r'] = dict(mu_intercept=1.0, sigma_intercept=1.0,
                                   cauchy_sigma_intercept=0.75, transform='softplus')
        if getattr(self, 'measurement_shift', None):
            pars['shift_beta'] = dict(mu_intercept=0.0, sigma_intercept=0.5,
                                      cauchy_sigma_intercept=0.25, transform='identity')
        if getattr(self, 'noise_modulator', None):
            pars['noise_gamma'] = dict(mu_intercept=0.0, sigma_intercept=0.5,
                                       cauchy_sigma_intercept=0.25, transform='identity')
        if getattr(self, 'leak', None):
            pars['leak_w'] = dict(mu_intercept=0.0, sigma_intercept=0.25,
                                  cauchy_sigma_intercept=0.1, transform='identity')
        if getattr(self, 'prior_leak', None):
            if not (isinstance(self.prior, GaussianPrior)
                    and getattr(self.prior, 'space', 'internal') == 'n'):
                raise NotImplementedError('prior_leak needs a Gaussian prior in items')
            pars['prior_leak_k'] = dict(mu_intercept=0.0, sigma_intercept=0.25,
                                        cauchy_sigma_intercept=0.1, transform='identity')
        if self.lapse:
            pars['p_lapse'] = dict(transform='beta', beta_mu_mean=0.05, beta_mu_kappa=8.0)
        return pars

    def get_model_inputs(self, parameters):
        return {key: self.subjectwise(key) for key in self.free_parameters}

    def build_hierarchical_nodes(self, name, mu_intercept=None, sigma_intercept=None,
                                 cauchy_sigma_intercept=None, transform='identity', **kwargs):
        """Hierarchical node, optionally with a subject-level covariate.

        With ``covariates={name: column}`` the participant values of ``name``
        (on the untransformed scale) are ``mu + beta * z_s + sd * offset_s``,
        where ``z_s`` is the (standardised) column, constant within subject.
        The slope is ``{name}_{column}``, prior N(0, sigma_intercept).
        """
        if name not in self.covariates:
            return super().build_hierarchical_nodes(
                name, mu_intercept=mu_intercept, sigma_intercept=sigma_intercept,
                cauchy_sigma_intercept=cauchy_sigma_intercept, transform=transform, **kwargs)
        if transform not in ('identity', 'softplus'):
            raise NotImplementedError('covariates only for identity / softplus parameters')
        from ..core import _group_sd_rv
        col = self.covariates[name]
        data = self.paradigm
        z = pd.Series(data[col].values, index=data.index.get_level_values('subject'))
        z = z.groupby(level=0, sort=False).first()
        model = pm.Model.get_context()
        z = z.loc[list(model.coords['subject'])].values.astype(float)
        mu_name = f'{name}_mu' if transform == 'identity' else f'{name}_mu_untransformed'
        group_mu = pm.Normal(mu_name, mu=mu_intercept, sigma=sigma_intercept)
        if transform == 'softplus':
            pm.Deterministic(f'{name}_mu', pt.softplus(group_mu))
        beta = pm.Normal(f'{name}_{col}', mu=0.0, sigma=sigma_intercept)
        group_sd = _group_sd_rv(f'{name}_sd', cauchy_sigma_intercept,
                                getattr(self, 'group_sd_dist', 'halfnormal'))
        offset = pm.Normal(f'{name}_offset', 0.0, 1.0, dims=('subject',))
        x = group_mu + beta * z + group_sd * offset
        if transform == 'identity':
            return pm.Deterministic(name, x, dims=('subject',))
        pm.Deterministic(f'{name}_untransformed', x, dims=('subject',))
        return pm.Deterministic(name, pt.softplus(x), dims=('subject',))

    # ------------------------------------------------------------------
    def _setup_grids(self, paradigm=None):
        rlo, rhi = self.response_range
        self.response_options = np.arange(rlo, rhi + 1, dtype=float)
        # finite outer edges (censoring); infinite ones give NaN gradients
        self.response_edges = np.concatenate(
            [[1e-3], self.response_options[:-1] + 0.5, [1e4]])
        h0, h1 = self.hypothesis_range
        self.meas_range = (h0, h1)        # the measurement grid always spans f(h0)..f(h1)
        if getattr(self.prior, 'discrete', False):
            # the observer believes only in the integers lo..hi
            self.hyp_grid = np.arange(self.prior.lo, self.prior.hi + 1, dtype=float)
            self.d_hyp = 1.0
            self.prior_mask = np.ones_like(self.hyp_grid)
        else:
            self.hyp_grid = np.linspace(h0, h1, self.n_hypothesis_grid)
            self.d_hyp = self.hyp_grid[1] - self.hyp_grid[0]
            if getattr(self.prior, 'truncated', False):
                a, b = self.prior.lo, self.prior.hi
                left = np.clip(self.hyp_grid - self.d_hyp / 2, a, b)
                right = np.clip(self.hyp_grid + self.d_hyp / 2, a, b)
                self.prior_mask = (right - left) / self.d_hyp      # fraction of each cell inside
            else:
                self.prior_mask = np.ones_like(self.hyp_grid)
        self.prior_mask = self.prior_mask / self.prior_mask.sum()
        # measurement grid in units of f(h1) - f(h0)
        self.meas_t = np.linspace(-0.5, 1.5, self.n_measurement_grid)

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
        for col, key in ((getattr(self, 'measurement_shift', None), 'shift'),
                         (getattr(self, 'noise_modulator', None), 'noise')):
            if col is None:
                continue
            ix, centres = self._bin_covariate(paradigm[col].values.astype(float),
                                              np.asarray(paradigm_['subject_ix']))
            paradigm_[f'{key}_bin'] = ix
            setattr(self, f'{key}_centres', centres)
        if self.condition is not None:
            vals = (paradigm.index.get_level_values(self.condition)
                    if self.condition in paradigm.index.names else paradigm[self.condition].values)
            self.condition_levels = sorted(pd.unique(vals))
            if len(self.condition_levels) != 2:
                raise ValueError('condition must have exactly two levels')
            paradigm_['cond_ix'] = (np.asarray(vals) == self.condition_levels[1]).astype(int)
        # grid path works per unique magnitude (the graph captures n_unique at build)
        self.n_unique, paradigm_['n_ix'] = np.unique(paradigm_['n'], return_inverse=True)
        if getattr(self, 'leak', None) or getattr(self, 'prior_leak', None):
            self.n_unique, paradigm_['n_ix'] = self._pair_index(
                paradigm_['n'], paradigm[self.leak or self.prior_leak].values)
        if 'response' in paradigm.columns:
            r = paradigm['response'].values.astype(float)
            if np.any(np.isnan(r)):
                raise ValueError('response has NaNs; drop no-response trials first')
            r = np.clip(np.round(r), rlo, rhi)
            paradigm_['response_ix'] = (r - rlo).astype(int)
        else:
            paradigm_['response_ix'] = np.zeros(len(paradigm), dtype=int)
        self.censored = 'censor' in paradigm.columns
        if self.censored:
            c = paradigm['censor'].fillna(0).values.astype(int)
            if not np.isin(c, (-1, 0, 1)).all():
                raise ValueError('censor must be -1, 0 or +1')
            paradigm_['censor'] = c
        return paradigm_

    def _pair_index(self, n, u):
        """(K, 2) table of (n, n_u) and each trial's row in it.

        leak only: the unique pairs. With prior_leak: the full grid, n_u-major
        (row = u_ix * N + n_ix), so the table can be computed per (participant, n_u).
        """
        n, u = np.asarray(n, float), np.asarray(u, float)
        if not getattr(self, 'prior_leak', None):
            pairs, ix = np.unique(np.column_stack([n, u]), axis=0, return_inverse=True)
            return pairs, np.asarray(ix).ravel()
        nv, n_ix = np.unique(n, return_inverse=True)
        uv, u_ix = np.unique(u, return_inverse=True)
        pairs = np.column_stack([np.tile(nv, len(uv)), np.repeat(uv, len(nv))])
        return pairs, np.asarray(u_ix).ravel() * len(nv) + np.asarray(n_ix).ravel()

    def _bin_covariate(self, x, subject_ix):
        """Within-participant quantile bins of a trial covariate.

        Returns per-trial bin index and the (S, B) bin means (the value used for all
        trials of a bin). Ties are broken by rank, so bins are equally filled.
        """
        B = self.n_covariate_bins
        S = int(subject_ix.max()) + 1
        ix = np.zeros(len(x), dtype=int)
        centres = np.zeros((S, B))
        for s_ in range(S):
            m = subject_ix == s_
            r = pd.Series(x[m]).rank(method='first').values
            b = np.minimum((B * (r - 1) / m.sum()).astype(int), B - 1)
            ix[m] = b
            centres[s_] = [x[m][b == k].mean() for k in range(B)]
        return ix, centres

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
        f_ends = self._f(pt.as_tensor_variable(np.array(self.meas_range)), alpha)   # (S, 2)
        lo = f_ends[:, :1]
        hi = f_ends[:, 1:]
        m = lo + pt.as_tensor_variable(self.meas_t)[None, :] * (hi - lo)  # (S, M)
        loglik = -0.5 * ((m[:, :, None] - fh[:, None, :]) / nu_f[:, None, None]) ** 2
        logmask = pt.log(pt.as_tensor_variable(self.prior_mask) + 1e-300)[None, :]
        if isinstance(self.prior, UniformPrior):
            logprior = logmask
        elif getattr(self.prior, 'space', 'internal') == 'n':
            # Gaussian in items, independent of alpha
            logprior = (-0.5 * ((h[None, :] - inputs['prior_mu'][:, None])
                                / inputs['prior_sd'][:, None]) ** 2 + logmask)
        else:
            mu_f = self.encoding_space.transform(inputs['prior_mu'], alpha)          # (S,)
            sd_f = inputs['prior_sd'] * inputs['prior_mu'] ** alpha
            # density in n: N(f(h); mu_f, sd_f) * f'(h),  f'(h) = h^(alpha - 1)
            logprior = (-0.5 * ((fh - mu_f[:, None]) / sd_f[:, None]) ** 2
                        + (alpha[:, None] - 1) * pt.log(h)[None, :]) + logmask
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

    def _option_table_grid(self, inputs, n_unique, S, shift_nu=None):
        """(S, K, R) option probabilities for every participant x unique magnitude.

        Computed per unique presented magnitude (K values), not per trial, and
        gathered afterwards: the motor-noise integral over the measurement grid is
        then an (S, K, M) x (S, M, R) batched matmul instead of a per-trial
        (T, M, R) tensor (which made the motor-noise fits ~50x slower).
        """
        if getattr(self, 'prior_leak', None):
            return self._option_table_prior_leak(inputs, n_unique, S)
        m, n_hat, alpha, nu_f = self.estimate_on_grid(inputs, S)
        nk = np.asarray(n_unique, dtype=float)
        if getattr(self, 'leak', None):
            # n_unique are (n, n_u) pairs: m centred on (1 - w) f(n) + w f(n_u)
            w = inputs['leak_w'][:, None]
            fn = ((1 - w) * self._f(pt.as_tensor_variable(nk[:, 0]), alpha)
                  + w * self._f(pt.as_tensor_variable(nk[:, 1]), alpha))  # (S, K)
        else:
            fn = self._f(pt.as_tensor_variable(nk), alpha)                  # (S, K)
        if shift_nu is not None:
            # measurement centres shifted by shift_nu (in units of nu_f): (S, K * Z)
            Z = shift_nu.shape[1]
            fn = (fn[:, :, None] + (nu_f[:, None] * shift_nu)[:, None, :]).reshape(
                (fn.shape[0], fn.shape[1] * Z))
        if not self.motor_noise:
            m_edges = self._invert(m, n_hat, self.response_edges)          # (S, R+1)
            z = (m_edges[:, None, :] - fn[:, :, None]) / nu_f[:, None, None]
            cdf = _norm_cdf(z)
            return cdf[:, :, 1:] - cdf[:, :, :-1]
        # motor noise: integrate the measurement density over the grid
        dens = pt.exp(-0.5 * ((m[:, None, :] - fn[:, :, None]) / nu_f[:, None, None]) ** 2)
        dens = dens / pt.sum(dens, axis=2, keepdims=True)                   # (S, K, M)
        edges = pt.as_tensor_variable(self.response_edges)
        sig = inputs['sigma_r'][:, None, None]
        cdf = _norm_cdf((edges[None, None, :] - n_hat[:, :, None]) / sig)   # (S, M, R+1)
        kern = cdf[:, :, 1:] - cdf[:, :, :-1]                               # (S, M, R)
        return pt.batched_dot(dens, kern)                                   # (S, K, R)

    def _option_table_prior_leak(self, inputs, n_unique, S):
        """(S, U * N, R) option probabilities with the prior centre shifted by n_u.

        ``n_unique`` is the full (n, n_u) grid from ``_pair_index`` (n_u-major). One
        pseudo-participant per (participant, n_u): prior centre mu + k (n_u - mu), its
        own estimate n_hat(m); the measurement is centred on f(n), or with ``leak`` on
        (1 - w) f(n) + w f(n_u).
        """
        nk = np.asarray(n_unique, dtype=float)
        nv, uv = np.unique(nk[:, 0]), np.unique(nk[:, 1])
        N, U = len(nv), len(uv)
        rows = {k: pt.repeat(v, U) for k, v in inputs.items()}               # row s * U + u
        u_row = pt.as_tensor_variable(np.tile(uv, S))
        mu = rows['prior_mu']
        rows['prior_mu'] = mu + rows['prior_leak_k'] * (u_row - mu)
        m, n_hat, alpha, nu_f = self.estimate_on_grid(rows, S * U)          # (SU, M)
        fn = self._f(pt.as_tensor_variable(nv), alpha)                      # (SU, N)
        if getattr(self, 'leak', None):
            pick = pt.as_tensor_variable(np.tile(np.eye(U), (S, 1)))        # (SU, U)
            fu = pt.sum(self._f(pt.as_tensor_variable(uv), alpha) * pick, axis=1)
            w = rows['leak_w'][:, None]
            fn = (1 - w) * fn + w * fu[:, None]
        if not self.motor_noise:
            m_edges = self._invert(m, n_hat, self.response_edges)          # (SU, R+1)
            cdf = _norm_cdf((m_edges[:, None, :] - fn[:, :, None]) / nu_f[:, None, None])
            p = cdf[:, :, 1:] - cdf[:, :, :-1]
        else:
            dens = pt.exp(-0.5 * ((m[:, None, :] - fn[:, :, None]) / nu_f[:, None, None]) ** 2)
            dens = dens / pt.sum(dens, axis=2, keepdims=True)               # (SU, N, M)
            edges = pt.as_tensor_variable(self.response_edges)
            sig = rows['sigma_r'][:, None, None]
            cdf = _norm_cdf((edges[None, None, :] - n_hat[:, :, None]) / sig)
            p = pt.batched_dot(dens, cdf[:, :, 1:] - cdf[:, :, :-1])        # (SU, N, R)
        return p.reshape((S, U * N, p.shape[-1]))

    def trial_probabilities(self, inputs, n, subject_ix, S, n_ix=None, n_unique=None,
                            shift_nu=None):
        """Per-trial option probabilities (T, R).

        Grid path: pass ``n_ix`` (index of each trial's magnitude into
        ``n_unique``); the table is computed per unique magnitude.
        """
        if self.closed_form:
            p = self._option_probs_closed(inputs, n, subject_ix)
        else:
            table = self._option_table_grid(inputs, n_unique, S, shift_nu)
            p = table[subject_ix, n_ix]
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

    def _condition_inputs(self, inputs, S):
        """Per (participant, condition) parameters, 2S long (row 2s + c).

        For each parameter in ``condition_parameters`` (on its untransformed scale):
        x[s, c] = x_s + (1/2 - c) * (delta + delta_s), delta_s ~ N(0, sd) non-centred,
        so ``{p}_cond_mu`` = delta is level 0 minus level 1 (condition_levels).
        """
        model = pm.Model.get_context()
        sign = pt.as_tensor_variable(np.array([.5, -.5]))
        out = {}
        for k, v in inputs.items():
            if k not in self.condition_parameters:
                out[k] = pt.repeat(v, 2)
                continue
            spec = self.free_parameters[k]
            tr = spec.get('transform', 'identity')
            if tr not in ('identity', 'softplus'):
                raise NotImplementedError(f'condition offsets for {tr} parameters')
            base = v if tr == 'identity' else model[f'{k}_untransformed']
            scale = spec.get('sigma_intercept', 1.0) / 2
            d_mu = pm.Normal(f'{k}_cond_mu', 0.0, scale)
            d_sd = pm.HalfNormal(f'{k}_cond_sd', spec.get('cauchy_sigma_intercept', .5) / 2)
            d_off = pm.Normal(f'{k}_cond_offset', 0.0, 1.0, dims=('subject',))
            delta = d_mu + d_sd * d_off                                     # (S,)
            x = base[:, None] + sign[None, :] * delta[:, None]               # (S, 2)
            x = x if tr == 'identity' else pt.softplus(x)
            pm.Deterministic(f'{k}_by_condition', x, dims=('subject', 'condition'))
            out[k] = x.reshape((2 * S,))
        return out

    def build_likelihood(self, parameters, save_p_choice=False):
        model = pm.Model.get_context()
        inputs = self.get_model_inputs(parameters)
        S = len(model.coords['subject']) if 'subject' in model.coords else 1
        sub_ix = model['subject_ix']
        n_ix = model['n_ix']
        shift_nu = None
        if (self.leak or self.prior_leak) and (self.measurement_shift or self.noise_modulator):
            raise NotImplementedError('leak with trial covariates')
        if self.condition is not None:
            if self.measurement_shift or self.noise_modulator:
                raise NotImplementedError('condition offsets with trial covariates')
            model.add_coord('condition', [str(x) for x in self.condition_levels])
            inputs = self._condition_inputs(inputs, S)
            sub_ix = 2 * sub_ix + model['cond_ix']
            S = 2 * S
        if self.noise_modulator:
            # nu_t = nu * exp(gamma * u): one pseudo-participant per (participant, u bin);
            # the observer decodes with its own trial noise
            U = self.n_covariate_bins
            uc = pt.as_tensor_variable(self.noise_centres)                 # (S, U)
            scale = pt.exp(inputs['noise_gamma'][:, None] * uc).reshape((S * U,))
            inputs = {k: pt.repeat(v, U) for k, v in inputs.items()}
            inputs['nu'] = inputs['nu'] * scale
            sub_ix = U * sub_ix + model['noise_bin']
            shift_rows = pt.repeat(inputs['shift_beta'][::U], U) if self.measurement_shift else None
            zc_rows = pt.repeat(pt.as_tensor_variable(self.shift_centres), U, axis=0) \
                if self.measurement_shift else None
            S = S * U
        elif self.measurement_shift:
            shift_rows = inputs['shift_beta']
            zc_rows = pt.as_tensor_variable(self.shift_centres)
        if self.measurement_shift:
            # m ~ N(f(n) + beta * nu_f * z, nu_f^2): one stimulus position per (n, z bin)
            shift_nu = shift_rows[:, None] * zc_rows                            # (S, Z)
            n_ix = n_ix * self.n_covariate_bins + model['shift_bin']
        p = self.trial_probabilities(inputs, model['n'], sub_ix, S, n_ix=n_ix,
                                     n_unique=self.n_unique, shift_nu=shift_nu)
        if save_p_choice:
            p = pm.Deterministic('p_response', p)
        if not getattr(self, 'censored', False):
            pm.Categorical('response', p=p, observed=model['response_ix'])
            return
        # Censored responses: censor = +1 means the intended report is at least the
        # final position (still moving up when time ran out), -1 at most, 0 observed.
        R = len(self.response_options)
        pm.CustomDist('response', p, model['censor'], logp=_censored_logp(R),
                      random=_censored_random, signature='(r),()->()', dtype='int64',
                      observed=model['response_ix'])

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
        n = paradigm['n'].values.astype(float)
        n_unique, n_ix = np.unique(n, return_inverse=True)
        if getattr(self, 'leak', None) or getattr(self, 'prior_leak', None):
            n_unique, n_ix = self._pair_index(n, paradigm[self.leak or self.prior_leak].values)
        p = self.trial_probabilities(inputs, pt.as_tensor_variable(n), pt.as_tensor_variable(codes),
                                     len(uniques), n_ix=pt.as_tensor_variable(n_ix),
                                     n_unique=n_unique).eval()
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
        arr = pp.posterior_predictive['response'].values[0].astype(int)
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
