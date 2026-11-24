"""
Magnitude -> value estimation: a number (or numerosity) is learned to be worth a value through a curved mapping G,
and participants report that value (e.g. a BDM bid). Which processing stages carry noise, and which prior do they use?

Generative model, per trial with stimulus x on [lo, hi] and mapping G (concave / convex / linear):

    1. Number stage (``perception=True``): x is encoded efficiently under the number prior p(x),
       c = F(x) + noise, F the CDF of p(x) on [0, 1], noise SD ``sigma_per``; decoded as the posterior mean x_hat.
       Without it, x_hat = x.
    2. Value stage (``valuation=True``): the value G(x_hat) is encoded efficiently under the value prior p(v),
       m = F_v(G(x_hat)) + noise, SD ``sigma_rep``; decoded as the posterior mean of v given m. Under the objective
       value prior this is a Jensen pull toward the diagonal on a curved G.
       Without it, v_hat = G(x_hat).
    3. Response: v_hat + Gaussian response noise ``sigma_motor``, truncated to [lo, hi]; with probability ``rho``
       rounded to a multiple of ``round_step`` (values below the lowest multiple in the range round up to it),
       otherwise to ``response_bin_width``; a fixed uniform lapse.

All noise SDs are in % of the code range (order 1 to 10), so priors do not depend on the stimulus range. With both
stages (``share_parameterisation``) the sampler sees the total noise T and the number stage's share s of its
variance, sigma_per = T sqrt(s), sigma_rep = T sqrt(1 - s): the two SDs trade off along a ridge, which made the
analogous orientation fits saturate the tree depth (abstract_values, 2026-10).

Priors have few parameters, so that the data can pin them down:

* number code: Stevens' power law, c = f(x) rescaled to [0, 1] with f(x) = (x^beta - 1) / beta (Box-Cox, as
  :class:`bauer.models.estimation_task.PowerSpace`): beta = 1 is a linear code (constant noise along the number line),
  beta -> 0 a log code (Weber). ``number_encoding`` = 'linear', 'log' or 'power' (beta fitted). Decoding uses
  ``number_decoding_prior`` = 'efficient' (the prior this code is efficient for, p(x) ~ x^(beta - 1)) or 'objective'
  (uniform, as the numbers are drawn). x is shifted by max(0, 1 - lo) so that x = 0 stays finite.
* value prior: the objective prior (uniform numbers pushed through the true mapping, which piles values up where the
  curve is flat), mixed with a uniform prior over values with weight w. ``value_prior`` = 'objective' (w = 0),
  'uniform' (w = 1) or 'mixture' (w fitted).

Each stage codes efficiently for its own prior (the code is the prior's CDF) and decodes with it.

``formats``: several stimulus formats (e.g. ('numeral', 'dots')) in one fit, named in a 'format' column. The number
stage gets its own noise and code exponent per format (``sigma_per_<format>``, ``beta_<format>``); the value stage,
the value prior, the mapping shrinkage, response noise and rounding are shared, because they act on value, not on the
stimulus. A format with a well-identified split (dots: large, compressive number noise) then pins the value-stage
noise, and with it the split of a format that cannot pin it alone (numerals). Uses separate stage SDs (no
``share_parameterisation``).

``fit_mapping_shrinkage``: the internal mapping is (1 - lambda) G + lambda * identity, an imperfectly learned curve
pulled toward the diagonal (as ``mapping_shrinkage`` in :mod:`bauer.efficient_coding`).

The likelihood is a grid computation through ``pm.Potential``; per-trial log-likelihoods for PSIS-LOO come from
:meth:`MagnitudeValueModel.pointwise_log_likelihood`.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .efficient_coding import value_noise_kernel
from .estimation import EstimationBaseModel


# Noise SDs in % of the stimulus range. softplus(N(mu, sd)); the group-SD scale follows the parameter's own prior SD
# (1.5x), as for kappa_r in efficient_coding: bauer's 0.25 default would pool every participant completely.
SIGMA_STAGE_PRIOR = {'mu_intercept': 3.0, 'sigma_intercept': 3.0, 'cauchy_sigma_intercept': 4.5,
                     'transform': 'softplus'}
SIGMA_MOTOR_PRIOR = {'mu_intercept': 1.0, 'sigma_intercept': 2.0, 'cauchy_sigma_intercept': 3.0,
                     'transform': 'softplus'}
MAPPING_SHRINKAGE_PRIOR = {'mu_intercept': -2.0, 'sigma_intercept': 1.5, 'cauchy_sigma_intercept': 1.5,
                           'transform': 'logistic'}
ROUNDING_PRIOR = {'mu_intercept': -1.0, 'sigma_intercept': 1.5, 'cauchy_sigma_intercept': 1.5,
                  'transform': 'logistic'}


# Stevens exponent of the number code, as bauer's PowerLawEncoding models: softplus(N(0.5, 0.5)), mostly 0.2-1.3
BETA_PRIOR = {'mu_intercept': 0.5, 'sigma_intercept': 0.5, 'cauchy_sigma_intercept': 0.75, 'transform': 'softplus'}
# Weight of the uniform component in the value prior: mostly objective a priori
VALUE_MIX_PRIOR = {'mu_intercept': -1.5, 'sigma_intercept': 1.5, 'cauchy_sigma_intercept': 1.5,
                   'transform': 'logistic'}
TOTAL_NOISE_PRIOR = {'mu_intercept': 4.0, 'sigma_intercept': 3.0, 'cauchy_sigma_intercept': 4.5,
                     'transform': 'softplus'}
SHARE_PRIOR = {'mu_intercept': 0.0, 'sigma_intercept': 1.0, 'cauchy_sigma_intercept': 1.0, 'transform': 'logistic'}
NUMBER_ENCODINGS = ('linear', 'log', 'power')
NUMBER_DECODING_PRIORS = ('efficient', 'objective')
VALUE_PRIORS = ('objective', 'uniform', 'mixture')


def mapping_curve_np(u, mapping, curvature):
    """[0, 1] -> [0, 1]. Linear; concave (1 - exp(-k u)) / (1 - exp(-k)); convex is concave mirrored in both axes."""
    u = np.asarray(u, dtype=float)
    concave = lambda x: (1 - np.exp(-curvature * x)) / (1 - np.exp(-curvature))
    return {'linear': u, 'concave': concave(u), 'convex': 1 - concave(1 - u)}[mapping]


def objective_value_prior_np(v, mapping, curvature, n_fine=20001):
    """Density on [0, 1] of G(U), U uniform: 1 / G'(G^-1(v)), evaluated at v."""
    u = np.linspace(0, 1, n_fine)
    g = mapping_curve_np(u, mapping, curvature)
    dens = 1 / np.gradient(g, u)
    p = np.interp(v, g, dens)
    return p / np.trapezoid(p, v)


def tent(y, grid, d):
    """Linear-interpolation weights of values y (..., M) onto an equally spaced grid (G,): (..., M, G).
    Each row sums to 1 for y inside the grid."""
    y = pt.clip(y, grid[0], grid[-1])
    return pt.maximum(0., 1. - pt.abs(y[..., None] - grid) / d)


class MagnitudeValueModel(EstimationBaseModel):
    """Number/numerosity -> value estimation with optional number and value stages; see the module docstring.

    Parameters
    ----------
    paradigm : DataFrame with 'number', 'mapping' and 'response' (and 'subject' as index level for hierarchical fits).
    perception, valuation : bool
        Which noisy, Bayesian-decoded stages are present. Both False: exact value plus response noise.
    number_encoding : 'linear', 'log' or 'power'
    number_decoding_prior : 'efficient' or 'objective'
    value_prior : 'objective', 'uniform' or 'mixture'
    share_parameterisation : bool
        With both stages, sample total noise and the number stage's share (default) instead of the two SDs.
    stimulus_range : (lo, hi)
        Range of numbers and values (the mapping maps it onto itself).
    curvature : float
        k of the concave/convex mapping.
    response_bin_width, round_step : float
        Answers are recorded to ``response_bin_width``; rounding to multiples of ``round_step`` has probability rho.
    grid_resolution : int
        Points on the number, code and value grids. The grid step, 100 / (grid_resolution - 1) % of the range, is
        the resolution ceiling of every noise SD (the cell kernel makes the limit smooth, not a cliff).
    """

    # Default inits jitter the untransformed parameters around 0; on the hierarchical grid model that left 2 of 4
    # chains frozen (step size ~1e-17, a subject's noise SD near 0) in every design-recovery run. MAP + jitter is the fix.
    recommended_init = 'mapjitter'
    paradigm_keys = ['number']
    warn_on_input_scale = False  # noise SDs are in % of range by construction

    def __init__(self, paradigm=None, perception=True, valuation=True, number_encoding='linear',
                 number_decoding_prior='efficient', value_prior='objective', share_parameterisation=True, stimulus_range=(1., 50.), curvature=2.79,
                 response_bin_width=1.0, round_step=5.0, fit_rounding=True, fit_mapping_shrinkage=False,
                 grid_resolution=301, lapse_rate=0.01, formats=None, **kwargs):
        if number_encoding not in NUMBER_ENCODINGS:
            raise ValueError(f'number_encoding must be one of {NUMBER_ENCODINGS}')
        if number_decoding_prior not in NUMBER_DECODING_PRIORS:
            raise ValueError(f'number_decoding_prior must be one of {NUMBER_DECODING_PRIORS}')
        if value_prior not in VALUE_PRIORS:
            raise ValueError(f'value_prior must be one of {VALUE_PRIORS}')
        self.perception, self.valuation = bool(perception), bool(valuation)
        self.number_encoding, self.number_decoding_prior = number_encoding, number_decoding_prior
        self.value_prior = value_prior
        self.formats = tuple(formats) if formats else None
        self.share_parameterisation = (bool(share_parameterisation) and self.perception and self.valuation
                                       and self.formats is None)
        self.lo, self.hi = map(float, stimulus_range)
        self.offset = max(0., 1. - self.lo)
        self.curvature = float(curvature)
        self.round_step = float(round_step)
        self.fit_rounding = bool(fit_rounding)
        self.fit_mapping_shrinkage = bool(fit_mapping_shrinkage)
        super().__init__(paradigm, grid_resolution=grid_resolution, response_bin_width=response_bin_width,
                         lapse_rate=lapse_rate, **kwargs)

    # ---- parameters ----

    def get_free_parameters(self):
        pars = {}
        if self.share_parameterisation:
            pars.update(sigma_total=TOTAL_NOISE_PRIOR, share_per=SHARE_PRIOR)
        else:
            if self.perception:
                pars.update({n: SIGMA_STAGE_PRIOR for n in self._per_format('sigma_per')})
            if self.valuation:
                pars['sigma_rep'] = SIGMA_STAGE_PRIOR
        pars['sigma_motor'] = SIGMA_MOTOR_PRIOR
        if self.fit_rounding:
            pars['rho'] = ROUNDING_PRIOR
        if self.fit_mapping_shrinkage:
            pars['mapping_shrinkage'] = MAPPING_SHRINKAGE_PRIOR
        if self.perception and self.number_encoding == 'power':
            pars.update({n: BETA_PRIOR for n in self._per_format('beta')})
        if self.valuation and self.value_prior == 'mixture':
            pars['value_prior_uniform'] = VALUE_MIX_PRIOR
        return pars

    def _per_format(self, name):
        """Parameter names of a number-stage parameter: one, or one per format."""
        return [name] if self.formats is None else [f'{name}_{f}' for f in self.formats]

    def get_model_inputs(self, parameters):
        model = pm.Model.get_context()
        inputs = {name: self.subjectwise(name) for name in self.free_parameters}
        inputs['response'] = model['response']
        inputs['subject_ix'] = model['subject_ix']
        return inputs

    # ---- grids and paradigm ----

    def _setup_grids(self, paradigm):
        N = self.grid_resolution
        self.u_grid = np.linspace(0., 1., N)                    # number and value, as a fraction of the range
        self.du = self.u_grid[1] - self.u_grid[0]
        self.x_grid = self.lo + (self.hi - self.lo) * self.u_grid
        self.response_grid = self.x_grid
        self.mappings = sorted(paradigm['mapping'].astype(str).unique())
        self.G = np.stack([mapping_curve_np(self.u_grid, m, self.curvature) for m in self.mappings])  # (C, N)
        self.p_value_objective = np.stack([objective_value_prior_np(self.u_grid, m, self.curvature)
                                           for m in self.mappings])                                   # (C, N)

    def build_estimation_model(self, data=None, coords=None, hierarchical=True, save_p_choice=False,
                               flat_prior=False, paradigm=None):
        if data is None:
            data = paradigm if paradigm is not None else self.paradigm
        self._setup_grids(data)
        super().build_estimation_model(data, coords, hierarchical, save_p_choice, flat_prior)

    def _get_paradigm(self, paradigm=None, subject_mapping=None):
        paradigm_ = super()._get_paradigm(paradigm, subject_mapping)
        u = (paradigm_['number'].astype(float) - self.lo) / (self.hi - self.lo)
        # The stimulus lies between two grid points: its answer distribution is interpolated linearly between theirs
        # (snapping to the nearest point shifts a numeral by up to du/2, ~0.08 of 1-50, i.e. ~0.25 CHF where the
        # mapping is steep)
        f = np.clip(u / self.du, 0, self.grid_resolution - 1)
        lo_ix = np.clip(np.floor(f), 0, self.grid_resolution - 2).astype(int)
        paradigm_['stimulus_ix'] = lo_ix
        paradigm_['stimulus_w'] = (f - lo_ix).astype(float)
        self._n_sub = int(np.max(paradigm_['subject_ix'])) + 1
        paradigm_['mapping_ix'] = np.array([self.mappings.index(str(m)) for m in paradigm['mapping'].astype(str)],
                                           dtype=int)
        if self.formats is None:
            paradigm_['format_ix'] = np.zeros(len(paradigm), dtype=int)
        else:
            unknown = set(paradigm['format'].astype(str)) - set(self.formats)
            if unknown:
                raise ValueError(f'formats {unknown} not in {self.formats}')
            paradigm_['format_ix'] = np.array([self.formats.index(str(f)) for f in paradigm['format'].astype(str)],
                                              dtype=int)
        k = paradigm_['response'] / self.round_step
        paradigm_['is_round'] = (np.abs(k - np.round(k)) < 1e-6).astype(float)
        return paradigm_

    def _get_response_grid(self):
        return self.response_grid

    # ---- the model ----

    @staticmethod
    def _cdf(p, du):
        """Normalised CDF on [0, 1] of densities p (..., N) on an equally spaced grid."""
        cum = pt.cumsum((p[..., 1:] + p[..., :-1]) / 2 * du, axis=-1)
        cum = pt.concatenate([pt.zeros_like(p[..., :1]), cum], axis=-1)
        return cum / cum[..., -1:]

    def _kernel(self, F, sigma):
        """p(code cell | F): F (..., N) code positions, sigma broadcastable to F[..., None], in % of range.
        Returns (..., N, N_code), rows sum to 1."""
        code = pt.as_tensor_variable(self.u_grid)
        k = value_noise_kernel(code, F[..., None], sigma / 100., self.du, 'cell')
        return k / (pt.sum(k, axis=-1, keepdims=True) + 1e-30)

    def _stage_sds(self, inputs, n_sub):
        shape = lambda name: pt.specify_shape(inputs[name], (n_sub,))
        if self.share_parameterisation:
            total, share = shape('sigma_total'), shape('share_per')
            return [total * pt.sqrt(share)], total * pt.sqrt(1 - share)
        per = [shape(n) for n in self._per_format('sigma_per')] if self.perception else None
        return per, (shape('sigma_rep') if self.valuation else None)

    def _response_masses(self, inputs):
        """(S, F, C, N_number, N_response) probability masses of the answer (before rounding and lapse), F formats."""
        n_sub = self._n_sub  # from the paradigm: prediction models built without coords need it too
        u = pt.as_tensor_variable(self.u_grid)
        sigma_per, sigma_rep = self._stage_sds(inputs, n_sub)
        G = pt.as_tensor_variable(self.G)[None]                                        # (1, C, N)
        if self.fit_mapping_shrinkage:
            lam = pt.specify_shape(inputs['mapping_shrinkage'], (n_sub,))[:, None, None]
            G = (1 - lam) * G + lam * u[None, None, :]
        else:
            G = pt.tile(G, (n_sub, 1, 1))                                               # (S, C, N)

        # Value stage (or the exact mapping): from the number that reaches it to the value, (S, C, N, N_value)
        if self.valuation:
            p_v = pt.as_tensor_variable(self.p_value_objective)[None]                   # (1, C, N)
            if self.value_prior == 'uniform':
                p_v = pt.ones_like(p_v)
            elif self.value_prior == 'mixture':
                w = pt.specify_shape(inputs['value_prior_uniform'], (n_sub,))[:, None, None]
                p_v = (1 - w) * p_v + w
            p_v = p_v * pt.ones((n_sub, 1, 1))                                          # (S, C, N)
            F_v = self._cdf(p_v, self.du)                                               # (S, C, N)
            enc = pt.einsum('scnv,scv->scn', tent(G, u, self.du), F_v)                  # F_v(G(x)), (S, C, N)
            sr = sigma_rep[:, None, None, None]
            K_enc = self._kernel(enc, sr)                                               # (S, C, N, M)
            K_dec = self._kernel(F_v, sr) * p_v[..., None]                              # (S, C, V, M)
            v_hat = pt.sum(K_dec * u[None, None, :, None], axis=2) / (pt.sum(K_dec, axis=2) + 1e-30)  # (S, C, M)
            to_value = pt.einsum('scnm,scmv->scnv', K_enc, tent(v_hat, u, self.du))
        else:
            to_value = tent(G, u, self.du)                                              # (S, C, N, N_value)

        # Number stage, per format: from the presented number to the decoded one, (S, N, N)
        n_formats = 1 if self.formats is None else len(self.formats)
        if self.perception:
            logx = np.log(self.x_grid + self.offset)[None, :]
            per_format = []
            for f, beta_name in enumerate(self._per_format('beta')):
                if self.number_encoding == 'linear':
                    beta = pt.ones((n_sub, 1))
                elif self.number_encoding == 'log':
                    beta = pt.zeros((n_sub, 1))
                else:
                    beta = pt.specify_shape(inputs[beta_name], (n_sub,))[:, None]
                b = pt.switch(pt.lt(pt.abs(beta), 1e-4), 1e-4, beta)
                fx = pt.expm1(b * logx) / b                                             # Box-Cox, (S, N)
                F_x = (fx - fx[:, :1]) / (fx[:, -1:] - fx[:, :1])
                if self.number_decoding_prior == 'efficient':
                    p_x = pt.exp((beta - 1) * logx)                                     # dF/dx
                else:
                    p_x = pt.ones((n_sub, self.grid_resolution))
                K = self._kernel(F_x, sigma_per[f][:, None, None])                      # (S, N, M)
                post = K * p_x[:, :, None]
                x_hat = pt.sum(post * u[None, :, None], axis=1) / (pt.sum(post, axis=1) + 1e-30)  # (S, M)
                to_number = pt.einsum('snm,smk->snk', K, tent(x_hat, u, self.du))
                per_format.append(pt.einsum('snk,sckv->scnv', to_number, to_value))
            to_value = pt.stack(per_format, axis=1)                                     # (S, F, C, N, V)
        else:
            to_value = pt.stack([to_value] * n_formats, axis=1)

        # Response noise, truncated to the range
        sm = pt.specify_shape(inputs['sigma_motor'], (n_sub,))
        R = self._kernel(pt.tile(u[None, :], (n_sub, 1)), sm[:, None, None])            # (S, N_value, N_response)
        return pt.einsum('sfcnv,svr->sfcnr', to_value, R)

    @staticmethod
    def _trial_masses(masses, subject_ix):
        """Answer masses (T, N) of every trial: linear interpolation between the two grid points around its stimulus."""
        model = pm.Model.get_context()
        ix = (subject_ix, model['format_ix'], model['mapping_ix'])
        w = model['stimulus_w'][:, None]
        return (1 - w) * masses[ix + (model['stimulus_ix'],)] + w * masses[ix + (model['stimulus_ix'] + 1,)]

    def _trial_log_likelihood(self, inputs):
        model = pm.Model.get_context()
        masses = self._trial_masses(self._response_masses(inputs), inputs['subject_ix'])                   # (T, N)
        # CDF at the cell edges: grid point i owns [x_i - d/2, x_i + d/2]
        d = self.x_grid[1] - self.x_grid[0]
        cdf = pt.concatenate([pt.zeros((masses.shape[0], 1)), pt.cumsum(masses, axis=1)], axis=1)  # (T, N + 1)
        edges0 = self.x_grid[0] - d / 2
        n_edges = self.grid_resolution + 1
        grid_ix = pt.arange(n_edges)

        def cdf_at(x):
            f = pt.clip((x - edges0) / d, 0, n_edges - 1)
            i = pt.clip(pt.cast(pt.floor(f), 'int64'), 0, n_edges - 2)
            w = f - i
            lo_ = pt.sum(cdf * pt.eq(grid_ix[None, :], i[:, None]), axis=1)
            hi_ = pt.sum(cdf * pt.eq(grid_ix[None, :], (i + 1)[:, None]), axis=1)
            return lo_ + w * (hi_ - lo_)

        r = inputs['response']
        p_exact = cdf_at(r + self.response_bin_width / 2) - cdf_at(r - self.response_bin_width / 2)
        if self.fit_rounding:
            rho = inputs['rho'][inputs['subject_ix']]
            # A rounded answer r owns [r - step/2, r + step/2]; the lowest multiple in the range also owns everything
            # below it (a value of 2 rounded to the nearest 5 is answered as 5, not 0)
            lowest = self.round_step * np.ceil(self.lo / self.round_step)
            lower = pt.switch(pt.le(r, lowest + 1e-6), self.lo - d, r - self.round_step / 2)
            p_round = (cdf_at(r + self.round_step / 2) - cdf_at(lower)) * model['is_round']
            p_answer = (1 - rho) * p_exact + rho * p_round
        else:
            p_answer = p_exact
        p_answer = (1 - self.lapse_rate) * p_answer + self.lapse_rate * self.response_bin_width / (self.hi - self.lo + d)
        return pt.log(pt.clip(p_answer, 1e-300, 1.))

    def build_likelihood(self, parameters, save_p_choice=False):
        inputs = self.get_model_inputs(parameters)
        pm.Potential('ll', self._trial_log_likelihood(inputs).sum())

    def _compute_trial_distributions(self, model_inputs):
        """Per-trial density on the response grid (rounding and lapse not included), for predict()."""
        model = pm.Model.get_context()
        d = self.x_grid[1] - self.x_grid[0]
        return self._trial_masses(self._response_masses(model_inputs), model_inputs['subject_ix']) / d

    # ---- after fitting ----

    def pointwise_log_likelihood(self, data, idata, max_draws=None):
        """Per-trial log-likelihood for every posterior draw, as an InferenceData ``log_likelihood`` group
        (dims chain, draw, trial), ready for ``az.loo``. ``data`` must be the data the model was fitted on."""
        import arviz as az
        import pytensor
        post = idata.posterior
        names = list(self.free_parameters)
        draws = post.sizes['draw'] if max_draws is None else min(max_draws, post.sizes['draw'])
        hierarchical = 'subject' in post[names[0]].dims
        coords = {'subject': data.index.unique(level='subject')} if hierarchical else None
        self._setup_grids(data)
        with pm.Model(coords=coords) as m:
            self.set_paradigm(self._get_paradigm(paradigm=data))
            for n in names:
                v = post[n].isel(chain=0, draw=0).values
                pm.Data(n, np.atleast_1d(v) if hierarchical else v)
            ll = self._trial_log_likelihood(self.get_model_inputs(None))
        fn = pytensor.function([], ll)
        out = np.empty((post.sizes['chain'], draws, len(data)))
        for c in range(post.sizes['chain']):
            for i in range(draws):
                for n in names:
                    v = post[n].isel(chain=c, draw=i).values
                    m[n].set_value(np.atleast_1d(v) if hierarchical else v)
                out[c, i] = fn()
        return az.from_dict(log_likelihood={'response': out}, dims={'response': ['trial']},
                            coords={'trial': np.arange(len(data))})


# ---- likelihood relative to the truth: per-trial quadrature instead of global response grids ----

def _phi_cdf(z):
    return 0.5 * pt.erfc(-z / np.sqrt(2.0))


def _trunc_cdf(c, mu, s):
    """CDF at c of N(mu, s) truncated to [0, 1] (the code range)."""
    lo, hi = _phi_cdf((0. - mu) / s), _phi_cdf((1. - mu) / s)
    return pt.clip((_phi_cdf((c - mu) / s) - lo) / pt.maximum(hi - lo, 1e-30), 0., 1.)


def _interp_uniform(table, prefix, x, n):
    """Linear interpolation of tables (..., n) on the uniform grid [0, 1] at x, rows picked by the index arrays in
    prefix (broadcast against x)."""
    f = pt.clip(x, 0., 1.) * (n - 1)
    i = pt.clip(pt.floor(f), 0, n - 2).astype('int64')
    w = f - i
    return (1 - w) * table[prefix + (i,)] + w * table[prefix + (i + 1,)]


def _inverse_table(f, n):
    """Inverse of non-decreasing tables f (..., n) on the uniform grid [0, 1], tabulated on the same grid: the code c
    with f(c) = y for y = 0, 1/(n-1), ..., 1 (0 below the range of f, 1 above it)."""
    y = pt.as_tensor_variable(np.linspace(0., 1., n))
    cnt = pt.sum(pt.le(f[..., None, :], y[:, None]), axis=-1)                 # (..., n_y)
    j = pt.clip(cnt - 1, 0, n - 2).astype('int64')
    f0, f1 = pt.take_along_axis(f, j, axis=-1), pt.take_along_axis(f, j + 1, axis=-1)
    frac = pt.clip((y - f0) / pt.maximum(f1 - f0, 1e-12), 0., 1.)
    return (j + frac) / (n - 1)


def _cell_mean_window(y0, y1, a0, a1, s):
    """Mean over y in [y0, y1] of P(y + N(0, s) in [a0, a1]), exact (I(z) = z Phi(z) + phi(z) is the integral of
    Phi); the midpoint value when the cell is much narrower than s."""
    I = lambda z: z * _phi_cdf(z) + pt.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    w = y1 - y0
    exact = s / pt.maximum(w, 1e-12) * ((I((a1 - y0) / s) - I((a1 - y1) / s)) - (I((a0 - y0) / s) - I((a0 - y1) / s)))
    ym = (y0 + y1) / 2
    mid = _phi_cdf((a1 - ym) / s) - _phi_cdf((a0 - ym) / s)
    return pt.switch(pt.lt(w, 1e-3 * s), mid, exact)


class MagnitudeValueLocalModel(MagnitudeValueModel):
    """:class:`MagnitudeValueModel` with a likelihood computed relative to each trial's truth and response, instead of
    on global grids over the whole range.

    The stages only use global grids for 1-D tables (the codes F, the decoders x_hat(c) and v_hat(c), the mapping), at
    ``grid_resolution`` points. Per trial:

    * the number-stage noise is integrated by quadrature over ``n_nodes`` cells of the truncated Gaussian around the
      trial's own code F(x) (cell weights are CDF differences, so a vanishing SD is the exact deterministic limit);
    * the LAST noisy code stage is handled analytically: the decoder is monotone, so the CDF of the decoded estimate is
      the code noise's CDF at the inverse decoder, P(v_hat <= y) = Phi((v_hat^-1(y) - mu) / sigma);
    * response noise is integrated exactly over ``n_cells`` cells of a window around the RESPONSE's own bin
      (bin +- 5 SD of the response noise): the mean of Phi over each cell has a closed form.

    So the resolution follows the noise and the response instead of the range: a numeral answered to within 0.1 CHF
    is resolved as finely as a dot cloud answered 10 CHF off, and the likelihood stays smooth in every SD.
    Prediction on a response grid (``predict``) is not available; fit and score with this class.
    """

    def __init__(self, *args, grid_resolution=601, n_nodes=48, n_cells=40, answer_precision_bins=False,
                 group_sd_factor=1.0, **kwargs):
        self.n_nodes, self.n_cells = int(n_nodes), int(n_cells)
        # Score each answer in the bin of the precision it was typed with (whole number: 1, x.5: 0.5, anything finer:
        # 0.01), instead of one response_bin_width for everyone: a numeral typed as '37' was rounded, '36.85' was not.
        # The bins are the same for every model, so comparisons between models stay fair.
        self.answer_precision_bins = bool(answer_precision_bins)
        # Multiplies the scale of every group-SD prior (hierarchical fits): < 1 tames the funnel
        self.group_sd_factor = float(group_sd_factor)
        super().__init__(*args, grid_resolution=grid_resolution, **kwargs)

    def get_free_parameters(self):
        pars = super().get_free_parameters()
        if self.group_sd_factor != 1.0:
            pars = {k: dict(v, cauchy_sigma_intercept=v['cauchy_sigma_intercept'] * self.group_sd_factor)
                    for k, v in pars.items()}
        return pars

    def _get_paradigm(self, paradigm=None, subject_mapping=None):
        paradigm_ = super()._get_paradigm(paradigm, subject_mapping)
        r = np.asarray(paradigm_['response'], dtype=float)
        if self.answer_precision_bins:
            whole = np.abs(r - np.round(r)) < 1e-6
            half = np.abs(2 * r - np.round(2 * r)) < 1e-6
            paradigm_['bin_width'] = np.where(whole, 1.0, np.where(half, 0.5, 0.01))
        else:
            paradigm_['bin_width'] = np.full(len(r), float(self.response_bin_width))
        return paradigm_

    def _tables(self, inputs):
        """1-D tables on the uniform grid: F_x (S, F, N), x_hat (S, F, N), G (S, C, N), F_v (S, C, N), v_hat (S, C, N)."""
        n_sub, N = self._n_sub, self.grid_resolution
        u = pt.as_tensor_variable(self.u_grid)
        sigma_per, sigma_rep = self._stage_sds(inputs, n_sub)
        G = pt.as_tensor_variable(self.G)[None]
        if self.fit_mapping_shrinkage:
            lam = pt.specify_shape(inputs['mapping_shrinkage'], (n_sub,))[:, None, None]
            G = (1 - lam) * G + lam * u[None, None, :]
        else:
            G = pt.tile(G, (n_sub, 1, 1))
        out = dict(G=G, sigma_per=sigma_per, sigma_rep=sigma_rep)
        if self.valuation:
            p_v = pt.as_tensor_variable(self.p_value_objective)[None]
            if self.value_prior == 'uniform':
                p_v = pt.ones_like(p_v)
            elif self.value_prior == 'mixture':
                w = pt.specify_shape(inputs['value_prior_uniform'], (n_sub,))[:, None, None]
                p_v = (1 - w) * p_v + w
            p_v = p_v * pt.ones((n_sub, 1, 1))
            F_v = self._cdf(p_v, self.du)                                                   # (S, C, N)
            K_dec = self._kernel(F_v, sigma_rep[:, None, None, None]) * p_v[..., None]    # (S, C, V, M)
            out['F_v'] = F_v
            out['v_hat'] = pt.sum(K_dec * u[None, None, :, None], axis=2) / (pt.sum(K_dec, axis=2) + 1e-30)
        if self.perception:
            logx = np.log(self.x_grid + self.offset)[None, :]
            F_xs, x_hats = [], []
            for f, beta_name in enumerate(self._per_format('beta')):
                if self.number_encoding == 'linear':
                    beta = pt.ones((n_sub, 1))
                elif self.number_encoding == 'log':
                    beta = pt.zeros((n_sub, 1))
                else:
                    beta = pt.specify_shape(inputs[beta_name], (n_sub,))[:, None]
                b = pt.switch(pt.lt(pt.abs(beta), 1e-4), 1e-4, beta)
                fx = pt.expm1(b * logx) / b
                F_x = (fx - fx[:, :1]) / (fx[:, -1:] - fx[:, :1])
                p_x = pt.exp((beta - 1) * logx) if self.number_decoding_prior == 'efficient' \
                    else pt.ones((n_sub, N))
                post = self._kernel(F_x, sigma_per[f][:, None, None]) * p_x[:, :, None]   # (S, N, M)
                F_xs.append(F_x)
                x_hats.append(pt.sum(post * u[None, :, None], axis=1) / (pt.sum(post, axis=1) + 1e-30))
            out['F_x'], out['x_hat'] = pt.stack(F_xs, axis=1), pt.stack(x_hats, axis=1)  # (S, F, N)
        return out

    def _window_probability(self, H_at, a0, a1, s_m):
        """P(answer in [a0, a1]) (T,), with v_hat's CDF H_at(edges (T, L+1)) -> (T, K, L+1) and response noise s_m (T,);
        everything in [0, 1] units of the range. Window: [a0 - 5 s_m, a1 + 5 s_m] in n_cells cells, within [0, 1]."""
        L = self.n_cells
        t = pt.as_tensor_variable(np.linspace(0., 1., L + 1))
        w0, w1 = pt.clip(a0 - 5 * s_m, 0., 1.), pt.clip(a1 + 5 * s_m, 0., 1.)
        edges = w0[:, None] + (w1 - w0)[:, None] * t[None, :]                            # (T, L+1)
        H = H_at(edges)                                                                  # (T, K, L+1)
        dH = H[..., 1:] - H[..., :-1]                                                    # (T, K, L)
        y0, y1 = edges[:, :-1], edges[:, 1:]
        sm = s_m[:, None]
        psi = _cell_mean_window(y0, y1, a0[:, None], a1[:, None], sm)                     # (T, L)
        ym = (y0 + y1) / 2
        z_trunc = _phi_cdf((1. - ym) / sm) - _phi_cdf((0. - ym) / sm)                    # response noise truncated
        psi = psi / pt.maximum(z_trunc, 1e-30)
        return pt.sum(dH * psi[:, None, :], axis=2)                                      # (T, K)

    def _trial_log_likelihood(self, inputs):
        model = pm.Model.get_context()
        N, K = self.grid_resolution, self.n_nodes
        tb = self._tables(inputs)
        s_ix, f_ix, c_ix = inputs['subject_ix'], model['format_ix'], model['mapping_ix']
        s1, f1, c1 = s_ix[:, None], f_ix[:, None], c_ix[:, None]
        rng = self.hi - self.lo
        x_u = pt.clip((model['number'] - self.lo) / rng, 0., 1.)                          # (T,) truth, [0, 1] units
        s_m = pt.specify_shape(inputs['sigma_motor'], (self._n_sub,))[s_ix] / 100.         # (T,) in [0, 1] units

        if self.valuation:
            s_r = (tb['sigma_rep'] / 100.)[s_ix]
            if self.perception:
                s_p = (pt.stack(tb['sigma_per'], axis=1) / 100.)[s_ix, f_ix]              # (T,)
                F0 = _interp_uniform(tb['F_x'], (s_ix, f_ix), x_u, N)                      # (T,)
                z = np.linspace(-5., 5., K + 1)
                ce = pt.clip(F0[:, None] + s_p[:, None] * z[None, :], 0., 1.)              # (T, K+1) cell edges
                cdf = _trunc_cdf(ce, F0[:, None], s_p[:, None])
                wk = cdf[:, 1:] - cdf[:, :-1]
                wk = wk / pt.maximum(pt.sum(wk, axis=1, keepdims=True), 1e-30)             # (T, K)
                node = (ce[:, 1:] + ce[:, :-1]) / 2
                x_hat = _interp_uniform(tb['x_hat'], (s1, f1), node, N)                   # (T, K)
            else:
                wk = pt.ones((x_u.shape[0], 1))
                x_hat = x_u[:, None]
            v_in = _interp_uniform(tb['G'], (s1, c1), x_hat, N)
            e = _interp_uniform(tb['F_v'], (s1, c1), v_in, N)                             # (T, K) value codes
            inv = _inverse_table(tb['v_hat'], N)                                          # (S, C, N)

            def H_at(edges):
                c = _interp_uniform(inv, (s1, c1), edges, N)                              # (T, L+1)
                return _trunc_cdf(c[:, None, :], e[:, :, None], s_r[:, None, None])
        elif self.perception:
            s_p = (pt.stack(tb['sigma_per'], axis=1) / 100.)[s_ix, f_ix]
            F0 = _interp_uniform(tb['F_x'], (s_ix, f_ix), x_u, N)
            # value of a number code: G(x_hat(c)), per subject, format and mapping; its inverse gives the CDF
            S, Fn, C = self._n_sub, (1 if self.formats is None else len(self.formats)), len(self.mappings)
            xh = pt.clip(tb['x_hat'], 0., 1.) * (N - 1)                                   # (S, F, N)
            i = pt.clip(pt.floor(xh), 0, N - 2).astype('int64')
            w = xh - i
            Gt = tb['G'][:, None, :, :]                                                   # (S, 1, C, N)
            ii, ww = i[:, :, None, :], w[:, :, None, :]
            comp = (1 - ww) * pt.take_along_axis(pt.broadcast_to(Gt, (S, Fn, C, N)), pt.broadcast_to(ii, (S, Fn, C, N)), axis=-1) \
                + ww * pt.take_along_axis(pt.broadcast_to(Gt, (S, Fn, C, N)), pt.broadcast_to(ii + 1, (S, Fn, C, N)), axis=-1)
            inv = _inverse_table(comp, N)                                                 # (S, F, C, N)
            wk = pt.ones((x_u.shape[0], 1))

            def H_at(edges):
                c = _interp_uniform(inv, (s1, f1, c1), edges, N)
                return _trunc_cdf(c[:, None, :], F0[:, None, None], s_p[:, None, None])
        else:
            v0 = _interp_uniform(tb['G'], (s_ix, c_ix), x_u, N)                           # exact value, no code noise
            H_at = None

        r = (inputs['response'] - self.lo) / rng

        def prob(a0, a1):
            # the response noise is truncated to the range: only the part of the bin inside it can be answered
            a0, a1 = pt.clip(a0, 0., 1.), pt.clip(a1, 0., 1.)
            if H_at is None:   # response noise only: closed form
                zt = _phi_cdf((1. - v0) / s_m) - _phi_cdf((0. - v0) / s_m)
                return (_phi_cdf((a1 - v0) / s_m) - _phi_cdf((a0 - v0) / s_m)) / pt.maximum(zt, 1e-30)
            return pt.sum(wk * self._window_probability(H_at, a0, a1, s_m), axis=1)

        bw = model['bin_width']                                                          # (T,) CHF
        hb = bw / 2 / rng
        p_exact = prob(r - hb, r + hb)
        if self.fit_rounding:
            rho = inputs['rho'][s_ix]
            lowest = self.round_step * np.ceil(self.lo / self.round_step)
            lower = pt.switch(pt.le(inputs['response'], lowest + 1e-6), 0., r - self.round_step / 2 / rng)
            p_round = prob(lower, r + self.round_step / 2 / rng) * model['is_round']
            p_answer = (1 - rho) * p_exact + rho * p_round
        else:
            p_answer = p_exact
        d = self.x_grid[1] - self.x_grid[0]
        p_answer = (1 - self.lapse_rate) * p_answer + self.lapse_rate * bw / (self.hi - self.lo + d)
        return pt.log(pt.clip(p_answer, 1e-300, 1.))

    def _compute_trial_distributions(self, model_inputs):
        raise NotImplementedError('MagnitudeValueLocalModel scores responses; predict with MagnitudeValueModel')
