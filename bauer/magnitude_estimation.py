"""
Bayesian-observer models of magnitude estimation on a bounded, discrete scale.

The task: on each trial a magnitude ``n`` is shown (e.g. a dot array) and the
participant reports it on a bounded response scale (e.g. an integer slider from
5 to 25).  The generative model is

1. **Encoding.**  A noisy internal measurement ``m``:

   * ``encoding='weber'``     m ~ N(log n, nu^2)                 (log-normal noise)
   * ``encoding='constant'``  m ~ N(n, nu^2)                     (constant, items)
   * ``encoding='flexible'``  m ~ N(n, (nu * (n / n_ref)^beta)^2) (power law; beta = 0
     is ``constant``, beta = 1 is scalar variability)

2. **Prior.**  The observer's prior over ``n``:

   * ``prior='true'``       uniform on the presented range (the task distribution)
   * ``prior='fitted'``     Gaussian with free mean and width, in the encoding
     space (log-normal for ``weber``, Gaussian in items otherwise)
   * ``prior='empirical'``  each participant's own histogram of presented ``n``

3. **Readout.**  The posterior mean (``readout='mean'``, squared-error loss) or
   the posterior median (``readout='median'``, absolute-error loss) of ``n``
   given ``m``, in the units of the response scale.

4. **Response.**  Motor noise ``r ~ N(n_hat, sigma_r^2)``, discretised to the
   integer response options and **censored** at the scale ends (all mass below
   the lowest option goes to that option, likewise at the top).  With
   ``lapse=True`` a free per-participant lapse rate ``p_lapse`` mixes in a
   uniform distribution over the response options.

5. **Leakage of a second, uncued magnitude** (optional; needs ``n_uncued``):

   * ``leakage='mixture'``  with probability ``leak`` the participant reports the
     *other* array (a swap), else the cued one.
   * ``leakage='average'``  the report is ``(1 - leak) * n_hat(m_cued) +
     leak * n_hat(m_uncued)``.  The uncued term is integrated with a Gaussian
     approximation (its mean and variance given ``n_uncued``), exact in the
     limit of small ``leak``.

Implementation: everything is computed on fixed grids.  For each participant
the estimate ``n_hat(m)`` is precomputed on a measurement grid (posterior over a
hypothesis grid), the response distribution of every presented ``n`` is the
measurement density integrated against the discretised motor kernel (one
batched matmul), and the per-trial likelihood is a ``pm.Categorical`` over the
response options.  Because the likelihood is an *observed* RV, pointwise
log-likelihoods (``pm.compute_log_likelihood`` -> LOO) and posterior-predictive
draws (``pm.sample_posterior_predictive``) work out of the box.

Data conventions: ``n`` (presented, integer), ``response`` (integer on the
response scale), optional ``n_uncued``; a ``subject`` index level for
hierarchical fits.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .estimation import EstimationBaseModel


ENCODINGS = ('weber', 'constant', 'flexible')
PRIORS = ('true', 'fitted', 'empirical')
READOUTS = ('mean', 'median')
LEAKAGES = (None, 'mixture', 'average')


def _norm_cdf(x):
    return 0.5 * pt.erfc(-x / np.sqrt(2.0))


class MagnitudeEstimationModel(EstimationBaseModel):
    """Hierarchical Bayesian observer for continuous magnitude estimation.

    Parameters
    ----------
    paradigm : pd.DataFrame
        Columns ``n`` and ``response`` (integers), optionally ``n_uncued``;
        a ``subject`` index level for hierarchical fits.
    encoding : {'weber', 'constant', 'flexible'}
    prior : {'true', 'fitted', 'empirical'}
    readout : {'mean', 'median'}
    lapse : bool
        Free per-participant lapse rate (uniform over response options).
    leakage : {None, 'mixture', 'average'}
        Influence of the uncued magnitude ``n_uncued`` (see module docstring).
    stimulus_range : (int, int)
        Smallest and largest presented magnitude (inclusive integers).  The
        ``'true'`` prior is uniform on ``[lo - .5, hi + .5]``.
    response_range : (int, int)
        Smallest and largest response option (inclusive integers).
    n_ref : float
        Reference magnitude of the ``flexible`` noise power law; ``nu`` is the
        encoding s.d. (items) at ``n_ref``.
    n_measurement_grid, n_hypothesis_grid : int
        Grid sizes (measurement axis; observer's hypothesis axis).
    hypothesis_range : (float, float)
        Support of the observer's hypothesis grid (must contain the prior's mass).
    """

    paradigm_keys = ['n']

    def __init__(self, paradigm=None, encoding='weber', prior='true', readout='mean',
                 lapse=True, leakage=None, stimulus_range=(5, 24),
                 response_range=(5, 25), n_ref=12.0, n_measurement_grid=150,
                 n_hypothesis_grid=246, hypothesis_range=(1.0, 50.0), **kwargs):
        if encoding not in ENCODINGS:
            raise ValueError(f'encoding must be one of {ENCODINGS}')
        if prior not in PRIORS:
            raise ValueError(f'prior must be one of {PRIORS}')
        if readout not in READOUTS:
            raise ValueError(f'readout must be one of {READOUTS}')
        if leakage not in LEAKAGES:
            raise ValueError(f'leakage must be one of {LEAKAGES}')
        self.encoding = encoding
        self.prior = prior
        self.readout = readout
        self.lapse = lapse
        self.leakage = leakage
        self.stimulus_range = tuple(int(x) for x in stimulus_range)
        self.response_range = tuple(int(x) for x in response_range)
        self.n_ref = float(n_ref)
        self.n_measurement_grid = int(n_measurement_grid)
        self.n_hypothesis_grid = int(n_hypothesis_grid)
        self.hypothesis_range = tuple(float(x) for x in hypothesis_range)
        if leakage is not None:
            self.paradigm_keys = ['n', 'n_uncued']
        super().__init__(paradigm, lapse_rate=0.0, **kwargs)
        self._setup_grids(paradigm)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    @property
    def log_space(self):
        return self.encoding == 'weber'

    def get_free_parameters(self):
        pars = {}
        if self.encoding == 'weber':
            # nu in log units; softplus(-1) = .31
            pars['nu'] = dict(mu_intercept=-1.0, sigma_intercept=0.75,
                              cauchy_sigma_intercept=0.5, transform='softplus')
        else:
            # nu in items (at n_ref for 'flexible'); softplus(5) ~ 5
            pars['nu'] = dict(mu_intercept=5.0, sigma_intercept=2.5,
                              cauchy_sigma_intercept=1.5, transform='softplus')
        if self.encoding == 'flexible':
            pars['nu_exponent'] = dict(mu_intercept=0.5, sigma_intercept=1.0,
                                       cauchy_sigma_intercept=0.5, transform='identity')
        if self.prior == 'fitted':
            lo, hi = self.stimulus_range
            if self.log_space:
                mid = 0.5 * (np.log(lo) + np.log(hi))
                pars['prior_mu'] = dict(mu_intercept=mid, sigma_intercept=0.3,
                                        cauchy_sigma_intercept=0.2, transform='identity')
                pars['prior_sd'] = dict(mu_intercept=-0.5, sigma_intercept=0.75,
                                        cauchy_sigma_intercept=0.5, transform='softplus')
            else:
                pars['prior_mu'] = dict(mu_intercept=0.5 * (lo + hi), sigma_intercept=3.0,
                                        cauchy_sigma_intercept=2.0, transform='identity')
                pars['prior_sd'] = dict(mu_intercept=(hi - lo + 1) / np.sqrt(12),
                                        sigma_intercept=2.5, cauchy_sigma_intercept=1.5,
                                        transform='softplus')
        # motor noise in items; softplus(1) = 1.31
        pars['sigma_r'] = dict(mu_intercept=1.0, sigma_intercept=1.0,
                               cauchy_sigma_intercept=0.75, transform='softplus')
        if self.lapse:
            pars['p_lapse'] = dict(transform='beta', beta_mu_mean=0.05, beta_mu_kappa=8.0)
        if self.leakage is not None:
            pars['leak'] = dict(transform='beta', beta_mu_mean=0.05, beta_mu_kappa=8.0)
        return pars

    def get_model_inputs(self, parameters):
        return {key: self.subjectwise(key) for key in self.free_parameters}

    # ------------------------------------------------------------------
    # Grids and paradigm
    # ------------------------------------------------------------------

    def _setup_grids(self, paradigm):
        lo, hi = self.stimulus_range
        self.stimuli = np.arange(lo, hi + 1, dtype=float)                   # (K,)
        rlo, rhi = self.response_range
        self.response_options = np.arange(rlo, rhi + 1, dtype=float)        # (R,)
        # Category edges, censored at the scale ends
        # (finite outer edges: an infinite edge makes d/d(sd) = inf * 0 = NaN)
        self.response_edges = np.concatenate(
            [[-1e4], self.response_options[:-1] + 0.5, [1e4]])

        h0, h1 = self.hypothesis_range
        self.hyp_grid = np.linspace(h0, h1, self.n_hypothesis_grid)        # (H,)
        self.d_hyp = self.hyp_grid[1] - self.hyp_grid[0]

        if self.log_space:
            self.meas_grid = np.linspace(np.log(h0) - 0.5, np.log(h1) + 0.5,
                                         self.n_measurement_grid)
        else:
            self.meas_grid = np.linspace(h0 - 20.0, h1 + 10.0, self.n_measurement_grid)
        self.d_meas = self.meas_grid[1] - self.meas_grid[0]

        # True prior: uniform on [lo - .5, hi + .5]; cell-overlap weights.
        a, b = lo - 0.5, hi + 0.5
        left = np.clip(self.hyp_grid - self.d_hyp / 2, a, b)
        right = np.clip(self.hyp_grid + self.d_hyp / 2, a, b)
        w = right - left
        self.true_prior = w / w.sum()

        self.subject_ids = None
        self.empirical_prior = None
        if paradigm is not None:
            if 'subject' in paradigm.index.names:
                subj = paradigm.index.get_level_values('subject')
            elif 'subject' in paradigm.columns:
                subj = paradigm['subject'].values
            else:
                subj = np.zeros(len(paradigm), dtype=int)
            codes, uniques = pd.factorize(subj)
            self.subject_ids = uniques
            if self.prior == 'empirical':
                n = paradigm['n'].values.astype(int)
                counts = np.zeros((len(uniques), len(self.stimuli)))
                np.add.at(counts, (codes, n - lo), 1.0)
                # Spread each integer's mass over its [k - .5, k + .5] cell
                cell = np.zeros((len(self.stimuli), len(self.hyp_grid)))
                for k, nk in enumerate(self.stimuli):
                    l_ = np.clip(self.hyp_grid - self.d_hyp / 2, nk - .5, nk + .5)
                    r_ = np.clip(self.hyp_grid + self.d_hyp / 2, nk - .5, nk + .5)
                    cell[k] = (r_ - l_)
                emp = counts @ cell
                self.empirical_prior = emp / emp.sum(1, keepdims=True)

    def _get_paradigm(self, paradigm=None, subject_mapping=None):
        if paradigm is None:
            paradigm = self.paradigm
        lo, hi = self.stimulus_range
        rlo, rhi = self.response_range
        n = paradigm['n'].values
        if np.any(n != np.round(n)) or n.min() < lo or n.max() > hi:
            raise ValueError(f'n must be integers in {self.stimulus_range}')

        paradigm_ = {'_data_n': len(paradigm)}
        if subject_mapping is None:
            if 'subject' in paradigm.index.names:
                paradigm_['subject_ix'], _ = pd.factorize(paradigm.index.get_level_values('subject'))
            elif 'subject' in paradigm.columns:
                paradigm_['subject_ix'], _ = pd.factorize(paradigm['subject'])
            else:
                paradigm_['subject_ix'] = np.zeros(len(paradigm), dtype=int)
        else:
            s = (paradigm.index.get_level_values('subject') if 'subject' in paradigm.index.names
                 else paradigm['subject'])
            paradigm_['subject_ix'] = np.array([subject_mapping[x] for x in s])

        paradigm_['stimulus_ix'] = (n - lo).astype(int)
        if self.leakage is not None:
            u = paradigm['n_uncued'].values
            paradigm_['uncued_ix'] = (u - lo).astype(int)

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
    # The observer
    # ------------------------------------------------------------------

    def _encoding_sd(self, nu, nu_exponent, x):
        """Encoding s.d. at magnitudes x. nu: (S,), x: (..., P) -> (S, P)."""
        if self.encoding == 'flexible':
            return nu[:, None] * (x[None, :] / self.n_ref) ** nu_exponent[:, None]
        return pt.ones((1, x.shape[0])) * nu[:, None]

    def _encode(self, x):
        return np.log(x) if self.log_space else x

    def _prior_on_grid(self, inputs):
        """(S, H) normalised prior mass on the hypothesis grid."""
        h = pt.as_tensor_variable(self.hyp_grid)
        if self.prior == 'true':
            return pt.as_tensor_variable(self.true_prior)[None, :]
        if self.prior == 'empirical':
            return pt.as_tensor_variable(self.empirical_prior)
        mu = inputs['prior_mu'][:, None]
        sd = inputs['prior_sd'][:, None]
        if self.log_space:
            logdens = -0.5 * ((pt.log(h)[None, :] - mu) / sd) ** 2 - pt.log(h)[None, :]
        else:
            logdens = -0.5 * ((h[None, :] - mu) / sd) ** 2
        logdens = logdens - pt.max(logdens, axis=1, keepdims=True)
        dens = pt.exp(logdens)
        return dens / pt.sum(dens, axis=1, keepdims=True)

    def _estimate_on_measurement_grid(self, inputs):
        """n_hat(m): (S, M) estimate for every measurement on the grid."""
        nu = inputs['nu']
        beta = inputs.get('nu_exponent', None)
        h = self.hyp_grid
        m = pt.as_tensor_variable(self.meas_grid)
        sd_h = self._encoding_sd(nu, beta, pt.as_tensor_variable(h))          # (S|1, H)
        mu_h = pt.as_tensor_variable(self._encode(h))                           # (H,)
        # log p(m | h): (S, M, H)
        loglik = (-0.5 * ((m[None, :, None] - mu_h[None, None, :]) / sd_h[:, None, :]) ** 2
                  - pt.log(sd_h)[:, None, :])
        logprior = pt.log(self._prior_on_grid(inputs) + 1e-300)                 # (S|1, H)
        logpost = loglik + logprior[:, None, :]
        logpost = logpost - pt.max(logpost, axis=2, keepdims=True)
        post = pt.exp(logpost)
        post = post / pt.sum(post, axis=2, keepdims=True)                       # (S, M, H)
        hh = pt.as_tensor_variable(h)
        if self.readout == 'mean':
            return pt.sum(post * hh[None, None, :], axis=2)
        # Median: posterior mass is piecewise constant over cells of width d_hyp
        # centred on the grid; the median is the point where the cumulative mass
        # crosses .5, linearly interpolated within its cell (continuous in the
        # parameters, so NUTS sees a usable gradient).
        cum = pt.cumsum(post, axis=2)
        before = cum - post
        frac = pt.clip((0.5 - before) / (post + 1e-30), 0.0, 1.0)
        return (h[0] - self.d_hyp / 2) + self.d_hyp * pt.sum(frac, axis=2)

    def _measurement_density(self, inputs):
        """p(m | n_k) * dm for every presented magnitude: (S, K, M)."""
        nu = inputs['nu']
        beta = inputs.get('nu_exponent', None)
        x = pt.as_tensor_variable(self.stimuli)
        sd = self._encoding_sd(nu, beta, x)                                     # (S|1, K)
        mu = pt.as_tensor_variable(self._encode(self.stimuli))
        m = pt.as_tensor_variable(self.meas_grid)
        logd = -0.5 * ((m[None, None, :] - mu[None, :, None]) / sd[:, :, None]) ** 2
        logd = logd - pt.max(logd, axis=2, keepdims=True)
        d = pt.exp(logd)
        return d / pt.sum(d, axis=2, keepdims=True)

    def _motor_kernel(self, centre, sd):
        """P(response option | centre) for centre (..., M), sd broadcastable."""
        edges = pt.as_tensor_variable(self.response_edges)
        z = (edges - centre[..., None]) / sd[..., None]
        cdf = _norm_cdf(z)
        return cdf[..., 1:] - cdf[..., :-1]                                     # (..., R)

    def response_table(self, inputs):
        """Response distribution per participant and presented magnitude.

        Returns (S, K, R), or (S, K, K, R) indexed [s, cued, uncued, r] for
        ``leakage='average'``.  Lapses are mixed in, mixtures are not (the
        mixture is formed per trial in ``trial_probabilities``).
        """
        n_hat = self._estimate_on_measurement_grid(inputs)                      # (S, M)
        p_m = self._measurement_density(inputs)                                 # (S, K, M)
        sigma_r = inputs['sigma_r']
        R = len(self.response_options)

        if self.leakage == 'average':
            leak = inputs['leak']
            mu_u = pt.sum(p_m * n_hat[:, None, :], axis=2)                      # (S, K)
            var_u = pt.sum(p_m * (n_hat[:, None, :] - mu_u[:, :, None]) ** 2, axis=2)
            centre = ((1 - leak)[:, None, None] * n_hat[:, None, :]
                      + leak[:, None, None] * mu_u[:, :, None])                 # (S, U, M)
            sd = pt.sqrt(sigma_r[:, None] ** 2 + leak[:, None] ** 2 * var_u)   # (S, U)
            kern = self._motor_kernel(centre, pt.repeat(sd[:, :, None], centre.shape[2], axis=2))
            # (S, K_c, M) x (S, U, M, R) -> (S, K_c, U, R), as one batched matmul
            # over M (no 5-D intermediate in the gradient graph)
            S_, U_ = kern.shape[0], kern.shape[1]
            kern_m = kern.dimshuffle(0, 2, 1, 3).reshape((S_, kern.shape[2], U_ * R))
            table = pt.batched_dot(p_m, kern_m).reshape((S_, p_m.shape[1], U_, R))
        else:
            kern = self._motor_kernel(n_hat, sigma_r[:, None] * pt.ones_like(n_hat))  # (S, M, R)
            table = pt.batched_dot(p_m, kern)                                   # (S, K, R)

        if self.lapse:
            lam = inputs['p_lapse']
            lam = lam.reshape((-1,) + (1,) * (table.ndim - 1))
            table = (1 - lam) * table + lam / R
        return table

    def trial_probabilities(self, inputs, subject_ix, stimulus_ix, uncued_ix=None):
        table = self.response_table(inputs)
        if self.leakage == 'average':
            p = table[subject_ix, stimulus_ix, uncued_ix]
        elif self.leakage == 'mixture':
            leak = inputs['leak'][subject_ix][:, None]
            p = (1 - leak) * table[subject_ix, stimulus_ix] + leak * table[subject_ix, uncued_ix]
        else:
            p = table[subject_ix, stimulus_ix]
        p = pt.clip(p, 1e-10, 1.0)
        return p / pt.sum(p, axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # PyMC graph
    # ------------------------------------------------------------------

    def build_estimation_model(self, data=None, coords=None, hierarchical=True,
                               save_p_choice=False, flat_prior=False, paradigm=None):
        if data is None:
            data = paradigm if paradigm is not None else self.paradigm
        self._setup_grids(data)
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
        p = self.trial_probabilities(
            inputs, model['subject_ix'], model['stimulus_ix'],
            model['uncued_ix'] if self.leakage is not None else None)
        if save_p_choice:
            p = pm.Deterministic('p_response', p)
        pm.Categorical('response', p=p, observed=model['response_ix'])

    # ------------------------------------------------------------------
    # Prediction / simulation
    # ------------------------------------------------------------------

    def _parameter_arrays(self, parameters, subject_ids):
        """dict name -> (S,) float array, aligned to ``subject_ids``."""
        if isinstance(parameters, pd.DataFrame):
            pars = parameters.loc[list(subject_ids)]
            return {k: pars[k].values.astype(float) for k in self.free_parameters}
        out = {}
        for k in self.free_parameters:
            v = np.atleast_1d(np.asarray(parameters[k], dtype=float))
            out[k] = np.broadcast_to(v, (len(subject_ids),)).copy()
        return out

    def predict_response_distribution(self, paradigm, parameters):
        """Per-trial probabilities over the response options.

        Returns a DataFrame (index = paradigm.index, columns = response options).
        ``parameters``: DataFrame indexed by subject, or a dict of scalars/arrays.
        """
        if self.prior == 'empirical':
            self._setup_grids(paradigm)
        if 'subject' in paradigm.index.names:
            subj = paradigm.index.get_level_values('subject')
        elif 'subject' in paradigm.columns:
            subj = paradigm['subject'].values
        else:
            subj = np.zeros(len(paradigm), dtype=int)
        codes, uniques = pd.factorize(subj)
        pars = self._parameter_arrays(parameters, uniques)
        inputs = {k: pt.as_tensor_variable(v) for k, v in pars.items()}
        par_ = self._get_paradigm(paradigm)
        par_['subject_ix'] = codes
        p = self.trial_probabilities(
            inputs, pt.as_tensor_variable(par_['subject_ix']),
            pt.as_tensor_variable(par_['stimulus_ix']),
            pt.as_tensor_variable(par_['uncued_ix']) if self.leakage is not None else None).eval()
        return pd.DataFrame(p, index=paradigm.index,
                            columns=pd.Index(self.response_options.astype(int), name='response'))

    def predict(self, paradigm, parameters):
        """Predicted mean and s.d. of the response per trial."""
        p = self.predict_response_distribution(paradigm, parameters)
        r = self.response_options
        mean = p.values @ r
        sd = np.sqrt(np.maximum(p.values @ r ** 2 - mean ** 2, 0))
        out = paradigm.copy()
        out['predicted_mean'] = mean
        out['predicted_sd'] = sd
        return out

    def simulate(self, paradigm, parameters, n_samples=1, rng=None):
        """Simulate integer responses (lapses, censoring and leakage included).

        Returns a long DataFrame with a ``sample`` index level and a
        ``simulated_response`` column (the observed ``response`` column, if
        any, is dropped to avoid collisions).
        """
        rng = np.random.default_rng(rng)
        p = self.predict_response_distribution(paradigm, parameters).values
        cdf = np.cumsum(p, axis=1)
        u = rng.random((n_samples, len(p), 1))
        ix = (u > cdf[None]).sum(-1)
        ix = np.minimum(ix, p.shape[1] - 1)
        sim = self.response_options[ix].T                                      # (n_trials, n_samples)
        # Built by position, not by index join: a paradigm index need not be unique.
        par = paradigm.drop(columns=['response'], errors='ignore')
        out = pd.concat([par.assign(sample=k + 1, simulated_response=sim[:, k])
                         for k in range(n_samples)])
        return out.set_index('sample', append=True)

    def ppc(self, paradigm=None, idata=None, n_posterior_samples=200, random_seed=None,
            progressbar=False):
        """Posterior-predictive simulated datasets for the fitted trials.

        Draws ``n_posterior_samples`` posterior draws and simulates one full
        dataset (same trials) from each.  Returns a long DataFrame with index
        ``paradigm.index + ppc_sample`` and a ``simulated_response`` column.
        """
        idata = self._get_idata(idata)
        if paradigm is None:
            paradigm = self.paradigm
        post = idata.posterior.stack(sample=('chain', 'draw'))
        rng = np.random.default_rng(random_seed)
        sel = rng.choice(post.sizes['sample'], size=min(n_posterior_samples, post.sizes['sample']),
                         replace=False)
        thinned = post.isel(sample=sel)
        thinned = thinned.assign_coords(sample=np.arange(len(sel))).rename(sample='draw')
        thinned = thinned.expand_dims('chain').transpose('chain', 'draw', ...)
        import arviz as az
        id_thin = az.InferenceData(posterior=thinned)
        with self.estimation_model:
            pp = pm.sample_posterior_predictive(id_thin, var_names=['response'],
                                                random_seed=random_seed, progressbar=progressbar)
        arr = pp.posterior_predictive['response'].values[0]                     # (draw, trial)
        sim = self.response_options[arr].T
        out = pd.DataFrame(sim, index=paradigm.index,
                           columns=pd.Index(np.arange(len(sel)), name='ppc_sample'))
        return out.stack().rename('simulated_response').to_frame()

    def sample_parameters_from_prior(self, n_subjects=None):
        return super().sample_parameters_from_prior(n_subjects=n_subjects)
