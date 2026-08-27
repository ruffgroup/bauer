"""
Efficient coding models for orientation-to-value estimation tasks.

Implements the three representational architectures from:
"Beyond perception: Multi-stage efficient coding of perceptual and value
representations" (Bedi, de Hollander, Harl & Ruff).

Models:
    - EfficientPerceptionModel: efficient coding only in orientation space
    - EfficientValuationModel: efficient coding only in value space
    - SequentialEfficientCodingModel: efficient coding at both stages
    - CategoricalSequentialModel: + cardinal categorical stabilization
"""

import numpy as np
from numpy import trapezoid as np_trapz
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytensor.tensor.math as ptm
from .estimation import EstimationBaseModel


# ============================================================================
# Orientation-to-value lookup tables
# ============================================================================

# 25 equally spaced orientations from 0 to 180 degrees
MAPPING_ORIENTATIONS_DEG = np.linspace(0, 180, 25)

MAPPING_VALUES = {
    # "Uniform" / "linear" mapping: near-linear
    'linear': np.array([
        2.0, 3.5, 5.5, 7.0, 8.5, 10.5, 12.0, 13.5,
        15.5, 17.0, 18.5, 20.5, 22.0, 23.5, 25.5, 27.0,
        28.5, 30.5, 32.0, 33.5, 35.5, 37.0, 38.5, 40.5, 42.0]),
    # "CDF" / "Misaligned" mapping: S-shaped, values compressed at extremes
    'cdf': np.array([
        2.0, 5.5, 8.0, 10.0, 11.0, 11.5, 12.0, 12.5,
        13.0, 14.0, 16.0, 18.5, 22.0, 25.5, 28.0, 30.0,
        31.0, 31.5, 32.0, 32.5, 33.0, 34.0, 36.0, 38.5, 42.0]),
    # "Inverse CDF" / "Aligned" mapping: inverted S, compressed in middle
    'inverse_cdf': np.array([
        2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 11.5, 16.0,
        18.5, 20.0, 21.0, 21.5, 22.0, 22.5, 23.0, 24.0,
        25.5, 28.0, 32.5, 36.5, 39.0, 40.0, 41.0, 41.5, 42.0]),
}

# Alias paper names
MAPPING_VALUES['uniform'] = MAPPING_VALUES['linear']
MAPPING_VALUES['misaligned'] = MAPPING_VALUES['cdf']
MAPPING_VALUES['aligned'] = MAPPING_VALUES['inverse_cdf']

# Value range
V_MIN = 2.0
V_MAX = 42.0

# Perceptual loss exponent.  Bedi et al. fix q_per = 8 across all fits (and
# q_val = 2, the posterior mean, in value space).  q_per = 2 would make the
# perceptual estimator the circular posterior mean, which under-predicts
# repulsion away from the cardinal orientations by roughly a factor of four.
Q_PER = 8.0

# ---------------------------------------------------------------------------
# kappa_r in interpretable units
# ---------------------------------------------------------------------------
# kappa_r is a von Mises concentration in DOUBLED-angle space (phi = 2*theta),
# which makes it hard to reason about.  Circular SD in phi is ~1/sqrt(kappa)
# radians, so the corresponding ORIENTATION SD is half that, in degrees:
#
#     kappa =  12 -> 8.3 deg     kappa =  83 -> 3.1 deg
#     kappa =  30 -> 5.2 deg     kappa = 443 -> 1.4 deg
#
# The paper's fitted kappa_r of 12-83 is therefore 3-8 deg of perceptual noise,
# which is what you would expect for 2%-contrast gratings.


def orientation_sd_from_kappa(kappa):
    """Orientation SD in degrees implied by a doubled-angle von Mises kappa."""
    return np.rad2deg(1.0 / np.sqrt(kappa)) / 2.0


def kappa_from_orientation_sd(sd_deg):
    """Inverse of :func:`orientation_sd_from_kappa`."""
    return (np.rad2deg(1.0) / 2.0 / sd_deg) ** 2


def max_resolvable_kappa(grid_resolution):
    """Largest kappa an N-point orientation grid can distinguish.

    The grid spans [0, 180) degrees in N steps, so its orientation step is
    180/N degrees.  Once the noise SD falls below one step the likelihood is
    flat in kappa and the parameter is unidentified -- posteriors simply run
    off to wherever the prior stops them.  Closed form: (N / 2*pi)**2.

        N =  31 -> kappa above  24 is unresolvable (5.8 deg step)
        N =  51 -> kappa above  66 is unresolvable (3.5 deg step)
        N = 101 -> kappa above 258 is unresolvable (1.8 deg step)

    Observed: a single-subject fit at N=51 with a loose prior returned
    kappa_r = 443, i.e. 1.4 deg of noise on a 3.5 deg grid -- 2.6x narrower
    than one cell, and pure prior/flat-likelihood wandering.
    """
    return (grid_resolution / (2.0 * np.pi)) ** 2


def kappa_r_prior(grid_resolution=None):
    """Prior on kappa_r, capped at what the grid can actually resolve.

    Centred to cover the paper's fitted 12-83 (3-8 deg of noise).  When the
    grid is coarse enough that this would spill into the unresolvable region,
    the prior is tightened so it stops before the flat part of the likelihood
    rather than letting the sampler wander there.
    """
    mu, sigma = 30.0, 20.0
    if grid_resolution is not None:
        k_max = max_resolvable_kappa(grid_resolution)
        mu, sigma = min(mu, k_max / 2.0), min(sigma, k_max / 4.0)
    return {'mu_intercept': mu, 'sigma_intercept': sigma,
            'transform': 'softplus'}


# Back-compat default for callers that do not know their grid yet.
KAPPA_R_PRIOR = kappa_r_prior()
SIGMA_REP_PRIOR = {'mu_intercept': 0.5, 'sigma_intercept': 1.0,
                   'transform': 'softplus'}


def _efficient_cdf_pt(prior, d_ori):
    """Efficient-coding transform F(theta) from an orientation prior (Eq. 1).

    Tensor version, so the prior can depend on a fitted parameter.  `prior` is
    (S, O) and unnormalised; returns (S, O) mapped onto [0, 2*pi).
    """
    p = prior / (pt.sum(prior, axis=-1, keepdims=True) * d_ori + 1e-30)
    c = pt.cumsum(p * d_ori, axis=-1)
    c = c - c[..., :1]
    return c / (c[..., -1:] + 1e-30) * 2 * np.pi


def _interp_pt(x, xp_last, values):
    """Linear interpolation of `values` (S, O) at positions `x` (K,) given a
    uniform grid spanning [0, xp_last).  Used to locate each stimulus on a
    parameter-dependent encoding transform."""
    n = values.shape[-1]
    f = x / xp_last * pt.cast(n, 'floatX')
    i0 = pt.clip(pt.floor(f), 0, pt.cast(n, 'floatX') - 2.0)
    w = f - i0
    i0 = pt.cast(i0, 'int64')
    return values[..., i0] * (1.0 - w) + values[..., i0 + 1] * w


def _seam_mask(encoded_positions, rep_grid, step):
    """(..., O, M) indicator that a hypothesised orientation and the measurement
    that produced it lie on the same side of the 0/180 deg seam.

    Orientation is pi-periodic, so 0 deg and 180 deg are the same grating -- but
    G is not: G(0 deg) = 2 CHF and G(180 deg) = 42 CHF.  A participant who has
    learned the mapping never bids 2 CHF for a 179 deg grating, so the decoder
    must not put posterior mass on the far side of the seam from its own
    measurement.  Note the reference is the MEASUREMENT (rep_grid), not the
    stimulus: the observer does not know the stimulus, and referencing it makes
    the mask a (K, O) object that cannot even be broadcast against the
    (S, O, M) posterior.

    Implemented as a smooth (differentiable) window one grid cell wide rather
    than a hard cut, so it does not reintroduce a discontinuity for NUTS.
    Allowed pairs get 0.5 rather than 1.0, which cancels in the renormalisation
    that always follows.
    """
    d = encoded_positions[..., None] - rep_grid                      # (..., O, M)
    wrapped = pt.abs(pt.mod(d + np.pi, 2 * np.pi) - np.pi)           # circular distance
    straight = pt.abs(d)                                             # distance on the line
    # A candidate crosses the seam when the short way round is not the direct
    # way; suppress those smoothly.
    return ptm.sigmoid(-(straight - wrapped) / (0.5 * step + 1e-12))


def _value_from_theta_hat(posterior, ori_grid, G_ext, d_ori, q_per=Q_PER,
                          n_cand=None):
    """Map a perceptual posterior to a value estimate, per paper Eqs. 4 and 7.

    theta_hat = argmin_t E_post[(1 - cos(theta - t))^(q_per/2)].  For q_per = 2
    this is the circular posterior mean and has a closed form; for the paper's
    q_per = 8 it does not, so it is minimised on the orientation grid and then
    refined to sub-grid resolution by fitting a parabola to the loss at the
    three points around the minimum.

    The refinement is not cosmetic.  A plain grid argmin makes theta_hat -- and
    therefore the whole likelihood -- a STEP function of kappa_r, and NUTS cannot
    integrate that: on a 26-subject fit it drove the step size to ~2e-4 and hit
    the 1023-step treedepth cap on every iteration, for an ETA of 24 h.  The
    parabolic vertex moves continuously with kappa_r, so the surface is smooth
    and differentiable almost everywhere, and it removes the +-d_ori/2 snapping
    error (0.9 CHF at N=101) into the bargain.

    ``G_ext`` is G evaluated on ``ori_grid`` PLUS one extra point at 2*pi, i.e.
    180 deg.  Orientation is pi-periodic but G is not -- G(0 deg) = 2 CHF and
    G(180 deg) = 42 CHF -- so the value interpolation must run off the end of
    the grid to 180 deg rather than wrapping around to 0 deg.

    Parameters
    ----------
    posterior : (S, O, M) tensor, normalised over the orientation axis O.
    ori_grid : (O,) doubled-angle orientation grid.
    G_ext : (C, O + 1) value of each grid orientation under each mapping, with
        the 2*pi endpoint appended.
    d_ori : float, orientation grid spacing.

    Returns
    -------
    (C, S, M) tensor of value estimates.
    """
    # Loss of each candidate estimate under the posterior: (S, M, O_cand).
    # Written as a matmul rather than broadcast-and-sum: the latter materialises
    # an (S, O, M, O_cand) intermediate, O(N^3) per subject.
    cos_dt = ptm.cos(ori_grid[:, None] - ori_grid[None, :])          # (O, O_cand)
    loss_kernel = (1.0 - cos_dt) ** (q_per / 2.0)
    post_sm_o = posterior.dimshuffle(0, 2, 1)                        # (S, M, O)
    S, M, O = post_sm_o.shape[0], post_sm_o.shape[1], post_sm_o.shape[2]
    loss = pt.dot(pt.reshape(post_sm_o, (S * M, O)),
                  loss_kernel * d_ori)                               # (S*M, O_cand)

    # Parabolic refinement of the grid argmin.  The loss is circular in the
    # candidate axis, so the neighbours wrap.
    # Gather the loss at the argmin and its two neighbours with a one-hot
    # contraction over the CANDIDATE axis, whose length is a Python int known at
    # graph-build time. Both the (arange, index) pair and take_along_axis emit a
    # pt.arange over a symbolic length, which JAX refuses to JIT ("requires the
    # arguments of jax.numpy.arange to be constants") -- that is what kills the
    # numpyro sampler.
    n_cand = int(n_cand) if n_cand is not None else int(ori_grid.type.shape[0])
    i = pt.argmin(loss, axis=1)                                      # (S*M,)
    cand = pt.arange(n_cand)                                         # static length

    def _at(idx):
        return pt.sum(loss * pt.eq(cand[None, :], idx[:, None]), axis=1)
    l_m = _at((i - 1) % n_cand)
    l_0 = _at(i)
    l_p = _at((i + 1) % n_cand)
    denom = l_m - 2.0 * l_0 + l_p
    delta = pt.switch(pt.gt(denom, 1e-30), 0.5 * (l_m - l_p) / (denom + 1e-30), 0.0)
    delta = pt.clip(delta, -0.5, 0.5)
    f = pt.mod(pt.cast(i, 'floatX') + delta, np.float64(n_cand))  # (S*M,)

    # Linear interpolation of G at the fractional index.  Index n_cand of G_ext
    # is 180 deg, NOT a wrap back to 0 deg.
    i0 = pt.clip(pt.floor(f), 0, np.float64(n_cand) - 1.0)
    w = f - i0
    i0 = pt.cast(i0, 'int64')
    v_hat = G_ext[:, i0] * (1.0 - w)[None, :] + G_ext[:, i0 + 1] * w[None, :]  # (C, S*M)
    v_hat = pt.reshape(v_hat, (G_ext.shape[0], S, M))
    return pt.clip(v_hat, V_MIN, V_MAX)                              # (C, S, M)


def orientation_to_value_np(orientation_deg, mapping='linear'):
    """Map orientation (degrees) to value (CHF) via linear interpolation.

    Works on numpy arrays. If ``mapping`` is a single string, applies the same
    mapping to all orientations. If ``mapping`` is an array of strings (one per
    orientation), applies each element's mapping.
    """
    if isinstance(mapping, str):
        return np.interp(orientation_deg, MAPPING_ORIENTATIONS_DEG,
                         MAPPING_VALUES[mapping])

    # Vectorized: different mapping per element
    orientation_deg = np.asarray(orientation_deg, dtype=float)
    mapping = np.asarray(mapping, dtype=str)
    result = np.empty_like(orientation_deg)
    for m in np.unique(mapping):
        mask = mapping == m
        result[mask] = np.interp(orientation_deg[mask],
                                 MAPPING_ORIENTATIONS_DEG,
                                 MAPPING_VALUES[str(m)])
    return result


# ============================================================================
# Orientation priors
# ============================================================================

PRIOR_WEIGHT_PRIOR = {'mu_intercept': 0.5, 'sigma_intercept': 0.5,
                      'transform': 'logistic'}


def orientation_prior_pt(phi, weight):
    """Orientation prior with a free peakedness, p(phi) ~ 1 - w*|sin phi|.

    Nests both fixed variants exactly: w = 0 is the uniform short-term prior,
    w = 0.5 is the paper's long-term cardinal prior (2 - |sin phi|, up to the
    normalising constant).  w -> 1 concentrates all the mass on the cardinals.

    `weight` may be a scalar or a per-subject vector; the returned prior is
    (S, O) and normalised over the orientation axis.
    """
    # No implicit reshaping: the caller decides the broadcast, otherwise a
    # (S, 1) weight silently becomes (S, 1, 1) and every downstream tensor
    # gains a phantom axis.
    w = pt.as_tensor_variable(weight)
    p = 1.0 - w * pt.abs(ptm.sin(phi))
    p = pt.maximum(p, 1e-6)
    return p


def long_term_orientation_prior_np(phi):
    """Long-term orientation prior: p(phi) proportional to 2 - |sin(phi)|.

    phi is in radians on [0, 2*pi] (doubled-angle space).
    Returns unnormalized density.
    """
    p = 2 - np.abs(np.sin(phi))
    return p / np_trapz(p, phi)


def uniform_orientation_prior_np(phi):
    """Uniform orientation prior on [0, 2*pi]."""
    return np.ones_like(phi) / (2 * np.pi)


# ============================================================================
# Model classes
# ============================================================================

class EfficientPerceptionModel(EstimationBaseModel):
    """Efficient coding and Bayesian decoding operate only in orientation space.

    The perceptual estimate theta_hat is mapped deterministically through
    the value function G to produce the value estimate: v_hat = G(theta_hat).

    Free parameter: kappa_r (von Mises precision of perceptual noise).
    """

    paradigm_keys = ['orientation']
    base_parameters = ['kappa_r']

    def __init__(self, paradigm=None, perceptual_prior='long_term',
                 grid_resolution=101, rep_grid_resolution=None, q_per=Q_PER,
                 fit_prior_weight=False, no_seam_crossing=False, **kwargs):
        """
        Parameters
        ----------
        paradigm : pd.DataFrame
            Must have 'orientation' (degrees), 'response' (CHF), 'mapping' columns.
        perceptual_prior : str
            'long_term' or 'uniform'. Which prior governs perceptual encoding/decoding.
        grid_resolution : int
            Number of points for orientation and value grids.
        rep_grid_resolution : int or None
            Number of points for internal representation grid. Defaults to grid_resolution.
        """
        self.perceptual_prior = perceptual_prior
        self.rep_grid_resolution = rep_grid_resolution or grid_resolution
        self.q_per = q_per
        self.fit_prior_weight = fit_prior_weight
        self.no_seam_crossing = no_seam_crossing
        super().__init__(paradigm, grid_resolution=grid_resolution, **kwargs)

    def get_free_parameters(self):
        pars = {'kappa_r': kappa_r_prior(self.grid_resolution)}
        if self.fit_prior_weight:
            pars['prior_weight'] = PRIOR_WEIGHT_PRIOR
        return pars

    def get_model_inputs(self, parameters):
        model = pm.Model.get_context()
        return {
            'kappa_r': self.subjectwise('kappa_r'),
            **({'prior_weight': self.subjectwise('prior_weight')}
               if self.fit_prior_weight else {}),
            'orientation': model['orientation'],
            'response': model['response'],
            'subject_ix': model['subject_ix'],
        }

    def _setup_grids(self, paradigm):
        """Precompute all fixed grids and constants. Called during build_estimation_model."""
        N = self.grid_resolution
        M = self.rep_grid_resolution

        # Orientation grid in doubled-angle space [0, 2*pi]
        self.ori_grid = np.linspace(0, 2 * np.pi, N, endpoint=False)
        self.d_ori = self.ori_grid[1] - self.ori_grid[0]

        # Representation grid (same domain as ori_grid for perceptual stage)
        self.rep_grid = np.linspace(0, 2 * np.pi, M, endpoint=False)
        self.d_rep = self.rep_grid[1] - self.rep_grid[0]

        # Value grid
        self.val_grid = np.linspace(V_MIN, V_MAX, N)
        self.d_val = self.val_grid[1] - self.val_grid[0]

        # Orientation prior
        if self.perceptual_prior == 'long_term':
            self.ori_prior = long_term_orientation_prior_np(self.ori_grid)
        else:
            self.ori_prior = uniform_orientation_prior_np(self.ori_grid)

        # Efficient coding CDF (encoding transform)
        self.ori_cdf = np.zeros(N)
        cumulative = np.cumsum((self.ori_prior[:-1] + self.ori_prior[1:]) / 2 * self.d_ori)
        self.ori_cdf[1:] = cumulative
        # Scale to [0, 2*pi]
        self.ori_cdf = self.ori_cdf / self.ori_cdf[-1] * 2 * np.pi

        # Precompute value mapping G(theta) on the orientation grid for each condition
        # The paradigm tells us which mapping conditions are present
        mappings = list(paradigm['mapping'].astype(str).unique()) if 'mapping' in paradigm.columns else ['linear']
        self.value_on_ori_grid = {}
        self.value_on_ori_grid_ext = {}
        for mapping in mappings:
            # Convert ori_grid (radians, doubled) back to degrees [0, 180]
            ori_deg = self.ori_grid / (2 * np.pi) * 180
            self.value_on_ori_grid[mapping] = orientation_to_value_np(ori_deg, mapping)
            # G is NOT pi-periodic: append the 180 deg endpoint so the value
            # interpolation runs off the end of the grid instead of wrapping.
            self.value_on_ori_grid_ext[mapping] = orientation_to_value_np(
                np.append(ori_deg, 180.0), mapping)

        # Unique stimulus orientations (in degrees) from paradigm
        self.unique_orientations_deg = np.sort(paradigm['orientation'].unique())
        # Convert to doubled-angle radians
        self.unique_orientations_rad = self.unique_orientations_deg * np.pi / 180.0 * 2

        # For each unique orientation, find F_ori(theta_0) via interpolation
        self.encoded_stimulus_locs = np.interp(
            self.unique_orientations_rad, self.ori_grid, self.ori_cdf)

    def build_estimation_model(self, data=None, coords=None, hierarchical=True,
                               save_p_choice=False, flat_prior=False, paradigm=None):
        if data is None:
            data = paradigm if paradigm is not None else self.paradigm
        self._setup_grids(data)
        super().build_estimation_model(data, coords, hierarchical, save_p_choice, flat_prior)

    def _get_paradigm(self, paradigm=None, subject_mapping=None):
        paradigm_ = super()._get_paradigm(paradigm, subject_mapping)

        # Map each trial's orientation to an index into unique_orientations
        ori_vals = paradigm_['orientation']
        stimulus_ix = np.searchsorted(self.unique_orientations_deg, ori_vals)
        paradigm_['stimulus_ix'] = stimulus_ix.astype(int)

        # Map each trial's mapping condition to an index
        if isinstance(paradigm, pd.DataFrame) and 'mapping' in paradigm.columns:
            mappings = list(self.value_on_ori_grid.keys())
            mapping_vals = paradigm['mapping'].astype(str).values
            paradigm_['mapping_ix'] = np.array([mappings.index(str(m)) for m in mapping_vals], dtype=int)
        else:
            paradigm_['mapping_ix'] = np.zeros(len(ori_vals), dtype=int)

        return paradigm_

    def _compute_trial_distributions(self, model_inputs):
        """Compute per-trial response PDF on the value grid."""
        kappa_r = model_inputs['kappa_r']
        subject_ix = model_inputs['subject_ix']

        model = pm.Model.get_context()
        stimulus_ix = model['stimulus_ix']
        mapping_ix = model['mapping_ix']

        ori_grid = pt.as_tensor_variable(self.ori_grid)
        rep_grid = pt.as_tensor_variable(self.rep_grid)
        val_grid = pt.as_tensor_variable(self.val_grid)
        if self.fit_prior_weight:
            # Prior peakedness is fitted, so the efficient-coding transform it
            # induces (and therefore where each stimulus lands in
            # representational space) has to be rebuilt inside the graph.
            w = model_inputs['prior_weight'][:, None]                # (S, 1)
            ori_prior = orientation_prior_pt(ori_grid[None, :], w)   # (S, O)
            ori_prior = ori_prior / (pt.sum(ori_prior, axis=-1, keepdims=True)
                                     * self.d_ori + 1e-30)
            ori_cdf = _efficient_cdf_pt(ori_prior, self.d_ori)       # (S, O)
            encoded_locs = _interp_pt(
                pt.as_tensor_variable(self.unique_orientations_rad),
                2 * np.pi, ori_cdf)                                  # (S, K)
        else:
            ori_prior = pt.as_tensor_variable(self.ori_prior)[None, :]
            ori_cdf = pt.as_tensor_variable(self.ori_cdf)[None, :]
            encoded_locs = pt.as_tensor_variable(
                self.encoded_stimulus_locs)[None, :]

        mappings = list(self.value_on_ori_grid.keys())
        G_on_grid = pt.as_tensor_variable(
            np.stack([self.value_on_ori_grid[m] for m in mappings]))
        G_ext = pt.as_tensor_variable(
            np.stack([self.value_on_ori_grid_ext[m] for m in mappings]))

        d_ori = self.d_ori
        d_rep = self.d_rep
        d_val = self.d_val

        if kappa_r.ndim == 0:
            kappa_r = kappa_r[None]

        # Step 1: Perceptual encoding — p(m_s | theta_0)
        # exp(k*(cos - 1)) rather than exp(k*cos)/(2 pi i0(k)): the normaliser is
        # constant over every axis that is renormalised downstream, so it cancels
        # exactly, while exp(k*cos) reaches 1.1e36 at the paper's top fitted
        # kappa of 83 and overflows outright in float32 above kappa ~ 89.
        p_ms = pt.exp(
            kappa_r[:, None, None] * (ptm.cos(rep_grid[None, None, :] - encoded_locs[:, :, None]) - 1.0)
        )  # (S, K, M), unnormalised

        # Step 2: Bayesian decoding
        likelihood = pt.exp(
            kappa_r[:, None, None] * (ptm.cos(rep_grid[None, None, :] - ori_cdf[:, :, None]) - 1.0)
        )  # (S, O, M), unnormalised
        posterior = likelihood * ori_prior[:, :, None]
        if self.no_seam_crossing:
            posterior = posterior * _seam_mask(ori_cdf, rep_grid, self.d_ori)
        posterior = posterior / (pt.sum(posterior, axis=1, keepdims=True) * d_ori + 1e-30)

        # Steps 3-4: q_per-optimal perceptual estimate, then v_hat = G(theta_hat)
        v_hat = _value_from_theta_hat(posterior, ori_grid, G_ext, d_ori,
                                      q_per=self.q_per,
                                      n_cand=self.grid_resolution)  # (C, S, M)

        # Step 5: Pushforward to value grid
        h_val = d_val * 0.75
        val_dists = (v_hat[:, :, :, None] - val_grid[None, None, None, :]) ** 2
        val_weights = pt.exp(-val_dists / (2 * h_val ** 2))
        val_weights = val_weights / (pt.sum(val_weights, axis=-1, keepdims=True) + 1e-30)

        # (1, S, K, M) @ (C, S, M, V) -> (C, S, K, V).  matmul rather than
        # broadcast-and-sum: the latter materialises a rank-5 (C, S, K, M, V)
        # intermediate that both operands of must stay live for the backward
        # pass.  Verified bit-comparable (max abs diff 4e-16).
        p_response = pt.matmul(p_ms[None], val_weights) * d_rep  # (C, S, K, V)
        p_response = p_response / (pt.sum(p_response, axis=-1, keepdims=True) * d_val + 1e-30)

        # Gather per-trial distributions
        return p_response[mapping_ix, subject_ix, stimulus_ix, :]  # (n_trials, V)

    def _get_response_grid(self):
        return self.val_grid


class EfficientValuationModel(EstimationBaseModel):
    """Efficient coding and Bayesian decoding operate only in value space.

    Perception is veridical: v0 = G(theta_0). Then v0 is encoded with
    truncated Gaussian noise in efficiently-coded value space, and decoded
    via Bayesian posterior mean.

    Free parameter: sigma_rep (value-space noise SD).
    """

    paradigm_keys = ['orientation']
    base_parameters = ['sigma_rep']

    def __init__(self, paradigm=None, grid_resolution=101, **kwargs):
        super().__init__(paradigm, grid_resolution=grid_resolution, **kwargs)

    def get_free_parameters(self):
        return {
            'sigma_rep': {'mu_intercept': 0.5, 'sigma_intercept': 1.0,
                          'transform': 'softplus'},
        }

    def get_model_inputs(self, parameters):
        model = pm.Model.get_context()
        return {
            'sigma_rep': self.subjectwise('sigma_rep'),
            'orientation': model['orientation'],
            'response': model['response'],
            'subject_ix': model['subject_ix'],
        }

    def _setup_grids(self, paradigm):
        """Precompute grids for value-only model."""
        N = self.grid_resolution

        self.val_grid = np.linspace(V_MIN, V_MAX, N)
        self.d_val = self.val_grid[1] - self.val_grid[0]

        # Rep grid in value space
        self.val_rep_grid = np.linspace(V_MIN, V_MAX, N)
        self.d_val_rep = self.val_rep_grid[1] - self.val_rep_grid[0]

        # Unique stimuli and their true values per mapping condition
        mappings = list(paradigm['mapping'].astype(str).unique()) if 'mapping' in paradigm.columns else ['linear']
        self.unique_orientations_deg = np.sort(paradigm['orientation'].unique())

        self.true_values = {}
        for mapping in mappings:
            self.true_values[mapping] = orientation_to_value_np(
                self.unique_orientations_deg, mapping)

        # Value prior: derived from uniform orientation sampling pushed through G
        # For each mapping, the value prior is the density of values induced by
        # uniform orientation sampling through the mapping function
        self.value_priors = {}
        for mapping in mappings:
            ori_dense = np.linspace(0, 180, 1000)
            vals_dense = orientation_to_value_np(ori_dense, mapping)
            # Histogram to estimate value prior
            counts, edges = np.histogram(vals_dense, bins=N, range=(V_MIN, V_MAX), density=True)
            # Interpolate to grid centers
            centers = (edges[:-1] + edges[1:]) / 2
            self.value_priors[mapping] = np.interp(self.val_grid, centers, counts)
            # Normalize
            self.value_priors[mapping] /= np_trapz(self.value_priors[mapping], self.val_grid)

        # Efficient coding CDF in value space for each mapping
        self.val_cdfs = {}
        for mapping in mappings:
            prior = self.value_priors[mapping]
            cdf = np.zeros(N)
            cdf[1:] = np.cumsum((prior[:-1] + prior[1:]) / 2 * self.d_val)
            cdf = V_MIN + cdf / cdf[-1] * (V_MAX - V_MIN)
            self.val_cdfs[mapping] = cdf

    def build_estimation_model(self, data=None, coords=None, hierarchical=True,
                               save_p_choice=False, flat_prior=False, paradigm=None):
        if data is None:
            data = paradigm if paradigm is not None else self.paradigm
        self._setup_grids(data)
        super().build_estimation_model(data, coords, hierarchical, save_p_choice, flat_prior)

    def _get_paradigm(self, paradigm=None, subject_mapping=None):
        paradigm_ = super()._get_paradigm(paradigm, subject_mapping)

        ori_vals = paradigm_['orientation']
        stimulus_ix = np.searchsorted(self.unique_orientations_deg, ori_vals)
        paradigm_['stimulus_ix'] = stimulus_ix.astype(int)

        if isinstance(paradigm, pd.DataFrame) and 'mapping' in paradigm.columns:
            mappings = list(self.true_values.keys())
            paradigm_['mapping_ix'] = np.array(
                [mappings.index(str(m)) for m in paradigm['mapping'].astype(str).values], dtype=int)
        else:
            paradigm_['mapping_ix'] = np.zeros(len(ori_vals), dtype=int)

        return paradigm_

    def _compute_trial_distributions(self, model_inputs):
        sigma_rep = model_inputs['sigma_rep']
        subject_ix = model_inputs['subject_ix']

        model = pm.Model.get_context()
        stimulus_ix = model['stimulus_ix']
        mapping_ix = model['mapping_ix']

        val_grid = pt.as_tensor_variable(self.val_grid)
        d_val = self.d_val

        if sigma_rep.ndim == 0:
            sigma_rep = sigma_rep[None]

        mappings = list(self.true_values.keys())
        n_conditions = len(mappings)
        K = len(self.unique_orientations_deg)
        N = self.grid_resolution

        # For each condition, build the value-stage computation
        # True values per condition: (C, K)
        true_vals = pt.as_tensor_variable(
            np.stack([self.true_values[m] for m in mappings]))

        # Value CDF per condition: (C, N)
        val_cdfs = pt.as_tensor_variable(
            np.stack([self.val_cdfs[m] for m in mappings]))

        # Value prior per condition: (C, N)
        val_priors = pt.as_tensor_variable(
            np.stack([self.value_priors[m] for m in mappings]))

        # ---- Step 1: Value encoding ----
        # Encoded location = F_val(v0) where v0 = G(theta_0)
        # true_vals: (C, K) -> encoded via CDF interpolation
        # For now, use soft lookup: for each true value, find its position on val_grid,
        # then interpolate the CDF
        # Approximate: encoded_loc[c, k] = interp(true_vals[c,k], val_grid, val_cdfs[c,:])
        # In pytensor, use soft interpolation
        h_val = d_val * 0.5
        # true_vals: (C, K), val_grid: (N,) -> (C, K, N)
        w = pt.exp(-((true_vals[:, :, None] - val_grid[None, None, :]) ** 2) / (2 * h_val**2))
        w = w / (pt.sum(w, axis=-1, keepdims=True) + 1e-30)
        encoded_locs = pt.sum(w * val_cdfs[:, None, :], axis=-1)  # (C, K)

        # ---- Step 2: Sensory noise in value rep space ----
        # p(m_v | v0) = TruncGauss(m_v; encoded_loc, sigma_rep^2, [V_MIN, V_MAX])
        # Approximate with Gaussian, then normalize on grid
        # rep_grid = val_grid (same grid for value representation)
        # Shape: (S, C, K, N)
        rep_diffs = val_grid[None, None, None, :] - encoded_locs[None, :, :, None]  # (1, C, K, N)
        p_mv = pt.exp(-0.5 * (rep_diffs / sigma_rep[:, None, None, None]) ** 2)
        # Normalize (truncation + normalization)
        p_mv = p_mv / (pt.sum(p_mv, axis=-1, keepdims=True) * d_val + 1e-30)

        # ---- Step 3: Bayesian decoding in value space ----
        # likelihood: p(m | v) for all v on val_grid
        # Shape: (S, C, V_decode, M_rep) = (S, C, N, N)
        # val_cdfs: (C, N) for encoding each hypothesized v
        val_encoded = val_cdfs  # (C, N) — F_val(v) for each v on grid
        rep_diffs_decode = val_grid[None, None, None, :] - val_encoded[None, :, :, None]
        # Wait, this is (1, C, N_val, N_rep) but we need to match m on rep_grid
        # Actually m IS on val_grid too (same grid)
        # p(m | v) = TruncGauss(m; F_val(v), sigma_rep^2)
        decode_likelihood = pt.exp(
            -0.5 * ((val_grid[None, None, None, :] - val_encoded[None, :, :, None]) / sigma_rep[:, None, None, None]) ** 2
        )  # (S, C, N_val, N_rep)
        decode_likelihood = decode_likelihood / (pt.sum(decode_likelihood, axis=-1, keepdims=True) * d_val + 1e-30)

        # Posterior: p(v | m) ∝ p(m | v) * p_val(v)
        # decode_likelihood: (S, C, N_val, N_rep), val_priors: (C, N_val)
        posterior_v = decode_likelihood * val_priors[None, :, :, None]  # (S, C, N_val, N_rep)
        posterior_v = posterior_v / (pt.sum(posterior_v, axis=2, keepdims=True) * d_val + 1e-30)

        # ---- Step 4: Posterior mean -> v_hat(m) ----
        v_hat = pt.sum(posterior_v * val_grid[None, None, :, None] * d_val, axis=2)  # (S, C, N_rep)

        # ---- Step 5: Pushforward to response grid ----
        h_push = d_val * 0.75
        val_dists = (v_hat[:, :, :, None] - val_grid[None, None, None, :]) ** 2  # (S, C, M, V)
        val_weights = pt.exp(-val_dists / (2 * h_push ** 2))
        val_weights = val_weights / (pt.sum(val_weights, axis=-1, keepdims=True) + 1e-30)

        # p_response: (S, C, K, V)
        # For each stimulus k: Σ_m val_weights[s,c,m,v] * p_mv[s,c,k,m] * d_val
        p_response = pt.sum(
            val_weights[:, :, None, :, :] * p_mv[:, :, :, :, None] * d_val,
            axis=3)  # (S, C, K, V)

        p_response = p_response / (pt.sum(p_response, axis=-1, keepdims=True) * d_val + 1e-30)

        # ---- Step 6: Per-trial distribution ----
        trial_dist = p_response[subject_ix, mapping_ix, stimulus_ix, :]  # (n_trials, V)

        return trial_dist

    def _get_response_grid(self):
        return self.val_grid


class SequentialEfficientCodingModel(EfficientPerceptionModel):
    """Sequential efficient coding: both perception and valuation stages.

    Stage 1: Orientation efficiently encoded + Bayesian decoded -> theta_hat
    Stage 2: G(theta_hat) -> value, then efficiently encoded + Bayesian decoded -> v_hat

    Perceptual uncertainty from stage 1 propagates into stage 2 via
    marginalization (Eq. 11 in paper).

    Free parameters: kappa_r (perceptual noise), sigma_rep (value noise).
    """

    base_parameters = ['kappa_r', 'sigma_rep']

    def __init__(self, paradigm=None, perceptual_prior='long_term',
                 grid_resolution=101, rep_grid_resolution=None, q_per=Q_PER,
                 fit_prior_weight=False, no_seam_crossing=False, **kwargs):
        super().__init__(paradigm, perceptual_prior=perceptual_prior,
                         grid_resolution=grid_resolution,
                         rep_grid_resolution=rep_grid_resolution,
                         q_per=q_per, fit_prior_weight=fit_prior_weight,
                         no_seam_crossing=no_seam_crossing, **kwargs)

    def get_free_parameters(self):
        pars = {'kappa_r': kappa_r_prior(self.grid_resolution),
                'sigma_rep': SIGMA_REP_PRIOR}
        if self.fit_prior_weight:
            pars['prior_weight'] = PRIOR_WEIGHT_PRIOR
        return pars

    def get_model_inputs(self, parameters):
        model = pm.Model.get_context()
        return {
            'kappa_r': self.subjectwise('kappa_r'),
            'sigma_rep': self.subjectwise('sigma_rep'),
            **({'prior_weight': self.subjectwise('prior_weight')}
               if self.fit_prior_weight else {}),
            'orientation': model['orientation'],
            'response': model['response'],
            'subject_ix': model['subject_ix'],
        }

    def _setup_grids(self, paradigm):
        """Set up grids for both perceptual and value stages."""
        # First set up perceptual grids
        super()._setup_grids(paradigm)

        # Then add value-stage grids
        mappings = list(paradigm['mapping'].astype(str).unique()) if 'mapping' in paradigm.columns else ['linear']

        # Value prior per condition (induced by uniform orientation sampling through G)
        self.value_priors = {}
        N = self.grid_resolution
        for mapping in mappings:
            ori_dense = np.linspace(0, 180, 1000)
            vals_dense = orientation_to_value_np(ori_dense, mapping)
            counts, edges = np.histogram(vals_dense, bins=N, range=(V_MIN, V_MAX), density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            self.value_priors[mapping] = np.interp(self.val_grid, centers, counts)
            self.value_priors[mapping] /= np_trapz(self.value_priors[mapping], self.val_grid)

        # Value CDF per condition
        self.val_cdfs = {}
        for mapping in mappings:
            prior = self.value_priors[mapping]
            cdf = np.zeros(N)
            cdf[1:] = np.cumsum((prior[:-1] + prior[1:]) / 2 * self.d_val)
            cdf = V_MIN + cdf / cdf[-1] * (V_MAX - V_MIN)
            self.val_cdfs[mapping] = cdf

    def _compute_trial_distributions(self, model_inputs):
        """Full sequential model: perception -> value encoding -> value decoding."""
        kappa_r = model_inputs['kappa_r']
        sigma_rep = model_inputs['sigma_rep']
        subject_ix = model_inputs['subject_ix']

        model = pm.Model.get_context()
        stimulus_ix = model['stimulus_ix']
        mapping_ix = model['mapping_ix']

        # Constants
        ori_grid = pt.as_tensor_variable(self.ori_grid)
        rep_grid = pt.as_tensor_variable(self.rep_grid)
        val_grid = pt.as_tensor_variable(self.val_grid)
        if self.fit_prior_weight:
            # Prior peakedness is fitted, so the efficient-coding transform it
            # induces (and therefore where each stimulus lands in
            # representational space) has to be rebuilt inside the graph.
            w = model_inputs['prior_weight'][:, None]                # (S, 1)
            ori_prior = orientation_prior_pt(ori_grid[None, :], w)   # (S, O)
            ori_prior = ori_prior / (pt.sum(ori_prior, axis=-1, keepdims=True)
                                     * self.d_ori + 1e-30)
            ori_cdf = _efficient_cdf_pt(ori_prior, self.d_ori)       # (S, O)
            encoded_locs = _interp_pt(
                pt.as_tensor_variable(self.unique_orientations_rad),
                2 * np.pi, ori_cdf)                                  # (S, K)
        else:
            ori_prior = pt.as_tensor_variable(self.ori_prior)[None, :]
            ori_cdf = pt.as_tensor_variable(self.ori_cdf)[None, :]
            encoded_locs = pt.as_tensor_variable(
                self.encoded_stimulus_locs)[None, :]

        mappings = list(self.value_on_ori_grid.keys())
        G_on_grid = pt.as_tensor_variable(
            np.stack([self.value_on_ori_grid[m] for m in mappings]))
        G_ext = pt.as_tensor_variable(
            np.stack([self.value_on_ori_grid_ext[m] for m in mappings]))

        val_cdfs = pt.as_tensor_variable(
            np.stack([self.val_cdfs[m] for m in mappings]))
        val_priors = pt.as_tensor_variable(
            np.stack([self.value_priors[m] for m in mappings]))

        d_ori = self.d_ori
        d_rep = self.d_rep
        d_val = self.d_val
        N = self.grid_resolution
        M = self.rep_grid_resolution

        if kappa_r.ndim == 0:
            kappa_r = kappa_r[None]
        if sigma_rep.ndim == 0:
            sigma_rep = sigma_rep[None]

        # ==== STAGE 1: Perceptual encoding + decoding ====
        # (Same as EfficientPerceptionModel steps 1-4)

        # p(m_s | theta_0) for each subject and unique stimulus
        p_ms = pt.exp(
            kappa_r[:, None, None] * (ptm.cos(rep_grid[None, None, :] - encoded_locs[:, :, None]) - 1.0)
        )  # (S, K, M), unnormalised

        # Bayesian decoding
        likelihood_ori = pt.exp(
            kappa_r[:, None, None] * (ptm.cos(rep_grid[None, None, :] - ori_cdf[:, :, None]) - 1.0)
        )  # (S, O, M), unnormalised
        posterior_ori = likelihood_ori * ori_prior[:, :, None]
        if self.no_seam_crossing:
            posterior_ori = posterior_ori * _seam_mask(ori_cdf, rep_grid, self.d_ori)
        posterior_ori = posterior_ori / (pt.sum(posterior_ori, axis=1, keepdims=True) * d_ori + 1e-30)

        # q_per-optimal perceptual estimate, then v_per(m) = G(theta_hat(m))
        v_per = _value_from_theta_hat(posterior_ori, ori_grid, G_ext, d_ori,
                                      q_per=self.q_per,
                                      n_cand=self.grid_resolution)  # (C, S, M)

        # Pushforward to value grid: p(v_per | theta_0)
        h_val_push = d_val * 0.75
        val_dists_per = (v_per[:, :, :, None] - val_grid[None, None, None, :]) ** 2  # (C, S, M, V)
        val_weights_per = pt.exp(-val_dists_per / (2 * h_val_push ** 2))
        val_weights_per = val_weights_per / (pt.sum(val_weights_per, axis=-1, keepdims=True) + 1e-30)

        # p_v_per: (C, S, K, V) = Σ_m val_weights_per[c,s,m,v] * p_ms[s,k,m] * d_rep
        # (1, S, K, M) @ (C, S, M, V) -> (C, S, K, V); see the note above.
        p_v_per = pt.matmul(p_ms[None], val_weights_per) * d_rep
        p_v_per = p_v_per / (pt.sum(p_v_per, axis=-1, keepdims=True) * d_val + 1e-30)

        # ==== STAGE 2: Value encoding + decoding ====
        # For each value v on val_grid: encode -> add noise -> decode

        # Encoded value locations: F_val(v) for each condition
        # val_cdfs: (C, N), these map each v on val_grid to its encoded position

        # p(m_v | v) = TruncGauss(m_v; F_val(v), sigma_rep^2)
        # val_cdfs: (C, N_v), val_grid: (N_rep) — same grid for both
        # Shape: (S, C, N_source_v, N_rep)
        rep_diffs = val_grid[None, None, None, :] - val_cdfs[None, :, :, None]  # (1, C, N, N)
        p_mv_given_v = pt.exp(-0.5 * (rep_diffs / sigma_rep[:, None, None, None]) ** 2)
        p_mv_given_v = p_mv_given_v / (pt.sum(p_mv_given_v, axis=-1, keepdims=True) * d_val + 1e-30)

        # Bayesian decoding in value space
        # p(v | m_v) ∝ p(m_v | v) * p_val(v)
        # decode_likelihood: (S, C, N_v, N_rep)
        posterior_val = p_mv_given_v * val_priors[None, :, :, None]  # (S, C, N_v, N_rep)
        posterior_val = posterior_val / (pt.sum(posterior_val, axis=2, keepdims=True) * d_val + 1e-30)

        # Posterior mean: v_hat(m_v) = Σ v * p(v|m_v)
        v_hat = pt.sum(posterior_val * val_grid[None, None, :, None] * d_val, axis=2)  # (S, C, N_rep)

        # Pushforward v_hat(m_v) to response grid
        h_push2 = d_val * 0.75
        val_dists2 = (v_hat[:, :, :, None] - val_grid[None, None, None, :]) ** 2  # (S, C, N_rep, V)
        val_weights2 = pt.exp(-val_dists2 / (2 * h_push2 ** 2))
        val_weights2 = val_weights2 / (pt.sum(val_weights2, axis=-1, keepdims=True) + 1e-30)

        # For each source value v_per on the grid:
        # p(v_hat | v_per) = Σ_m val_weights2[s,c,m,v] * p_mv_given_v[s,c,v_per_ix,m] * d_val
        # But we need to integrate over v_per weighted by p_v_per

        # p_response(v_hat | theta_0) = Σ_v_per p(v_hat | v_per) * p(v_per | theta_0)
        # p(v_hat | v_per) for all v_per on grid: use the value-stage computation
        # p_mv_given_v: (S, C, N_source, N_rep) — for each source v, distribution of m_v
        # val_weights2: (S, C, N_rep, V) — for each m_v, pushforward to response

        # p(response_v | source_v) = Σ_m p(m_v | source_v) * pushforward_weights(m_v)
        # Shape: (S, C, N_source, V)
        # (S, C, N_source, N_rep) x (S, C, N_rep, V) -> (S, C, N_source, V).
        # batched_dot rather than broadcast-and-sum: the latter builds an
        # (S, C, N_source, N_rep, V) intermediate, O(N^3) per subject.
        _S, _C = p_mv_given_v.shape[0], p_mv_given_v.shape[1]
        _Ns, _Nr = p_mv_given_v.shape[2], p_mv_given_v.shape[3]
        _V = val_weights2.shape[3]
        p_resp_given_source = pt.matmul(p_mv_given_v, val_weights2) * d_val

        # Marginalize over v_per:
        # p_response(v_hat | theta_0) = Σ_v_per p(v_hat | v_per) * p(v_per | theta_0) * d_val
        # p_v_per: (C, S, K, V_source), p_resp_given_source: (S, C, V_source, V_response)
        # Need to align: (S, C, K, V_source) x (S, C, V_source, V_response) -> (S, C, K, V_response)
        p_v_per_reordered = p_v_per.dimshuffle(1, 0, 2, 3)  # (S, C, K, V_source)
        # (S, C, K, Vs) @ (S, C, Vs, Vr) -> (S, C, K, Vr); see the note above.
        p_response = pt.matmul(p_v_per_reordered, p_resp_given_source) * d_val

        p_response = p_response / (pt.sum(p_response, axis=-1, keepdims=True) * d_val + 1e-30)

        # ---- Per-trial distribution ----
        # p_response: (S, C, K, V)
        trial_dist = p_response[subject_ix, mapping_ix, stimulus_ix, :]

        return trial_dist


class CategoricalSequentialModel(SequentialEfficientCodingModel):
    """Sequential model with cardinal categorical stabilization (paper, Fig. 6).

    Perceptual and value encoding/decoding are untouched; a hard category gate
    is applied to the final value-estimate distribution p(v_hat | phi_0).
    Stimuli fall into three categories relative to the 90 deg cardinal --
    below, at, above -- where "at" is within one model grid step of 90 deg.  In
    value space the middle category is [v_mid - delta, v_mid + delta] with
    v_mid = 22 CHF and delta = 0.25 CHF, and the outer categories are the values
    below and above that interval.  Mass outside the implied category is set to
    zero and the rest renormalised.

    Descriptive, and deliberately free-parameter-free: it is a proxy for the
    regime in which precision near the cardinal is high enough to remove
    category-level confusions, not a mechanism.  Motivated by the localized
    collapse of response variability at 90 deg, which every ungated
    architecture gets backwards in at least one mapping (it predicts a variance
    *peak* there for any mapping whose slope is steep at the cardinal).
    """

    V_MID = 22.0
    DELTA = 0.25

    def _setup_grids(self, paradigm):
        super()._setup_grids(paradigm)

        step_deg = 180.0 / self.grid_resolution
        ori_deg = np.rad2deg(self.unique_orientations_rad) / 2.0          # (K,)
        v = self.val_grid                                                 # (V,)

        at_cardinal = np.abs(ori_deg - 90.0) <= step_deg
        below = ori_deg < 90.0 - step_deg

        mid = np.abs(v - self.V_MID) <= self.DELTA
        low = v < self.V_MID - self.DELTA
        high = v > self.V_MID + self.DELTA

        mask = np.where(at_cardinal[:, None], mid[None, :],
                        np.where(below[:, None], low[None, :], high[None, :]))
        self.category_mask = mask.astype(float)                           # (K, V)

    def _compute_trial_distributions(self, model_inputs):
        trial_dist = super()._compute_trial_distributions(model_inputs)   # (T, V)
        model = pm.Model.get_context()
        stimulus_ix = model['stimulus_ix']
        mask = pt.as_tensor_variable(self.category_mask)[stimulus_ix]     # (T, V)
        gated = trial_dist * mask
        return gated / (pt.sum(gated, axis=-1, keepdims=True) * self.d_val + 1e-30)
