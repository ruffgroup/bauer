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


def orientation_to_value_np(orientation_deg, mapping='linear'):
    """Map orientation (degrees) to value (CHF) via linear interpolation.

    Works on numpy arrays. For precomputing G(theta) on a grid.
    """
    return np.interp(orientation_deg, MAPPING_ORIENTATIONS_DEG,
                     MAPPING_VALUES[mapping])


# ============================================================================
# Orientation priors
# ============================================================================

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
                 grid_resolution=101, rep_grid_resolution=None, **kwargs):
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
        super().__init__(paradigm, grid_resolution=grid_resolution, **kwargs)

    def get_free_parameters(self):
        return {
            'kappa_r': {'mu_intercept': 3.0, 'sigma_intercept': 1.0,
                        'transform': 'softplus'},
        }

    def get_model_inputs(self, parameters):
        model = pm.Model.get_context()
        return {
            'kappa_r': parameters['kappa_r'],
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
        mappings = paradigm['mapping'].unique() if 'mapping' in paradigm.columns else ['linear']
        self.value_on_ori_grid = {}
        for mapping in mappings:
            # Convert ori_grid (radians, doubled) back to degrees [0, 180]
            ori_deg = self.ori_grid / (2 * np.pi) * 180
            self.value_on_ori_grid[mapping] = orientation_to_value_np(ori_deg, mapping)

        # Unique stimulus orientations (in degrees) from paradigm
        self.unique_orientations_deg = np.sort(paradigm['orientation'].unique())
        # Convert to doubled-angle radians
        self.unique_orientations_rad = self.unique_orientations_deg * np.pi / 180.0 * 2

        # For each unique orientation, find F_ori(theta_0) via interpolation
        self.encoded_stimulus_locs = np.interp(
            self.unique_orientations_rad, self.ori_grid, self.ori_cdf)

    def build_estimation_model(self, data=None, coords=None, hierarchical=True,
                               save_p_choice=False, flat_prior=False):
        if data is None:
            data = self.paradigm
        self._setup_grids(data)
        super().build_estimation_model(data, coords, hierarchical, save_p_choice, flat_prior)

    def _get_paradigm(self, paradigm=None, subject_mapping=None):
        paradigm_ = super()._get_paradigm(paradigm, subject_mapping)

        # Map each trial's orientation to an index into unique_orientations
        ori_vals = paradigm_['orientation']
        stimulus_ix = np.searchsorted(self.unique_orientations_deg, ori_vals)
        paradigm_['stimulus_ix'] = stimulus_ix.astype(int)

        # Map each trial's mapping condition to an index
        if 'mapping' in paradigm.columns if isinstance(paradigm, pd.DataFrame) else False:
            mappings = list(self.value_on_ori_grid.keys())
            mapping_vals = paradigm['mapping'].values
            paradigm_['mapping_ix'] = np.array([mappings.index(m) for m in mapping_vals], dtype=int)
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
        ori_prior = pt.as_tensor_variable(self.ori_prior)
        ori_cdf = pt.as_tensor_variable(self.ori_cdf)
        encoded_locs = pt.as_tensor_variable(self.encoded_stimulus_locs)

        mappings = list(self.value_on_ori_grid.keys())
        G_on_grid = pt.as_tensor_variable(
            np.stack([self.value_on_ori_grid[m] for m in mappings]))

        d_ori = self.d_ori
        d_rep = self.d_rep
        d_val = self.d_val

        if kappa_r.ndim == 0:
            kappa_r = kappa_r[None]

        # Step 1: Perceptual encoding — p(m_s | theta_0)
        p_ms = pt.exp(
            kappa_r[:, None, None] * ptm.cos(rep_grid[None, None, :] - encoded_locs[None, :, None])
        ) / (2 * np.pi * ptm.i0(kappa_r[:, None, None]))  # (S, K, M)

        # Step 2: Bayesian decoding
        likelihood = pt.exp(
            kappa_r[:, None, None] * ptm.cos(rep_grid[None, None, :] - ori_cdf[None, :, None])
        ) / (2 * np.pi * ptm.i0(kappa_r[:, None, None]))  # (S, O, M)
        posterior = likelihood * ori_prior[None, :, None]
        posterior = posterior / (pt.sum(posterior, axis=1, keepdims=True) * d_ori + 1e-30)

        # Step 3: Circular posterior mean
        sin_mean = pt.sum(posterior * ptm.sin(ori_grid)[None, :, None] * d_ori, axis=1)
        cos_mean = pt.sum(posterior * ptm.cos(ori_grid)[None, :, None] * d_ori, axis=1)
        theta_hat = pt.mod(ptm.arctan2(sin_mean, cos_mean) + 2 * np.pi, 2 * np.pi)

        # Step 4: Value from perceptual estimate via soft interpolation
        h = d_ori * 0.5
        cos_dist = 1 - ptm.cos(theta_hat[:, :, None] - ori_grid[None, None, :])
        weights = pt.exp(-cos_dist / (2 * h ** 2))
        weights = weights / (pt.sum(weights, axis=-1, keepdims=True) + 1e-30)
        v_hat = pt.sum(weights[None, :, :, :] * G_on_grid[:, None, None, :], axis=-1)  # (C, S, M)

        # Step 5: Pushforward to value grid
        h_val = d_val * 0.75
        val_dists = (v_hat[:, :, :, None] - val_grid[None, None, None, :]) ** 2
        val_weights = pt.exp(-val_dists / (2 * h_val ** 2))
        val_weights = val_weights / (pt.sum(val_weights, axis=-1, keepdims=True) + 1e-30)

        p_response = pt.sum(
            val_weights[:, :, None, :, :] * p_ms[None, :, :, :, None] * d_rep,
            axis=3)  # (C, S, K, V)
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
            'sigma_rep': parameters['sigma_rep'],
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
        mappings = paradigm['mapping'].unique() if 'mapping' in paradigm.columns else ['linear']
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
                               save_p_choice=False, flat_prior=False):
        if data is None:
            data = self.paradigm
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
                [mappings.index(m) for m in paradigm['mapping'].values], dtype=int)
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
                 grid_resolution=101, rep_grid_resolution=None, **kwargs):
        super().__init__(paradigm, perceptual_prior=perceptual_prior,
                         grid_resolution=grid_resolution,
                         rep_grid_resolution=rep_grid_resolution, **kwargs)

    def get_free_parameters(self):
        return {
            'kappa_r': {'mu_intercept': 3.0, 'sigma_intercept': 1.0,
                        'transform': 'softplus'},
            'sigma_rep': {'mu_intercept': 0.5, 'sigma_intercept': 1.0,
                          'transform': 'softplus'},
        }

    def get_model_inputs(self, parameters):
        model = pm.Model.get_context()
        return {
            'kappa_r': parameters['kappa_r'],
            'sigma_rep': parameters['sigma_rep'],
            'orientation': model['orientation'],
            'response': model['response'],
            'subject_ix': model['subject_ix'],
        }

    def _setup_grids(self, paradigm):
        """Set up grids for both perceptual and value stages."""
        # First set up perceptual grids
        super()._setup_grids(paradigm)

        # Then add value-stage grids
        mappings = paradigm['mapping'].unique() if 'mapping' in paradigm.columns else ['linear']

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
        ori_prior = pt.as_tensor_variable(self.ori_prior)
        ori_cdf = pt.as_tensor_variable(self.ori_cdf)
        encoded_locs = pt.as_tensor_variable(self.encoded_stimulus_locs)

        mappings = list(self.value_on_ori_grid.keys())
        G_on_grid = pt.as_tensor_variable(
            np.stack([self.value_on_ori_grid[m] for m in mappings]))

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
            kappa_r[:, None, None] * ptm.cos(rep_grid[None, None, :] - encoded_locs[None, :, None])
        ) / (2 * np.pi * ptm.i0(kappa_r[:, None, None]))  # (S, K, M)

        # Bayesian decoding
        likelihood_ori = pt.exp(
            kappa_r[:, None, None] * ptm.cos(rep_grid[None, None, :] - ori_cdf[None, :, None])
        ) / (2 * np.pi * ptm.i0(kappa_r[:, None, None]))  # (S, O, M)
        posterior_ori = likelihood_ori * ori_prior[None, :, None]
        posterior_ori = posterior_ori / (pt.sum(posterior_ori, axis=1, keepdims=True) * d_ori + 1e-30)

        # Circular posterior mean
        sin_mean = pt.sum(posterior_ori * ptm.sin(ori_grid)[None, :, None] * d_ori, axis=1)
        cos_mean = pt.sum(posterior_ori * ptm.cos(ori_grid)[None, :, None] * d_ori, axis=1)
        theta_hat = pt.mod(ptm.arctan2(sin_mean, cos_mean) + 2 * np.pi, 2 * np.pi)  # (S, M)

        # Value from perceptual estimate: v_per(m) = G(theta_hat(m))
        h_ori = d_ori * 0.5
        cos_dist = 1 - ptm.cos(theta_hat[:, :, None] - ori_grid[None, None, :])
        weights_ori = pt.exp(-cos_dist / (2 * h_ori ** 2))
        weights_ori = weights_ori / (pt.sum(weights_ori, axis=-1, keepdims=True) + 1e-30)  # (S, M, O)

        # v_per for each condition: (C, S, M)
        v_per = pt.sum(weights_ori[None, :, :, :] * G_on_grid[:, None, None, :], axis=-1)

        # Pushforward to value grid: p(v_per | theta_0)
        h_val_push = d_val * 0.75
        val_dists_per = (v_per[:, :, :, None] - val_grid[None, None, None, :]) ** 2  # (C, S, M, V)
        val_weights_per = pt.exp(-val_dists_per / (2 * h_val_push ** 2))
        val_weights_per = val_weights_per / (pt.sum(val_weights_per, axis=-1, keepdims=True) + 1e-30)

        # p_v_per: (C, S, K, V) = Σ_m val_weights_per[c,s,m,v] * p_ms[s,k,m] * d_rep
        p_v_per = pt.sum(
            val_weights_per[:, :, None, :, :] * p_ms[None, :, :, :, None] * d_rep,
            axis=3)
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
        p_resp_given_source = pt.sum(
            p_mv_given_v[:, :, :, :, None] * val_weights2[:, :, None, :, :] * d_val,
            axis=3)  # (S, C, N_source, V)

        # Marginalize over v_per:
        # p_response(v_hat | theta_0) = Σ_v_per p(v_hat | v_per) * p(v_per | theta_0) * d_val
        # p_v_per: (C, S, K, V_source), p_resp_given_source: (S, C, V_source, V_response)
        # Need to align: (S, C, K, V_source) x (S, C, V_source, V_response) -> (S, C, K, V_response)
        p_v_per_reordered = p_v_per.dimshuffle(1, 0, 2, 3)  # (S, C, K, V_source)
        p_response = pt.sum(
            p_v_per_reordered[:, :, :, :, None] * p_resp_given_source[:, :, None, :, :] * d_val,
            axis=3)  # (S, C, K, V)

        p_response = p_response / (pt.sum(p_response, axis=-1, keepdims=True) * d_val + 1e-30)

        # ---- Per-trial distribution ----
        # p_response: (S, C, K, V)
        trial_dist = p_response[subject_ix, mapping_ix, stimulus_ix, :]

        return trial_dist


class CategoricalSequentialModel(SequentialEfficientCodingModel):
    """Sequential model with cardinal categorical stabilization.

    Same as SequentialEfficientCodingModel, but applies a category gate at the
    output: stimuli below/at/above 90 degrees are constrained to have value
    estimates in the corresponding category (below/at/above v_mid).
    """

    def _compute_trial_distributions(self, model_inputs):
        # Get the base sequential model's response distribution
        # Then apply the category gate
        # For now, delegate to parent and note that the gate
        # should be applied to p_response before computing log_prob

        # TODO: Implement category gate. For now, use parent.
        return super()._compute_trial_distributions(model_inputs)
