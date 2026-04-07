"""
Numerosity estimation models with flexible encoding functions.

Models for tasks where subjects estimate the numerosity of a dot cloud.
The generative process is:
    1. Encoding: n -> r ~ N(mu(n), nu^2)       [shared across conditions]
    2. Bayesian decoding: n_hat(r) = E[n | r]   [condition-specific prior]
    3. Motor noise: response ~ N(n_hat(r), sigma_motor^2)

Critically, when subjects perform the task under multiple conditions
(e.g., narrow [10,25] and wide [10,40] prior ranges), the encoding function
mu(n) and sensory noise nu are SHARED, while the Bayesian decoding prior
changes per condition. This cross-condition constraint is key to identifying
the model.

Data is loaded via bauer.utils.data.load_neuralpriors().
"""

import numpy as np
from numpy import trapezoid as np_trapz
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytensor.tensor.math as ptm
from .estimation import EstimationBaseModel


def bernstein_basis(t, degree):
    """Compute Bernstein basis polynomials of given degree at points t.

    Parameters
    ----------
    t : numpy array, shape (N,)
        Evaluation points in [0, 1].
    degree : int
        Polynomial degree.

    Returns
    -------
    basis : numpy array, shape (N, degree+1)
    """
    from scipy.special import comb
    n = degree
    k = np.arange(n + 1)
    coeffs = comb(n, k)
    basis = coeffs[np.newaxis, :] * (t[:, np.newaxis] ** k[np.newaxis, :]) * \
            ((1 - t[:, np.newaxis]) ** (n - k)[np.newaxis, :])
    return basis


# ============================================================================
# Condition definitions
# ============================================================================

RANGE_CONDITIONS = {
    'narrow': {'n_min': 10, 'n_max': 25},
    'wide': {'n_min': 10, 'n_max': 40},
}


class LogEncodingEstimationModel(EstimationBaseModel):
    """Numerosity estimation with fixed log encoding and condition-dependent priors.

    mu(n) = log(n), shared across conditions.
    nu (sensory noise) shared across conditions.
    sigma_motor (motor noise) shared across conditions.
    Bayesian decoding prior is Uniform[n_min_c, n_max_c] per condition c.

    Free parameters: nu, sigma_motor.
    """

    paradigm_keys = ['n']
    base_parameters = ['nu', 'sigma_motor']

    def __init__(self, paradigm=None, grid_resolution=101,
                 n_min=10, n_max=40, rep_grid_extent=4.0,
                 conditions=None, **kwargs):
        """
        Parameters
        ----------
        paradigm : pd.DataFrame
            Must have 'n' (stimulus), 'response' (estimate) columns.
            Must have 'range' column ('narrow'/'wide') for condition indexing.
        grid_resolution : int
            Grid points for stimulus and response grids.
        n_min, n_max : float
            Overall range of the stimulus/response grid (must span all conditions).
        conditions : dict or None
            {condition_name: {'n_min': ..., 'n_max': ...}}. If None, uses
            RANGE_CONDITIONS ('narrow' and 'wide').
        """
        self.n_min = n_min
        self.n_max = n_max
        self.rep_grid_extent = rep_grid_extent
        self.conditions = conditions or RANGE_CONDITIONS
        super().__init__(paradigm, grid_resolution=grid_resolution, **kwargs)

    def get_free_parameters(self):
        return {
            'nu': {'mu_intercept': -1.0, 'sigma_intercept': 1.0,
                   'transform': 'softplus'},
            'sigma_motor': {'mu_intercept': 0.0, 'sigma_intercept': 1.0,
                            'transform': 'softplus'},
        }

    def get_model_inputs(self, parameters):
        model = pm.Model.get_context()
        return {
            'nu': parameters['nu'],
            'sigma_motor': parameters['sigma_motor'],
            'n': model['n'],
            'response': model['response'],
            'subject_ix': model['subject_ix'],
        }

    def _setup_grids(self, paradigm):
        N = self.grid_resolution

        # Full stimulus/response grid spanning all conditions
        self.stim_grid = np.linspace(self.n_min, self.n_max, N)
        self.d_stim = self.stim_grid[1] - self.stim_grid[0]

        # Encoding: mu(n) = log(n) on the full grid
        self.mu_on_grid = np.log(self.stim_grid)

        # Representation grid
        mu_min = np.log(self.n_min)
        mu_max = np.log(self.n_max)
        mu_range = mu_max - mu_min
        R = N * 2
        self.rep_grid = np.linspace(
            mu_min - self.rep_grid_extent * mu_range * 0.2,
            mu_max + self.rep_grid_extent * mu_range * 0.2, R)
        self.d_rep = self.rep_grid[1] - self.rep_grid[0]

        # Condition-specific uniform priors on the SAME stim_grid
        # prior[c, i] = 1/(n_max_c - n_min_c) if n_min_c <= stim_grid[i] <= n_max_c, else 0
        self.condition_names = sorted(self.conditions.keys())
        n_cond = len(self.condition_names)
        self.priors = np.zeros((n_cond, N))
        for c_ix, cname in enumerate(self.condition_names):
            cond = self.conditions[cname]
            mask = (self.stim_grid >= cond['n_min']) & (self.stim_grid <= cond['n_max'])
            width = cond['n_max'] - cond['n_min']
            self.priors[c_ix, mask] = 1.0 / width

        # Unique stimuli across all conditions
        self.unique_stimuli = np.sort(paradigm['n'].unique().astype(float))

    def build_estimation_model(self, data=None, coords=None, hierarchical=True,
                               save_p_choice=False, flat_prior=False):
        if data is None:
            data = self.paradigm
        self._setup_grids(data)
        super().build_estimation_model(data, coords, hierarchical, save_p_choice, flat_prior)

    def _get_paradigm(self, paradigm=None, subject_mapping=None):
        paradigm_ = super()._get_paradigm(paradigm, subject_mapping)

        n_vals = paradigm_['n'].astype(float)
        stimulus_ix = np.searchsorted(self.unique_stimuli, n_vals)
        paradigm_['stimulus_ix'] = stimulus_ix.astype(int)

        # Condition index
        if isinstance(paradigm, pd.DataFrame) and 'range' in paradigm.columns:
            range_vals = paradigm['range'].values
        elif isinstance(paradigm, pd.DataFrame) and 'range' in paradigm.index.names:
            range_vals = paradigm.index.get_level_values('range').values
        else:
            range_vals = np.array(['narrow'] * len(n_vals))

        paradigm_['condition_ix'] = np.array(
            [self.condition_names.index(r) for r in range_vals], dtype=int)

        return paradigm_

    def _compute_trial_distributions(self, model_inputs):
        """Compute per-trial response PDF on the stimulus grid."""
        nu = model_inputs['nu']
        sigma_motor = model_inputs['sigma_motor']
        subject_ix = model_inputs['subject_ix']

        model = pm.Model.get_context()
        stimulus_ix = model['stimulus_ix']
        condition_ix = model['condition_ix']

        stim_grid = pt.as_tensor_variable(self.stim_grid)
        mu_on_grid = pt.as_tensor_variable(self.mu_on_grid)
        rep_grid = pt.as_tensor_variable(self.rep_grid)
        priors = pt.as_tensor_variable(self.priors)  # (C, N)

        d_stim = self.d_stim
        d_rep = self.d_rep

        unique_mu = pt.as_tensor_variable(np.log(self.unique_stimuli))

        if nu.ndim == 0:
            nu = nu[None]
        if sigma_motor.ndim == 0:
            sigma_motor = sigma_motor[None]

        # Step 1: Encoding (SHARED across conditions)
        p_r = pt.exp(-0.5 * ((rep_grid[None, None, :] - unique_mu[None, :, None]) / nu[:, None, None]) ** 2) \
              / (nu[:, None, None] * pt.sqrt(2 * np.pi))  # (S, K, R)

        # Step 2: Decoding (CONDITION-SPECIFIC prior)
        likelihood = pt.exp(-0.5 * ((rep_grid[None, None, :] - mu_on_grid[None, :, None]) / nu[:, None, None]) ** 2) \
                     / (nu[:, None, None] * pt.sqrt(2 * np.pi))  # (S, N, R)

        posterior = likelihood[:, None, :, :] * priors[None, :, :, None]  # (S, C, N, R)
        posterior = posterior / (pt.sum(posterior, axis=2, keepdims=True) * d_stim + 1e-30)

        n_hat = pt.sum(posterior * stim_grid[None, None, :, None] * d_stim, axis=2)  # (S, C, R)
        n_hat = pt.clip(n_hat, self.n_min, self.n_max)

        # Step 3: Motor noise (SHARED), truncated to [n_min, n_max]
        resp_grid = stim_grid
        p_resp_given_r = pt.exp(
            -0.5 * ((resp_grid[None, None, None, :] - n_hat[:, :, :, None]) / sigma_motor[:, None, None, None]) ** 2
        )
        # Truncate: zero out responses outside [n_min, n_max] and renormalize
        p_resp_given_r = p_resp_given_r / (pt.sum(p_resp_given_r, axis=-1, keepdims=True) + 1e-30)

        p_response = pt.sum(
            p_resp_given_r[:, :, None, :, :] * p_r[:, None, :, :, None] * d_rep,
            axis=3)  # (S, C, K, N)
        p_response = p_response / (pt.sum(p_response, axis=-1, keepdims=True) * d_stim + 1e-30)

        return p_response[subject_ix, condition_ix, stimulus_ix, :]  # (n_trials, N)

    def _get_response_grid(self):
        return self.stim_grid


class FlexibleEncodingEstimationModel(LogEncodingEstimationModel):
    """Numerosity estimation with flexible monotone encoding function.

    mu(n) is parameterized as a monotone Bernstein polynomial.
    Monotonicity enforced via cumulative softplus on increments.

    When ``condition_specific_encoding=True`` (default), each condition gets
    its own encoding function mu_c(n). This is essential for capturing how
    neural representations remap with the prior range (distributed range
    adaptation). Noise parameters (nu, sigma_motor) remain shared.

    When ``condition_specific_encoding=False``, a single encoding function
    is shared across conditions.

    Free parameters:
        - encoding increments per condition (or shared): define monotone mu(n)
        - nu: sensory noise SD (shared across conditions)
        - sigma_motor: motor noise SD (shared)
    """

    def __init__(self, paradigm=None, grid_resolution=101, n_poly=6,
                 n_min=10, n_max=40, condition_specific_encoding=True,
                 **kwargs):
        self.n_poly = n_poly
        self.condition_specific_encoding = condition_specific_encoding
        super().__init__(paradigm, grid_resolution=grid_resolution,
                         n_min=n_min, n_max=n_max, **kwargs)

    def _enc_key(self, cond_ix, inc_ix):
        """Parameter name for encoding increment."""
        if self.condition_specific_encoding:
            return f'enc_c{cond_ix}_inc_{inc_ix}'
        else:
            return f'enc_inc_{inc_ix}'

    def get_free_parameters(self):
        pars = super().get_free_parameters()
        if self.condition_specific_encoding:
            for c_ix in range(len(self.conditions)):
                for i in range(self.n_poly - 1):
                    pars[self._enc_key(c_ix, i)] = {
                        'mu_intercept': 0.5, 'sigma_intercept': 0.5,
                        'transform': 'identity'
                    }
        else:
            for i in range(self.n_poly - 1):
                pars[f'enc_inc_{i}'] = {
                    'mu_intercept': 0.5, 'sigma_intercept': 0.5,
                    'transform': 'identity'
                }
        return pars

    def get_model_inputs(self, parameters):
        model_inputs = super().get_model_inputs(parameters)
        if self.condition_specific_encoding:
            for c_ix in range(len(self.conditions)):
                for i in range(self.n_poly - 1):
                    key = self._enc_key(c_ix, i)
                    model_inputs[key] = parameters[key]
        else:
            for i in range(self.n_poly - 1):
                model_inputs[f'enc_inc_{i}'] = parameters[f'enc_inc_{i}']
        return model_inputs

    def _setup_grids(self, paradigm):
        super()._setup_grids(paradigm)
        # Bernstein basis matrices
        t = (self.stim_grid - self.n_min) / (self.n_max - self.n_min)
        self.bernstein_basis_matrix = bernstein_basis(t, self.n_poly - 1)
        t_unique = (self.unique_stimuli - self.n_min) / (self.n_max - self.n_min)
        self.bernstein_basis_unique = bernstein_basis(t_unique, self.n_poly - 1)

        # Override rep_grid: encoding is normalized to [0, 1], so rep space
        # spans [0, 1] with margin for noise tails
        margin = self.rep_grid_extent * 0.15
        R = self.grid_resolution * 2
        self.rep_grid = np.linspace(-margin, 1 + margin, R)
        self.d_rep = self.rep_grid[1] - self.rep_grid[0]

    def _build_coefficients(self, model_inputs, cond_ix=None):
        """Build monotone Bernstein coefficients from raw increments.

        The coefficients are normalized to [0, 1] so that the encoding
        function maps n -> [0, 1]. This removes the scale ambiguity
        between mu(n) and nu: the noise parameter nu is directly
        interpretable as noise relative to the unit representation range.

        Parameters
        ----------
        model_inputs : dict
        cond_ix : int or None
            If condition_specific_encoding, which condition's coefficients.
            If None and condition_specific, returns a list of all conditions.
        """
        if self.condition_specific_encoding:
            if cond_ix is not None:
                raw = pt.stack([model_inputs[self._enc_key(cond_ix, i)]
                                for i in range(self.n_poly - 1)])
                pos = pt.softplus(raw)
                unnormed = pt.concatenate([pt.zeros(1), pt.cumsum(pos)])
                return unnormed / (unnormed[-1] + 1e-10)  # normalize to [0, 1]
            else:
                return [self._build_coefficients(model_inputs, c)
                        for c in range(len(self.conditions))]
        else:
            raw = pt.stack([model_inputs[f'enc_inc_{i}']
                            for i in range(self.n_poly - 1)])
            pos = pt.softplus(raw)
            unnormed = pt.concatenate([pt.zeros(1), pt.cumsum(pos)])
            return unnormed / (unnormed[-1] + 1e-10)  # normalize to [0, 1]

    def get_encoding_function(self, pars, condition=None, n_points=200):
        """Evaluate the fitted encoding function mu(n) on a dense grid.

        Parameters
        ----------
        pars : dict
            Fitted parameters.
        condition : str or None
            Condition name (e.g., 'narrow', 'wide'). Required if
            condition_specific_encoding=True. If None and shared encoding,
            returns the single encoding function.
        n_points : int

        Returns
        -------
        n_grid : np.ndarray, shape (n_points,)
        mu_grid : np.ndarray, shape (n_points,)
        """
        from .utils.math import softplus_np

        if self.condition_specific_encoding:
            if condition is None:
                raise ValueError(
                    "condition_specific_encoding=True: pass condition='narrow' or 'wide'")
            c_ix = self.condition_names.index(condition)
            raw = np.array([float(pars[self._enc_key(c_ix, i)])
                            for i in range(self.n_poly - 1)])
        else:
            raw = np.array([float(pars[f'enc_inc_{i}'])
                            for i in range(self.n_poly - 1)])

        pos = softplus_np(raw)
        unnormed = np.concatenate([[0.0], np.cumsum(pos)])
        coefficients = unnormed / (unnormed[-1] + 1e-10)  # normalize to [0, 1]

        n_grid = np.linspace(self.n_min, self.n_max, n_points)
        t = (n_grid - self.n_min) / (self.n_max - self.n_min)
        basis = bernstein_basis(t, self.n_poly - 1)
        mu_grid = basis @ coefficients
        return n_grid, mu_grid

    def get_trialwise_variable(self, key):
        """Encoding increments are scalar (shared across subjects), not per-subject."""
        if key.startswith('enc_'):
            model = pm.Model.get_context()
            return model[key]
        return super().get_trialwise_variable(key)

    def build_estimation_model(self, data=None, coords=None, hierarchical=True,
                               save_p_choice=False, flat_prior=False):
        if data is None:
            data = self.paradigm
        self._setup_grids(data)

        if coords is None and hierarchical:
            assert 'subject' in data.index.names
            coords = {'subject': data.index.unique(level='subject')}

        with pm.Model(coords=coords) as self.estimation_model:
            paradigm = self._get_paradigm(paradigm=data)
            self.set_paradigm(paradigm)
            self.build_priors(hierarchical=hierarchical, flat_prior=flat_prior)
            parameters = self.get_parameter_values()
            self.build_likelihood(parameters)

    def _compute_trial_distributions(self, model_inputs):
        nu = model_inputs['nu']
        sigma_motor = model_inputs['sigma_motor']
        subject_ix = model_inputs['subject_ix']

        model = pm.Model.get_context()
        stimulus_ix = model['stimulus_ix']
        condition_ix = model['condition_ix']

        stim_grid = pt.as_tensor_variable(self.stim_grid)
        rep_grid = pt.as_tensor_variable(self.rep_grid)
        priors = pt.as_tensor_variable(self.priors)
        basis_grid = pt.as_tensor_variable(self.bernstein_basis_matrix)   # (N, n_poly)
        basis_unique = pt.as_tensor_variable(self.bernstein_basis_unique) # (K, n_poly)

        d_stim = self.d_stim
        d_rep = self.d_rep
        C = len(self.condition_names)

        if nu.ndim == 0:
            nu = nu[None]
        if sigma_motor.ndim == 0:
            sigma_motor = sigma_motor[None]

        if self.condition_specific_encoding:
            # Build per-condition encoding: mu_c(n) on stim_grid and unique_stimuli
            # Stack into (C, N) and (C, K)
            all_mu_grid = []
            all_mu_unique = []
            for c_ix in range(C):
                coeffs = self._build_coefficients(model_inputs, c_ix)
                all_mu_grid.append(pt.dot(basis_grid, coeffs))      # (N,)
                all_mu_unique.append(pt.dot(basis_unique, coeffs))   # (K,)
            mu_on_grid = pt.stack(all_mu_grid)    # (C, N)
            unique_mu = pt.stack(all_mu_unique)    # (C, K)

            # Encoding: p(r | n_i, c) = N(r; mu_c(n_i), nu^2)
            # unique_mu: (C, K), rep_grid: (R) -> (S, C, K, R)
            p_r = pt.exp(-0.5 * ((rep_grid[None, None, None, :] - unique_mu[None, :, :, None]) / nu[:, None, None, None]) ** 2) \
                  / (nu[:, None, None, None] * pt.sqrt(2 * np.pi))

            # Decoding likelihood: p(r | n, c) using condition-specific encoding
            # mu_on_grid: (C, N), rep_grid: (R) -> (S, C, N, R)
            likelihood = pt.exp(-0.5 * ((rep_grid[None, None, None, :] - mu_on_grid[None, :, :, None]) / nu[:, None, None, None]) ** 2) \
                         / (nu[:, None, None, None] * pt.sqrt(2 * np.pi))

            # Posterior: p(n | r, c) ∝ likelihood(c) * prior(c)
            posterior = likelihood * priors[None, :, :, None]  # (S, C, N, R)
            posterior = posterior / (pt.sum(posterior, axis=2, keepdims=True) * d_stim + 1e-30)

            n_hat = pt.sum(posterior * stim_grid[None, None, :, None] * d_stim, axis=2)  # (S, C, R)
            n_hat = pt.clip(n_hat, self.n_min, self.n_max)

            # Motor noise, truncated to [n_min, n_max]
            resp_grid = stim_grid
            p_resp_given_r = pt.exp(
                -0.5 * ((resp_grid[None, None, None, :] - n_hat[:, :, :, None]) / sigma_motor[:, None, None, None]) ** 2
            )
            p_resp_given_r = p_resp_given_r / (pt.sum(p_resp_given_r, axis=-1, keepdims=True) + 1e-30)

            # Marginalize over r
            p_response = pt.sum(
                p_resp_given_r[:, :, None, :, :] * p_r[:, :, :, :, None] * d_rep,
                axis=3)  # (S, C, K, N)
        else:
            # Shared encoding (original behavior)
            coefficients = self._build_coefficients(model_inputs)
            mu_on_grid = pt.dot(basis_grid, coefficients)     # (N,)
            unique_mu = pt.dot(basis_unique, coefficients)     # (K,)

            p_r = pt.exp(-0.5 * ((rep_grid[None, None, :] - unique_mu[None, :, None]) / nu[:, None, None]) ** 2) \
                  / (nu[:, None, None] * pt.sqrt(2 * np.pi))

            likelihood = pt.exp(-0.5 * ((rep_grid[None, None, :] - mu_on_grid[None, :, None]) / nu[:, None, None]) ** 2) \
                         / (nu[:, None, None] * pt.sqrt(2 * np.pi))

            posterior = likelihood[:, None, :, :] * priors[None, :, :, None]
            posterior = posterior / (pt.sum(posterior, axis=2, keepdims=True) * d_stim + 1e-30)

            n_hat = pt.sum(posterior * stim_grid[None, None, :, None] * d_stim, axis=2)
            n_hat = pt.clip(n_hat, self.n_min, self.n_max)

            resp_grid = stim_grid
            p_resp_given_r = pt.exp(
                -0.5 * ((resp_grid[None, None, None, :] - n_hat[:, :, :, None]) / sigma_motor[:, None, None, None]) ** 2
            )
            p_resp_given_r = p_resp_given_r / (pt.sum(p_resp_given_r, axis=-1, keepdims=True) + 1e-30)

            p_response = pt.sum(
                p_resp_given_r[:, :, None, :, :] * p_r[:, None, :, :, None] * d_rep,
                axis=3)

        p_response = p_response / (pt.sum(p_response, axis=-1, keepdims=True) * d_stim + 1e-30)
        return p_response[subject_ix, condition_ix, stimulus_ix, :]


class EfficientEncodingEstimationModel(LogEncodingEstimationModel):
    """Numerosity estimation with condition-specific efficient coding.

    The encoding function adapts to the condition's prior via efficient coding:
    for a uniform prior on [n_min_c, n_max_c] with log encoding, the CDF is:
        F_c(n) = (log(n) - log(n_min_c)) / (log(n_max_c) - log(n_min_c))

    This maps stimulus space uniformly onto representation space within the
    prior range, allocating more resolution to more probable stimuli.

    The sensory noise nu operates in the ENCODED representation space and is
    shared across conditions — same noise, different encoding.

    Free parameters: nu (shared), sigma_motor (shared).
    """

    def _setup_grids(self, paradigm):
        N = self.grid_resolution

        self.stim_grid = np.linspace(self.n_min, self.n_max, N)
        self.d_stim = self.stim_grid[1] - self.stim_grid[0]

        self.condition_names = sorted(self.conditions.keys())
        n_cond = len(self.condition_names)

        # Condition-specific encoding CDF: F_c(n) = (log(n) - log(n_min_c)) / (log(n_max_c) - log(n_min_c))
        self.encoding_cdfs = np.zeros((n_cond, N))
        for c_ix, cname in enumerate(self.condition_names):
            cond = self.conditions[cname]
            log_min = np.log(cond['n_min'])
            log_max = np.log(cond['n_max'])
            self.encoding_cdfs[c_ix] = np.clip(
                (np.log(self.stim_grid) - log_min) / (log_max - log_min), 0, 1)

        # Representation grid in normalized [0, 1] space with margin
        margin = 0.3
        R = N * 2
        self.rep_grid = np.linspace(-margin, 1 + margin, R)
        self.d_rep = self.rep_grid[1] - self.rep_grid[0]

        # Decoding priors
        self.priors = np.zeros((n_cond, N))
        for c_ix, cname in enumerate(self.condition_names):
            cond = self.conditions[cname]
            mask = (self.stim_grid >= cond['n_min']) & (self.stim_grid <= cond['n_max'])
            width = cond['n_max'] - cond['n_min']
            self.priors[c_ix, mask] = 1.0 / width

        self.unique_stimuli = np.sort(paradigm['n'].unique().astype(float))

        # Encoded locations for unique stimuli per condition
        self.encoded_unique = np.zeros((n_cond, len(self.unique_stimuli)))
        for c_ix, cname in enumerate(self.condition_names):
            cond = self.conditions[cname]
            log_min = np.log(cond['n_min'])
            log_max = np.log(cond['n_max'])
            self.encoded_unique[c_ix] = np.clip(
                (np.log(self.unique_stimuli) - log_min) / (log_max - log_min), 0, 1)

    def _compute_trial_distributions(self, model_inputs):
        """Condition-specific efficient encoding + shared noise + decoding."""
        nu = model_inputs['nu']
        sigma_motor = model_inputs['sigma_motor']
        subject_ix = model_inputs['subject_ix']

        model = pm.Model.get_context()
        stimulus_ix = model['stimulus_ix']
        condition_ix = model['condition_ix']

        stim_grid = pt.as_tensor_variable(self.stim_grid)
        rep_grid = pt.as_tensor_variable(self.rep_grid)
        priors = pt.as_tensor_variable(self.priors)
        enc_cdfs = pt.as_tensor_variable(self.encoding_cdfs)    # (C, N)
        enc_unique = pt.as_tensor_variable(self.encoded_unique)  # (C, K)

        d_stim = self.d_stim
        d_rep = self.d_rep

        if nu.ndim == 0:
            nu = nu[None]
        if sigma_motor.ndim == 0:
            sigma_motor = sigma_motor[None]

        # Step 1: Encoding — CONDITION-SPECIFIC
        # p(r | n_i, c) = N(r; F_c(n_i), nu^2)
        p_r = pt.exp(-0.5 * ((rep_grid[None, None, None, :] - enc_unique[None, :, :, None]) / nu[:, None, None, None]) ** 2) \
              / (nu[:, None, None, None] * pt.sqrt(2 * np.pi))  # (S, C, K, R)

        # Step 2: Bayesian decoding — condition-specific encoding + prior
        # p(r | n, c) = N(r; F_c(n), nu^2)
        likelihood = pt.exp(-0.5 * ((rep_grid[None, None, None, :] - enc_cdfs[None, :, :, None]) / nu[:, None, None, None]) ** 2) \
                     / (nu[:, None, None, None] * pt.sqrt(2 * np.pi))  # (S, C, N, R)

        posterior = likelihood * priors[None, :, :, None]
        posterior = posterior / (pt.sum(posterior, axis=2, keepdims=True) * d_stim + 1e-30)

        n_hat = pt.sum(posterior * stim_grid[None, None, :, None] * d_stim, axis=2)  # (S, C, R)
        n_hat = pt.clip(n_hat, self.n_min, self.n_max)

        # Step 3: Motor noise (SHARED), truncated to [n_min, n_max]
        resp_grid = stim_grid
        p_resp_given_r = pt.exp(
            -0.5 * ((resp_grid[None, None, None, :] - n_hat[:, :, :, None]) / sigma_motor[:, None, None, None]) ** 2
        )
        p_resp_given_r = p_resp_given_r / (pt.sum(p_resp_given_r, axis=-1, keepdims=True) + 1e-30)

        p_response = pt.sum(
            p_resp_given_r[:, :, None, :, :] * p_r[:, :, :, :, None] * d_rep,
            axis=3)  # (S, C, K, N)
        p_response = p_response / (pt.sum(p_response, axis=-1, keepdims=True) * d_stim + 1e-30)

        return p_response[subject_ix, condition_ix, stimulus_ix, :]
