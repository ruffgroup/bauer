"""
Base class for estimation models (continuous responses).

Unlike BaseModel (binary choice with pm.Bernoulli), EstimationBaseModel
handles tasks where subjects report a continuous estimate (e.g., value bid,
numerosity estimate). The likelihood is computed from grid-based response
distributions via pm.Potential.
"""

import numpy as np
from numpy import trapezoid as np_trapz
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytensor.tensor.math as ptm
from .core import BaseModel


class EstimationBaseModel(BaseModel):
    """Base class for continuous-response estimation models.

    Subclasses must implement:
        - get_free_parameters()
        - _get_response_log_likelihood(model_inputs) -> (n_trials,) log-prob
        - get_model_inputs(parameters)
    """

    def __init__(self, paradigm=None, grid_resolution=101, response_bin_width=0.5,
                 lapse_rate=0.01, **kwargs):
        """
        Parameters
        ----------
        paradigm : pd.DataFrame
            Trial-level data with 'response' column and stimulus columns.
        grid_resolution : int
            Number of grid points for stimulus and response grids.
            Coarser grids are faster (useful for MCMC); finer grids are
            more accurate (useful for MLE). Default 101.
        response_bin_width : float
            Width of the response bin for likelihood computation.
            Responses are snapped to grid, and probability is integrated
            over [response_center - bin_width/2, response_center + bin_width/2].
        lapse_rate : float
            Fixed probability of random response (uniform over response range).
        """
        self.grid_resolution = grid_resolution
        self.response_bin_width = response_bin_width
        self.lapse_rate = lapse_rate
        super().__init__(paradigm, **kwargs)

    def _get_paradigm(self, paradigm=None, subject_mapping=None):
        if paradigm is None:
            paradigm = self.data

        paradigm_ = {}
        paradigm_['_data_n'] = len(paradigm)

        for key in self.paradigm_keys:
            paradigm_[key] = paradigm[key].values

        if subject_mapping is None:
            if 'subject' in paradigm.index.names:
                paradigm_['subject_ix'], _ = pd.factorize(paradigm.index.get_level_values('subject'))
            elif 'subject' in paradigm.columns:
                paradigm_['subject_ix'], _ = pd.factorize(paradigm['subject'])
            else:
                # Single subject, no subject index — all trials belong to subject 0
                paradigm_['subject_ix'] = np.zeros(len(paradigm), dtype=int)
        else:
            if 'subject' in paradigm.index.names:
                paradigm_['subject_ix'] = [subject_mapping[subject] for subject in paradigm.index.get_level_values('subject')]
            else:
                paradigm_['subject_ix'] = [subject_mapping[subject] for subject in paradigm['subject']]

        paradigm_['response'] = paradigm['response'].values.astype(float)

        return paradigm_

    def build_likelihood(self, parameters, save_p_choice=False):
        model_inputs = self.get_model_inputs(parameters)
        trial_dist = self._compute_trial_distributions(model_inputs)
        response_grid = self._get_response_grid()
        log_prob = self.bin_probability(trial_dist, response_grid, model_inputs['response'])
        pm.Potential('ll', log_prob.sum())

        if save_p_choice:
            grid = pt.as_tensor_variable(response_grid)
            dv = float(response_grid[1] - response_grid[0])
            norm = pt.sum(trial_dist * dv, axis=1, keepdims=True) + 1e-30
            dist_norm = trial_dist / norm
            predicted_mean = pt.sum(dist_norm * grid[None, :] * dv, axis=1)
            predicted_var = pt.sum(
                dist_norm * (grid[None, :] - predicted_mean[:, None]) ** 2 * dv, axis=1)
            pm.Deterministic('predicted_mean', predicted_mean)
            pm.Deterministic('predicted_sd', pt.sqrt(predicted_var + 1e-30))

    def _compute_trial_distributions(self, model_inputs):
        """Compute predicted response PDF for each trial on the response grid.

        Must return a pytensor tensor of shape (n_trials, n_response_grid).
        Subclasses implement this.
        """
        raise NotImplementedError

    def _get_response_log_likelihood(self, model_inputs):
        """Compute per-trial log-likelihood. Default: uses _compute_trial_distributions."""
        trial_dist = self._compute_trial_distributions(model_inputs)
        return self.bin_probability(trial_dist, self._get_response_grid(), model_inputs['response'])

    # ---- Grid utilities (pytensor) ----

    @staticmethod
    def truncated_normal_pdf(x, mu, sigma, lower, upper):
        """Truncated Gaussian PDF on a grid, broadcast-safe.

        Computes N(x; mu, sigma^2) truncated to [lower, upper].
        Values outside the range are zeroed, then renormalized.
        Works on pytensor tensors.
        """
        raw = pt.exp(-0.5 * ((x - mu) / sigma) ** 2)
        mask = (x >= lower) & (x <= upper)
        raw = raw * mask
        # Normalize: sum over the last axis (grid axis)
        total = pt.sum(raw, axis=-1, keepdims=True) + 1e-30
        return raw / total

    @staticmethod
    def trapz(y, dx, axis=-1):
        """Trapezoidal integration along axis."""
        return pt.sum((y[..., :-1] + y[..., 1:]) / 2 * dx, axis=axis)

    @staticmethod
    def cumtrapz(y, dx, axis=-1):
        """Cumulative trapezoidal integration along axis (no prepended zero)."""
        return pt.cumsum((y[..., :-1] + y[..., 1:]) / 2 * dx, axis=axis)

    @staticmethod
    def normal_pdf(x, mu, sigma):
        """Gaussian PDF, broadcast-safe."""
        return pt.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * pt.sqrt(2 * np.pi))

    @staticmethod
    def normal_cdf(x, mu, sigma):
        """Gaussian CDF via erfc, broadcast-safe."""
        return 0.5 * ptm.erfc(-(x - mu) / (sigma * pt.sqrt(2.0)))

    @staticmethod
    def von_mises_pdf(x, mu, kappa):
        """Von Mises PDF, broadcast-safe."""
        return pt.exp(kappa * ptm.cos(x - mu)) / (2 * np.pi * ptm.i0(kappa))

    @staticmethod
    def von_mises_logpdf(x, mu, kappa):
        """Von Mises log-PDF, broadcast-safe."""
        return kappa * ptm.cos(x - mu) - pt.log(2 * np.pi * ptm.i0(kappa))

    def bin_probability(self, pdf_on_grid, response_grid, observed_responses):
        """Compute per-trial log P(response in bin) from a per-trial PDF on a grid.

        Parameters
        ----------
        pdf_on_grid : pytensor tensor, shape (n_trials, n_grid)
            Predicted response density per trial, evaluated on the grid.
        response_grid : numpy array, shape (n_grid,)
            The response grid values (equally spaced).
        observed_responses : pytensor tensor, shape (n_trials,)
            Observed response values.

        Returns
        -------
        log_prob : pytensor tensor, shape (n_trials,)
        """
        dv = float(response_grid[1] - response_grid[0])
        half_bin = self.response_bin_width / 2.0
        grid_min = float(response_grid[0])
        grid_max = float(response_grid[-1])
        n_grid = len(response_grid)

        # Build per-trial CDF via cumulative trapz, prepend 0
        # pdf_on_grid: (n_trials, n_grid)
        midpoints = (pdf_on_grid[:, :-1] + pdf_on_grid[:, 1:]) / 2 * dv  # (n_trials, n_grid-1)
        cum_mass = pt.cumsum(midpoints, axis=1)  # (n_trials, n_grid-1)
        cdf = pt.concatenate([pt.zeros((pdf_on_grid.shape[0], 1)), cum_mass], axis=1)  # (n_trials, n_grid)

        # Fractional grid indices for response ± half_bin
        response_lo = pt.clip(observed_responses - half_bin, grid_min, grid_max)
        response_hi = pt.clip(observed_responses + half_bin, grid_min, grid_max)

        frac_lo = (response_lo - grid_min) / dv  # (n_trials,)
        frac_hi = (response_hi - grid_min) / dv

        # Per-row linear interpolation of CDF
        def interp_per_row(cdf_row, frac_ix):
            """Interpolate CDF for each trial (row) at its fractional index."""
            n = n_grid
            ix = pt.cast(pt.floor(frac_ix), 'int64')
            ix = pt.clip(ix, 0, n - 2)
            w = frac_ix - pt.cast(ix, 'float64')
            row_indices = pt.arange(cdf_row.shape[0])
            val_lo = cdf_row[row_indices, ix]
            val_hi = cdf_row[row_indices, ix + 1]
            return val_lo + w * (val_hi - val_lo)

        cdf_lo = interp_per_row(cdf, frac_lo)
        cdf_hi = interp_per_row(cdf, frac_hi)

        p_bin = cdf_hi - cdf_lo

        # Mix with lapse
        lapse_p = self.response_bin_width / (grid_max - grid_min)
        p_bin = (1 - self.lapse_rate) * p_bin + self.lapse_rate * lapse_p

        return pt.log(pt.clip(p_bin, 1e-12, 1.0))

    # ---- Prediction / simulation / PPC ----

    def _get_response_grid(self):
        """Return the numpy response grid. Subclasses must implement."""
        raise NotImplementedError

    def predict(self, paradigm, parameters):
        """Compute predicted mean response for each trial.

        Parameters
        ----------
        paradigm : pd.DataFrame
            Trial-level data (same format as fitting data).
        parameters : dict or pd.DataFrame
            Fitted parameters. If DataFrame, must have 'subject' index matching paradigm.

        Returns
        -------
        pd.DataFrame with 'predicted_mean' and 'predicted_sd' columns.
        """
        if isinstance(parameters, pd.DataFrame):
            parameters = parameters.to_dict(orient='list')

        # Build a temporary model with parameters as Data
        self._setup_grids(paradigm)
        with pm.Model() as pred_model:
            paradigm_ = self._get_paradigm(paradigm=paradigm)
            self.set_paradigm(paradigm_)
            for key, value in parameters.items():
                pm.Data(key, value)
            params = self.get_parameter_values()
            model_inputs = self.get_model_inputs(params)
            trial_dist = self._compute_trial_distributions(model_inputs)

            response_grid = self._get_response_grid()
            grid = pt.as_tensor_variable(response_grid)
            dv = float(response_grid[1] - response_grid[0])
            norm = pt.sum(trial_dist * dv, axis=1, keepdims=True) + 1e-30
            dist_norm = trial_dist / norm
            mean = pt.sum(dist_norm * grid[None, :] * dv, axis=1)
            var = pt.sum(dist_norm * (grid[None, :] - mean[:, None]) ** 2 * dv, axis=1)

            pm.Deterministic('predicted_mean', mean)
            pm.Deterministic('predicted_sd', pt.sqrt(var + 1e-30))

        result = pd.DataFrame({
            'predicted_mean': pred_model['predicted_mean'].eval(),
            'predicted_sd': pred_model['predicted_sd'].eval(),
        }, index=paradigm.index)
        return result.join(paradigm)

    def simulate(self, paradigm, parameters, n_samples=1):
        """Draw simulated responses from the predicted distribution.

        Parameters
        ----------
        paradigm : pd.DataFrame
        parameters : dict or pd.DataFrame
        n_samples : int

        Returns
        -------
        pd.DataFrame with 'simulated_response' column, stacked over samples.
        """
        if isinstance(parameters, pd.DataFrame):
            parameters = parameters.to_dict(orient='list')

        self._setup_grids(paradigm)
        with pm.Model() as pred_model:
            paradigm_ = self._get_paradigm(paradigm=paradigm)
            self.set_paradigm(paradigm_)
            for key, value in parameters.items():
                pm.Data(key, value)
            params = self.get_parameter_values()
            model_inputs = self.get_model_inputs(params)
            trial_dist = self._compute_trial_distributions(model_inputs)
            pm.Deterministic('trial_dist', trial_dist)

        trial_dist_np = pred_model['trial_dist'].eval()
        response_grid = self._get_response_grid()
        dv = response_grid[1] - response_grid[0]

        # Inverse CDF sampling
        all_samples = []
        for _ in range(n_samples):
            u = np.random.uniform(size=trial_dist_np.shape[0])
            pdf = trial_dist_np / (np.sum(trial_dist_np * dv, axis=-1, keepdims=True) + 1e-30)
            cdf = np.cumsum(pdf * dv, axis=-1)
            indices = np.argmax(cdf >= u[:, np.newaxis], axis=-1)
            # Add uniform jitter within the bin
            samples = response_grid[indices] + np.random.uniform(-dv / 2, dv / 2,
                                                                   size=len(indices))
            all_samples.append(samples)

        all_samples = np.column_stack(all_samples)

        if not paradigm.index.name:
            paradigm.index.name = 'trial'

        data = pd.DataFrame(all_samples, index=paradigm.index,
                            columns=pd.Index(np.arange(n_samples) + 1, name='sample'))
        data = data.stack().to_frame('simulated_response')
        data = data.join(paradigm)
        return data

    def ppc(self, paradigm, idata, n_ppc_samples=1, var_names=None, progressbar=True):
        """Posterior predictive check.

        For each posterior draw, computes the predicted mean response per trial.
        Optionally simulates responses.

        Parameters
        ----------
        paradigm : pd.DataFrame
            Same trial-level data used for fitting.
        idata : arviz.InferenceData
            Posterior samples from MCMC.
        n_ppc_samples : int
            Number of simulated responses per posterior draw per trial.
        progressbar : bool

        Returns
        -------
        pd.DataFrame with predicted_mean (and optionally simulated responses)
            per trial, per (chain, draw).
        """
        # Rebuild the model with save_predictions=True
        self._setup_grids(paradigm)
        with pm.Model(coords={'subject': idata.posterior.coords.get('subject',
                              paradigm.index.unique(level='subject') if 'subject' in paradigm.index.names else [0])}) as ppc_model:
            paradigm_ = self._get_paradigm(paradigm=paradigm)
            self.set_paradigm(paradigm_)
            self.build_priors(hierarchical='subject' in paradigm.index.names)
            parameters = self.get_parameter_values()
            self.build_likelihood(parameters, save_p_choice=True)

        with ppc_model:
            ppc_data = pm.sample_posterior_predictive(
                idata, var_names=['predicted_mean', 'predicted_sd'],
                progressbar=progressbar)

        # Extract into a DataFrame
        pred_mean = ppc_data.posterior_predictive['predicted_mean']  # (chain, draw, trial)
        pred_sd = ppc_data.posterior_predictive['predicted_sd']

        return ppc_data

    # ---- Grid-search MLE ----

    def fit_mle_grid_search(self, data=None, param_grids=None, n_jobs=-1):
        """Fit per-subject MLE via exhaustive grid search.

        Parameters
        ----------
        data : pd.DataFrame
            Trial-level data with 'subject' index level.
        param_grids : dict
            {parameter_name: np.array of values to search over}
        n_jobs : int
            Number of parallel jobs for joblib (-1 = all cores).

        Returns
        -------
        pd.DataFrame
            Best-fit parameters per subject.
        """
        from itertools import product as iterproduct
        from joblib import Parallel, delayed

        if data is None:
            data = self.paradigm

        subjects = data.index.get_level_values('subject').unique()

        # Create all parameter combinations
        param_names = list(param_grids.keys())
        param_values = [param_grids[name] for name in param_names]
        grid_points = list(iterproduct(*param_values))

        def fit_single_subject(subj):
            subj_data = data.xs(subj, level='subject')
            best_nll = np.inf
            best_params = None

            for point in grid_points:
                params = dict(zip(param_names, point))

                # Build model and evaluate log-likelihood
                subj_model = self.__class__.__new__(self.__class__)
                subj_model.__dict__.update(self.__dict__)
                subj_model.paradigm = subj_data
                subj_model.build_estimation_model(data=subj_data,
                                                  hierarchical=False,
                                                  flat_prior=True)

                try:
                    with subj_model.estimation_model:
                        # Set parameter values
                        point_dict = {}
                        for name, val in params.items():
                            transform = self.free_parameters[name].get('transform', 'identity')
                            if transform == 'softplus':
                                # Inverse softplus to get untransformed value
                                point_dict[name + '_untransformed'] = np.log(np.exp(val) - 1)
                            elif transform == 'logistic':
                                point_dict[name + '_untransformed'] = np.log(val / (1 - val))
                            else:
                                point_dict[name] = val

                        nll = -subj_model.estimation_model.point_logps(point_dict)['ll']

                except Exception:
                    nll = np.inf

                if nll < best_nll:
                    best_nll = nll
                    best_params = params

            return {'subject': subj, **best_params, 'nll': best_nll}

        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_single_subject)(subj) for subj in subjects
        )

        return pd.DataFrame(results).set_index('subject')
