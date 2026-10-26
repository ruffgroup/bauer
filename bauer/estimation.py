"""
Base class for estimation models (continuous responses).

Unlike BaseModel (binary choice with pm.Bernoulli), EstimationBaseModel
handles tasks where subjects report a continuous estimate (e.g., value bid,
numerosity estimate). The likelihood is computed from grid-based response
distributions via pm.Potential.
"""

import numpy as np
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

    def subjectwise(self, name):
        """The subject-level `(n_subjects,)` RV, NOT its trialwise expansion.

        `BaseModel.get_parameter_values` hands every free parameter back through
        `get_trialwise_variable`, i.e. already indexed out to `(n_trials,)`.
        That is right for models that compute a scalar per trial, but these
        grid models build a `(n_subjects, ..., n_response)` table and then gather
        it with `subject_ix`.  Feeding them the trialwise vector makes the
        leading axis `n_trials`, so `table[subject_ix]` reads the first
        `n_subjects` TRIALS -- all of which belong to subject 0.  Every subject
        then shares subject 0's parameters, the group SD gets exactly zero
        gradient, and the table is `n_trials / n_subjects` times larger than it
        needs to be.

        Single-subject fits were unaffected: `pt.tile` of a scalar makes every
        row identical, so the gather picked a correct row.
        """
        model = pm.Model.get_context()
        var = model[name]
        if var.ndim == 0:
            n_sub = len(model.coords['subject']) if 'subject' in model.coords else 1
            var = pt.tile(var[None], n_sub)
        return var

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
            # One-hot contraction over the GRID axis, whose length is a Python
            # int. Indexing rows with pt.arange(cdf_row.shape[0]) emits an
            # arange over a symbolic trial count, which JAX cannot JIT, so the
            # numpyro sampler dies on any model whose trial count is not folded
            # to a constant.
            grid_ix = pt.arange(n)                                # static length
            oh_lo = pt.eq(grid_ix[None, :], ix[:, None])
            oh_hi = pt.eq(grid_ix[None, :], (ix + 1)[:, None])
            val_lo = pt.sum(cdf_row * oh_lo, axis=1)
            val_hi = pt.sum(cdf_row * oh_hi, axis=1)
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

    def _trial_distribution_function(self, paradigm, parameters, strict=False):
        """Compile the per-trial response density once for this paradigm.

        Returns ``(fn, set_parameters)``: ``fn()`` evaluates the
        ``(n_trials, n_response_grid)`` density at the current parameter
        values, and ``set_parameters(parameters)`` swaps in new values of the
        same shapes without recompiling (used by :meth:`ppc`, where compiling
        a graph per posterior draw would dominate the run time).

        With ``strict=True`` it raises if rebuilding the model created any
        random variable: a quantity the trial density needs that was not among
        ``parameters`` would otherwise be drawn from its prior on every call,
        silently replacing the posterior value.
        """
        import pytensor

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

        if strict and pred_model.basic_RVs:
            raise ValueError(
                f"Rebuilding the model from {sorted(parameters)} created random "
                f"variables {[v.name for v in pred_model.basic_RVs]}; the "
                f"posterior must supply the quantities they feed (pass them in "
                f"`parameters`, or list them in `ppc_extra_parameters`).")

        fn = pytensor.function([], trial_dist)
        keys = list(parameters)

        def set_parameters(new):
            if isinstance(new, pd.DataFrame):
                new = new.to_dict(orient='list')
            if set(new) != set(keys):
                raise ValueError(f"parameters {sorted(new)} != compiled {sorted(keys)}")
            for key in keys:
                shared = pred_model[key]
                shared.set_value(np.asarray(new[key], dtype=shared.dtype))

        return fn, set_parameters

    def _sample_responses(self, trial_dist_np, n_samples=1):
        """Inverse-CDF draws from per-trial densities on the response grid.

        Returns an ``(n_trials, n_samples)`` array.  Uses the global numpy RNG.
        """
        response_grid = self._get_response_grid()
        dv = response_grid[1] - response_grid[0]

        # The lapse is mixed in here, not only in bin_probability: without it
        # simulated datasets are the no-lapse model, and any PPC built on them
        # is checking a model the likelihood never fitted -- visible wherever
        # the fitted distribution is sharp (e.g. under a hard category gate,
        # where it is a spike).
        pdf = trial_dist_np / (np.sum(trial_dist_np * dv, axis=-1, keepdims=True) + 1e-30)
        lapse_rate = getattr(self, 'lapse_rate', 0.0) or 0.0
        if lapse_rate:
            uniform = np.full_like(pdf, 1.0 / (response_grid[-1] - response_grid[0]))
            pdf = (1.0 - lapse_rate) * pdf + lapse_rate * uniform

        cdf = np.cumsum(pdf * dv, axis=-1)
        # Normalise so the last bin is exactly 1: otherwise a uniform draw
        # above a CDF that tops out at 1 - eps finds no bin, and argmax
        # silently returns bin 0 (the bottom of the response range).
        cdf = cdf / cdf[:, -1:]

        all_samples = []
        for _ in range(n_samples):
            u = np.random.uniform(size=trial_dist_np.shape[0])
            indices = np.argmax(cdf >= u[:, np.newaxis], axis=-1)
            # Add uniform jitter within the bin
            samples = response_grid[indices] + np.random.uniform(-dv / 2, dv / 2,
                                                                 size=len(indices))
            all_samples.append(samples)

        return np.column_stack(all_samples)

    def simulate(self, paradigm, parameters, n_samples=1):
        """Draw simulated responses from the predicted distribution.

        Responses are sampled from the full trial distribution mixed with the
        lapse (``lapse_rate``), i.e. the same distribution the likelihood
        scores.

        Parameters
        ----------
        paradigm : pd.DataFrame
        parameters : dict or pd.DataFrame
        n_samples : int

        Returns
        -------
        pd.DataFrame with 'simulated_response' column, stacked over samples.
        """
        fn, _ = self._trial_distribution_function(paradigm, parameters)
        all_samples = self._sample_responses(fn(), n_samples)

        if not paradigm.index.name:
            paradigm.index.name = 'trial'

        data = pd.DataFrame(all_samples, index=paradigm.index,
                            columns=pd.Index(np.arange(n_samples) + 1, name='sample'))
        data = data.stack().to_frame('simulated_response')
        data = data.join(paradigm)
        return data

    # Free parameters the posterior must supply for ``simulate`` / ``predict``.
    # Subclasses whose trial distribution also reads a posterior Deterministic
    # (e.g. a subject x condition x stimulus offset array) list its name here,
    # otherwise rebuilding the model would silently replace it.
    ppc_extra_parameters = ()

    def _ppc_parameter_names(self, parameters=None):
        if parameters is not None:
            return list(parameters)
        return list(self.free_parameters) + list(self.ppc_extra_parameters)

    def ppc(self, paradigm, idata, n_draws=200, seed=0, parameters=None,
            response_step=None, progressbar=False):
        """Posterior predictive check: simulated replicate datasets.

        For each of ``n_draws`` posterior draws (sampled jointly over
        ``(chain, draw)`` without replacement, with ``seed``), every
        participant's parameters for that draw are taken -- aligned to the
        paradigm's subjects by the posterior ``subject`` coordinate -- and ONE
        full replicate dataset is simulated with :meth:`simulate` on exactly the
        paradigm's trials.  ``simulate`` samples from the model's full response
        distribution, including the lapse mixture, so each replicate carries
        response noise, lapses and (with ``response_step``) the response
        discretisation, as the observed data do.

        Any statistic of the data (a binned mean, an s.d., a slope) must be
        computed *per draw*, exactly as on the observed data; the predictive
        band is a quantile of that statistic over draws.  Do NOT band the
        per-trial predicted means (see :meth:`posterior_predicted_moments`):
        that band ignores trial-level noise and is far too narrow.

        Parameters
        ----------
        paradigm : pd.DataFrame
            Trial-level data with the paradigm columns (the ``response``
            column, if present, is dropped from the output).  With several
            subjects it needs a ``subject`` index level or column.
        idata : arviz.InferenceData
            Posterior samples (``idata.posterior``).
        n_draws : int
            Number of replicate datasets (capped at chains x draws).
        seed : int
            Seed for picking posterior draws AND for the simulation.  The
            global numpy RNG state is restored afterwards.
        parameters : list of str, optional
            Posterior variables handed to ``simulate``.  Default: the model's
            free parameters plus ``ppc_extra_parameters``.
        response_step : float, optional
            If the responses are discretised (e.g. integer estimates, bids in
            steps of 0.5), round simulated responses to this step, then clip
            to the response grid, so replicates match the data's resolution.
            ``None`` (default) leaves them continuous.
        progressbar : bool
            Print progress every 10 draws.

        Returns
        -------
        pd.DataFrame
            Index ``('ppc_draw', *paradigm.index.names)``; columns
            ``simulated_response``, ``chain`` and ``draw`` (the posterior
            sample the replicate came from), plus the paradigm columns.

        Examples
        --------
        Group mean response per stimulus, observed vs 95% predictive band:

        >>> sim = model.ppc(data, idata, n_draws=200, seed=1)
        >>> # one group curve per replicate: subject means first, then the group
        >>> curves = (sim.groupby(['ppc_draw', 'subject', 'n'])['simulated_response']
        ...              .mean().groupby(['ppc_draw', 'n']).mean()
        ...              .unstack('n'))                     # (n_draws, n_bins)
        >>> lo, hi = curves.quantile(.025), curves.quantile(.975)
        >>> obs = data.groupby(['subject', 'n'])['response'].mean().groupby('n').mean()
        >>> coverage = ((obs >= lo) & (obs <= hi)).mean()   # should be ~.95
        """
        names = self._ppc_parameter_names(parameters)
        subjects = _paradigm_subjects(paradigm)
        draws = posterior_subject_draws(idata, subjects, names, n_draws, seed=seed)

        paradigm = paradigm.copy()
        if paradigm.index.nlevels == 1 and paradigm.index.name is None:
            paradigm.index.name = 'trial'
        index_names = list(paradigm.index.names)

        # A subclass that overrides ``simulate`` gets it called per draw.
        # Otherwise the trial density is compiled once and only the parameter
        # values are swapped per draw -- the same density and the same sampler
        # as ``simulate``, without a graph compile per draw.
        fast = type(self).simulate is EstimationBaseModel.simulate
        if fast and draws:
            fn, set_parameters = self._trial_distribution_function(
                paradigm, draws[0][2], strict=True)

        state = np.random.get_state()
        np.random.seed(seed)
        try:
            reps = []
            for i, (chain, draw, pars) in enumerate(draws):
                if fast:
                    set_parameters(pars)
                    sim = pd.DataFrame(
                        {'simulated_response': self._sample_responses(fn(), 1)[:, 0]},
                        index=paradigm.index)
                else:
                    sim = self.simulate(paradigm, pars, n_samples=1)
                    sim = sim.droplevel('sample')[['simulated_response']]
                sim['chain'] = chain
                sim['draw'] = draw
                reps.append(sim)
                if progressbar and (i + 1) % 10 == 0:
                    print(f'  ppc draw {i + 1}/{len(draws)}', flush=True)
        finally:
            np.random.set_state(state)

        out = pd.concat(reps, keys=np.arange(len(reps)), names=['ppc_draw'])
        out.index.names = ['ppc_draw'] + index_names

        if response_step is not None:
            grid = self._get_response_grid()
            out['simulated_response'] = np.clip(
                np.round(out['simulated_response'] / response_step) * response_step,
                grid[0], grid[-1])

        context = paradigm.drop(columns=['response'], errors='ignore')
        context = context.drop(columns=[c for c in out.columns if c in context.columns])
        return out.join(context)

    def posterior_predicted_moments(self, paradigm, idata, n_draws=200, seed=0,
                                    parameters=None):
        """Per-trial mean and s.d. of the model's response distribution, per draw.

        This is NOT a posterior predictive check.  ``predicted_mean`` is the
        expectation of the response distribution on each trial; averaging it
        over trials and taking quantiles over draws gives a band for the
        *expected* statistic, which reflects parameter uncertainty only.  It
        ignores response noise, lapses (the lapse mixture is not included in
        these moments) and discreteness, so it is far too narrow to judge any
        statistic of observed data (a binned mean, an s.d.) against.  Use
        :meth:`ppc` for that.

        It is useful for plotting the model's predicted curve (e.g. a posterior
        median of the expected response per stimulus) or for scoring held-out
        means.

        Returns
        -------
        pd.DataFrame
            Index ``('ppc_draw', *paradigm.index.names)``; columns
            ``predicted_mean``, ``predicted_sd``, ``chain``, ``draw`` plus the
            paradigm columns.
        """
        names = self._ppc_parameter_names(parameters)
        subjects = _paradigm_subjects(paradigm)
        paradigm = paradigm.copy()
        if paradigm.index.nlevels == 1 and paradigm.index.name is None:
            paradigm.index.name = 'trial'
        index_names = list(paradigm.index.names)

        draws = posterior_subject_draws(idata, subjects, names, n_draws, seed=seed)
        fast = type(self).predict is EstimationBaseModel.predict
        if fast and draws:
            fn, set_parameters = self._trial_distribution_function(
                paradigm, draws[0][2], strict=True)
            response_grid = self._get_response_grid()
            dv = float(response_grid[1] - response_grid[0])

        reps = []
        for chain, draw, pars in draws:
            if fast:
                # Same moments as ``predict``, from a density compiled once.
                set_parameters(pars)
                dist = fn()
                dist = dist / (np.sum(dist * dv, axis=1, keepdims=True) + 1e-30)
                mean = np.sum(dist * response_grid[None, :] * dv, axis=1)
                var = np.sum(dist * (response_grid[None, :] - mean[:, None]) ** 2 * dv, axis=1)
                pred = pd.DataFrame({'predicted_mean': mean,
                                     'predicted_sd': np.sqrt(var + 1e-30)},
                                    index=paradigm.index)
            else:
                pred = self.predict(paradigm, pars)[['predicted_mean', 'predicted_sd']]
            pred['chain'] = chain
            pred['draw'] = draw
            reps.append(pred)
        out = pd.concat(reps, keys=np.arange(len(reps)), names=['ppc_draw'])
        out.index.names = ['ppc_draw'] + index_names
        context = paradigm.drop(columns=['response'], errors='ignore')
        return out.join(context)

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


def _paradigm_subjects(paradigm):
    """Subjects in the order ``_get_paradigm`` assigns ``subject_ix`` (first
    appearance), or ``None`` for a paradigm without a subject level/column."""
    if 'subject' in paradigm.index.names:
        return list(pd.unique(paradigm.index.get_level_values('subject')))
    if 'subject' in paradigm.columns:
        return list(pd.unique(paradigm['subject']))
    return None


def posterior_subject_draws(idata, subjects, parameters, n_draws, seed=0):
    """Per-subject parameters for a random subset of posterior draws.

    Draws are picked jointly over ``(chain, draw)`` without replacement.  Each
    posterior variable's subject axis is aligned to ``subjects`` through the
    InferenceData coordinate; a subject the posterior has no parameters for
    raises, and so does a subject axis without a coordinate whose length does
    not match.  Matching is on ``str(subject)`` so ``1`` and ``'1'`` align.
    Scalar (non-hierarchical) parameters are broadcast to every subject.

    Parameters
    ----------
    idata : arviz.InferenceData or xarray.Dataset
    subjects : list or None
        Paradigm subjects, in ``subject_ix`` order.  ``None`` means a
        single-subject paradigm without a subject level.
    parameters : list of str
    n_draws : int
    seed : int

    Returns
    -------
    list of (chain, draw, pars)
        ``pars`` is a DataFrame indexed by subject when every parameter is one
        number per subject, else a dict of arrays (parameters with extra axes
        beyond the subject, e.g. subject x condition x stimulus), which
        ``simulate`` / ``predict`` accept as well.
    """
    post = getattr(idata, 'posterior', idata)
    single = subjects is None
    n_sub = 1 if single else len(subjects)

    n_chain, n_draw = post.sizes['chain'], post.sizes['draw']
    rng = np.random.default_rng(seed)
    picks = rng.choice(n_chain * n_draw, size=min(n_draws, n_chain * n_draw),
                       replace=False)

    aligned = {}
    for par in parameters:
        if par not in post:
            raise KeyError(f"{par!r} not in posterior (have: {list(post.data_vars)})")
        v = post[par]
        sub_dims = [d for d in v.dims if d not in ('chain', 'draw')]
        values = v.transpose('chain', 'draw', *sub_dims).values
        if not sub_dims:
            aligned[par] = np.repeat(values[:, :, None], n_sub, axis=2)
            continue
        dim = sub_dims[0]
        coord = list(v.coords[dim].values) if dim in v.coords else None
        if single:
            if v.sizes[dim] != 1:
                raise ValueError(
                    f"{par}: posterior has {v.sizes[dim]} entries along {dim!r} "
                    f"but the paradigm has no subject level; select one subject.")
            aligned[par] = values
        elif coord is not None:
            # Positional fallback here would silently hand subject A's
            # parameters to subject B's trials, which no plot would reveal.
            lookup = {str(c): i for i, c in enumerate(coord)}
            missing = [s for s in subjects if str(s) not in lookup]
            if missing:
                raise ValueError(
                    f"{par}: posterior {dim!r} coordinate {coord[:6]} has no "
                    f"parameters for paradigm subjects {missing[:6]}")
            aligned[par] = values[:, :, [lookup[str(s)] for s in subjects]]
        else:
            if v.sizes[dim] != n_sub:
                raise ValueError(f"{par} has {v.sizes[dim]} entries along {dim!r} "
                                 f"for {n_sub} subjects")
            aligned[par] = values

    out = []
    for flat in picks:
        c, d = divmod(int(flat), n_draw)
        per_sub = {p: aligned[p][c, d] for p in parameters}
        if all(a.ndim == 1 for a in per_sub.values()):
            pars = pd.DataFrame(per_sub, index=pd.Index(
                [0] if single else subjects, name='subject'))
        else:
            pars = per_sub
        out.append((int(post['chain'].values[c]), int(post['draw'].values[d]), pars))
    return out
