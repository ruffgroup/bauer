import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from arviz import hdi
from patsy import dmatrix
from warnings import warn
from ..core import BaseModel, LapseModel, RegressionModel
from ..utils.bayes import cumulative_normal, get_posterior, get_diff_dist
from ..utils.math import inverse_softplus_np, softplus_np, inverse_softplus
from ..utils.plotting import plot_prediction

class MagnitudeComparisonModel(BaseModel):
    """Bayesian observer model for two-alternative magnitude comparison (e.g. numerosity).

    Choices between quantities n1 and n2 are modelled as Bayesian inference over log-scale
    representations corrupted by Gaussian noise.  The prior is either estimated from the
    stimulus distribution (``fit_prior=False``) or treated as free parameters.

    Parameters
    ----------
    paradigm : pd.DataFrame, optional
        Must contain columns ``n1``, ``n2``, and ``choice``.
    fit_prior : bool
        If True, fit ``prior_mu`` and ``prior_sd`` as free parameters.
    fit_seperate_evidence_sd : bool
        If True, fit separate noise parameters for n1 and n2 (or perceptual/memory noise
        when ``memory_model='shared_perceptual_noise'``).
    memory_model : {'independent', 'shared_perceptual_noise'}
        Noise structure. ``'independent'`` fits n1_evidence_sd and n2_evidence_sd separately.
        ``'shared_perceptual_noise'`` decomposes into perceptual and memory noise.
    """

    def __init__(self, paradigm=None, fit_prior=False, fit_seperate_evidence_sd=True, memory_model = 'independent',save_trialwise_n_estimates=False):

        self.fit_prior = fit_prior
        self.fit_seperate_evidence_sd = fit_seperate_evidence_sd
        self.memory_model = memory_model

        super().__init__(paradigm, save_trialwise_n_estimates=save_trialwise_n_estimates)

    def get_model_inputs(self, parameters):

        model = pm.Model.get_context()

        model_inputs = {}

        if self.fit_prior:
            model_inputs['n1_prior_mu'] = parameters['prior_mu']
            model_inputs['n2_prior_mu'] = parameters['prior_mu']
            model_inputs['n1_prior_sd'] = parameters['prior_sd']
            model_inputs['n2_prior_sd'] = parameters['prior_sd']

        else:
            mean_prior = (pt.mean(pt.log(model['n1'])) + pt.mean(pt.log(model['n2']))) / 2.
            mean_std = (pt.std(pt.log(model['n1'])) + pt.std(pt.log(model['n2']))) / 2.

            model_inputs['n1_prior_mu'] = mean_prior
            model_inputs['n2_prior_mu'] = mean_prior

            model_inputs['n1_prior_sd'] = mean_std
            model_inputs['n2_prior_sd'] = mean_std

        model_inputs['n1_evidence_mu'] = model['log(n1)']
        model_inputs['n2_evidence_mu'] = model['log(n2)']

        model_inputs['threshold'] =  0.0

        if self.fit_seperate_evidence_sd:
            if self.memory_model == 'independent':
                model_inputs['n1_evidence_sd'] = parameters['n1_evidence_sd']
                model_inputs['n2_evidence_sd']= parameters['n2_evidence_sd']
            elif self.memory_model == 'shared_perceptual_noise':
                perceptual_sd = parameters['perceptual_noise_sd']
                memory_sd = parameters['memory_noise_sd']

                model_inputs['n1_evidence_sd'] = perceptual_sd + memory_sd
                model_inputs['n2_evidence_sd'] = perceptual_sd
            else:
                raise ValueError('Unknown memory model')
        else:
            model_inputs['n1_evidence_sd'] = parameters['evidence_sd']
            model_inputs['n2_evidence_sd'] = parameters['evidence_sd']

        return model_inputs

    def get_free_parameters(self):

        free_parameters = {}

        if self.fit_seperate_evidence_sd:
            if self.memory_model == 'independent':
                free_parameters['n1_evidence_sd'] = {'mu_intercept': -1., 'transform': 'softplus'}
                free_parameters['n2_evidence_sd'] = {'mu_intercept': -1., 'transform': 'softplus'}
            elif self.memory_model == 'shared_perceptual_noise':
                free_parameters['perceptual_noise_sd'] = {'mu_intercept': -1., 'transform': 'softplus'}
                free_parameters['memory_noise_sd'] = {'mu_intercept': -1., 'transform': 'softplus'}

        else:
            free_parameters['evidence_sd'] = {'mu_intercept': -1., 'transform': 'softplus'}

        if self.fit_prior:
            log_ns = np.log(np.concatenate((self.paradigm['n1'], self.paradigm['n2'])))
            objective_mu = np.mean(log_ns)
            objective_sd = np.std(log_ns)

            free_parameters['prior_mu'] = {'mu_intercept': objective_mu, 'transform': 'identity'}
            free_parameters['prior_sd'] = {'mu_intercept': objective_sd, 'transform': 'softplus'}

        return free_parameters

    def _get_paradigm(self, paradigm=None):

        paradigm_ = super()._get_paradigm(paradigm)

        paradigm_['n1'] = paradigm['n1'].values
        paradigm_['n2'] = paradigm['n2'].values
        paradigm_['log(n1)'] = np.log(paradigm['n1'].values)
        paradigm_['log(n2)'] = np.log(paradigm['n2'].values)

        return paradigm_

    def _get_example_paradigm(self, n_fractions=5):
        base_ns = np.array([5, 7, 10, 14, 20, 28])
        fractions = np.exp(np.linspace(np.log(.5), np.log(2.), n_fractions))

        n1 = np.repeat(base_ns, len(fractions))
        n2 = (base_ns[:, None] * fractions[None, :]).ravel()

        paradigm = pd.DataFrame({'n1':n1, 'n2':n2})

        return paradigm

class MagnitudeComparisonRegressionModel(RegressionModel, MagnitudeComparisonModel):
    """MagnitudeComparisonModel with patsy formula regression on noise/prior parameters."""

    def __init__(self, paradigm, regressors, fit_prior=False, fit_seperate_evidence_sd=True, memory_model = 'independent',save_trialwise_estimates=False):
        RegressionModel.__init__(self, regressors)
        MagnitudeComparisonModel.__init__(self, paradigm, fit_prior, fit_seperate_evidence_sd, memory_model, save_trialwise_estimates)

class MagnitudeComparisonLapseModel(LapseModel, MagnitudeComparisonModel):
    """MagnitudeComparisonModel extended with a lapse rate parameter."""
    ...

class MagnitudeComparisonLapseRegressionModel(LapseModel, MagnitudeComparisonRegressionModel):
    """MagnitudeComparisonModel with both a lapse rate and patsy formula regression."""
    ...

class FlexibleNoiseComparisonModel(BaseModel):
    """Magnitude comparison model with stimulus-dependent noise parameterised by a polynomial spline.

    Unlike :class:`MagnitudeComparisonModel`, evidence noise is modelled as a polynomial
    function of log-magnitude, allowing the noise level to vary smoothly with stimulus size.

    Parameters
    ----------
    paradigm : pd.DataFrame
        Must contain columns ``n1``, ``n2``, and ``choice``.
    polynomial_order : int or tuple of int
        Order(s) of the polynomial for the noise curve (one per prospect when
        ``fit_seperate_evidence_sd=True``).
    memory_model : {'independent', 'shared_perceptual_noise'}
        Noise decomposition; see :class:`MagnitudeComparisonModel`.
    """

    def __init__(self, paradigm, fit_seperate_evidence_sd=True,
                 fit_prior=False,
                 polynomial_order=5,
                 memory_model='independent'):

        self.fit_prior = fit_prior
        self.fit_seperate_evidence_sd = fit_seperate_evidence_sd

        if ~fit_seperate_evidence_sd and (memory_model != 'independent'):
            raise ValueError('Single evidence_sd can only be used with memory_model=independent')

        if (type(polynomial_order) is int) and fit_seperate_evidence_sd:
            polynomial_order = polynomial_order, polynomial_order

        self.polynomial_order = polynomial_order
        self.max_polynomial_order = np.max(self.polynomial_order)
        self.memory_model = memory_model

        super().__init__(paradigm)

    def build_estimation_model(self, paradigm=None, coords=None, hierarchical=True, save_p_choice=False):

        coords = {}

        if paradigm is None:
            paradigm = self.paradigm

        if hierarchical and ('subject' not in coords.keys()):
            assert('subject' in paradigm.index.names), "Hierarchical estimation requires a multi-index with a 'subject' level."
            coords['subject'] = paradigm.index.unique(level='subject')

        coords['poly_order'] = np.arange(self.max_polynomial_order)

        return BaseModel.build_estimation_model(self, data=paradigm, coords=coords, hierarchical=hierarchical, save_p_choice=save_p_choice)

    def get_model_inputs(self, parameters):

        model = pm.Model.get_context()

        model_inputs = {}

        if self.fit_prior:
            prior_mu = model['prior_mu']
            prior_sd = model['prior_sd']
        else:
            prior_mu = pt.mean(pt.stack([model['n1'], model['n2']], axis=1), 1)
            prior_sd = pt.std(pt.stack([model['n1'], model['n2']], axis=1), 1)

        model_inputs['n1_prior_mu'] = prior_mu
        model_inputs['n2_prior_mu'] = prior_mu
        model_inputs['n1_prior_sd'] = prior_sd
        model_inputs['n2_prior_sd'] = prior_sd
        model_inputs['threshold'] =  0.0

        model_inputs['n1_evidence_mu'] = model['n1']
        model_inputs['n2_evidence_mu'] = model['n2']

        model_inputs['n1_evidence_sd'] = self._get_trialwise_evidence_sd('n1_evidence_sd', parameters)
        model_inputs['n2_evidence_sd'] = self._get_trialwise_evidence_sd('n2_evidence_sd', parameters)

        return model_inputs

    def get_free_parameters(self):

        free_parameters = {}

        if self.fit_seperate_evidence_sd:
            key1, key2 = self._get_evidence_sd_labels()

            for n in range(1, self.polynomial_order[0]+1):
                free_parameters[f'{key1}_spline{n}'] = {'mu_intercept': 5., 'sigma_intercept': 5., 'transform': 'identity'}

            for n in range(1, self.polynomial_order[1]+1):
                free_parameters[f'{key2}_spline{n}'] = {'mu_intercept': 5., 'sigma_intercept': 5., 'transform': 'identity'}

        else:
            for n in range(1, self.polynomial_order+1):
                free_parameters[f'evidence_sd_spline{n}'] = {'mu_intercept': 5., 'sigma_intercept': 5., 'transform': 'identity'}

        if self.fit_prior:
            if self.paradigm is not None:
                ns = np.concatenate((self.paradigm['n1'].values, self.paradigm['n2'].values))
                objective_mu = np.mean(ns)
                objective_sd = np.std(ns)

            else:
                objective_mu = np.log(25)
                objective_sd = 2

            free_parameters['prior_mu'] = {'mu_intercept': objective_mu, 'transform': 'identity'}
            free_parameters['prior_sd'] = {'mu_intercept': objective_sd, 'transform': 'softplus'}

        return free_parameters

    def _get_evidence_sd_spline_par_labels(self):
        if self.fit_seperate_evidence_sd:
            key1, key2 = self._get_evidence_sd_labels()
            label1 = [f'{key1}_spline{n}' for n in range(1, self.polynomial_order[0]+1)]
            label2 = [f'{key2}_spline{n}' for n in range(1, self.polynomial_order[1]+1)]
            return label1, label2
        else:
            labels = [f'evidence_sd_spline{n}' for n in range(1, self.polynomial_order+1)]
            return labels, labels

    def _get_evidence_sd_labels(self):
        if self.memory_model == 'independent':
            key1 = 'n1_evidence_sd'
            key2 = 'n2_evidence_sd'
        elif self.memory_model == 'shared_perceptual_noise':
            key1 = 'memory_noise_sd'
            key2 = 'perceptual_noise_sd'

        return key1, key2

    def _get_trialwise_evidence_sd(self, key, parameters):

        model = pm.Model.get_context()

        key1, key2 = self._get_evidence_sd_labels()
        labels1, labels2 = self._get_evidence_sd_spline_par_labels()

        if key == 'n1_evidence_sd':
            if self.memory_model == 'independent':
                dm = self.make_dm(x=self.paradigm['n1'], variable=key1)
                spline_pars = pt.stack([parameters[l1] for l1 in labels1], axis=1)

            elif self.memory_model == 'shared_perceptual_noise':
                dm1 = self.make_dm(x=self.paradigm['n1'], variable=key1)
                spline_pars1 = pt.stack([parameters[l1] for l1 in labels1], axis=1)
                dm2 = self.make_dm(x=self.paradigm['n1'], variable=key2)
                spline_pars2 = pt.stack([parameters[l2] for l2 in labels2], axis=1)

                return pt.softplus(pt.sum(spline_pars1 * dm1, 1) + pt.sum(spline_pars2 * dm2, 1))

        elif key == 'n2_evidence_sd':
            dm = self.make_dm(x=self.paradigm['n2'], variable=key2)
            spline_pars = pt.stack([parameters[l2] for l2 in labels2], axis=1)

        return pt.softplus(pt.sum(spline_pars * dm, 1))

    def make_dm(self, x, variable='n1_evidence_sd'):

        min_n, max_n = self.paradigm[['n1', 'n2']].min().min(), self.paradigm[['n1', 'n2']].max().max()

        if self.fit_seperate_evidence_sd:
            if variable in ['n1_evidence_sd', 'perceptual_noise_sd']:
                polynomial_order = self.polynomial_order[0]
            elif variable in ['n2_evidence_sd', 'memory_noise_sd']:
                polynomial_order = self.polynomial_order[1]
        else:
            polynomial_order = self.polynomial_order

        if polynomial_order > 1:
            dm = np.asarray(dmatrix(f"bs(x, degree=3, df={polynomial_order}, include_intercept=True, lower_bound={min_n}, upper_bound={max_n}) - 1",
                            {"x": x}))
        else:
            dm = np.asarray(dmatrix(f"bs(x, degree=0, df=0, include_intercept=False, lower_bound={min_n}, upper_bound={max_n})",
                            {"x": x}))

        return dm

    def get_sd_curve(self, idata=None, pars=None, x=None, variable='n1_evidence_sd', group=True, hierarchical=True, data=None):

        if (idata is None) and (pars is None):
            raise ValueError('Either idata or pars must be provided.')

        if (idata is not None) and (pars is not None):
            raise ValueError('Only one of idata or pars must be provided.')

        assert(variable in ['n1_evidence_sd', 'n2_evidence_sd', 'evidence_sd', 'both', 'perceptual_noise_sd', 'memory_noise_sd']), "Variable must be 'n1_evidence_sd', 'n2_evidence_sd', 'both', 'perceptual_noise_sd', or 'memory_noise_sd'."

        if variable == 'both':
            n1_evidence_sd = self.get_sd_curve(idata, pars, x=x, variable='n1_evidence_sd', group=group, hierarchical=hierarchical, data=data)
            n2_evidence_sd = self.get_sd_curve(idata, pars, x=x, variable='n2_evidence_sd', group=group, hierarchical=hierarchical, data=data)

            return n1_evidence_sd.join(n2_evidence_sd)

        if variable == 'evidence_sd':
            assert(not self.fit_seperate_evidence_sd), "Single evidence_sd only when not fit_seperate_evidence_sd."

            evidence_sd = self.get_sd_curve(idata, x=x, variable='n1_evidence_sd', group=group, hierarchical=hierarchical, data=data)

            evidence_sd.columns = ['evidence_sd']

            return evidence_sd


        if x is None:
            assert((data is not None) or (hasattr(self, 'data'))), "If x is not provided, data must be provided."

            if data is None:
                data = self.data

            x_min, x_max = data[['n1', 'n2']].min().min(), data[['n1', 'n2']].max().max()
            x = np.linspace(x_min, x_max, 100)

        if group and not hierarchical:
            raise ValueError('Groupwise estimates only for hierarchical models.')

        labels1, labels2 = self._get_evidence_sd_spline_par_labels()

        if group:
            if pars is not None:
                raise ValueError('MAP pars are not good for groupwise estimates')

            labels1 = [f'{l}_mu' for l in labels1]
            labels2 = [f'{l}_mu' for l in labels2]

        if (variable == 'n1_evidence_sd') & (self.memory_model == 'shared_perceptual_noise'):

            if pars is None:
                pars1 = idata.posterior[labels1].to_dataframe()
                pars2 = idata.posterior[labels2].to_dataframe()
            else:
                pars1 = pars[labels1]
                pars2 = pars[labels2]

            perceptual_noise = pars1.to_dataframe()
            memory_noise = pars2.to_dataframe()

            dm1 = self.make_dm(x=x, variable='perceptual_noise_sd')
            dm2 = self.make_dm(x=x, variable='memory_noise_sd')

            perceptual_noise = softplus_np(perceptual_noise.dot(dm1.T))
            memory_noise = softplus_np(memory_noise.dot(dm2.T))

            return perceptual_noise + memory_noise

        else:
            if variable in ['n1_evidence_sd', 'memory_noise']:
                labels = labels1
            else:
                labels = labels2

            if pars is None:
                pars = idata.posterior[labels].to_dataframe()
            else:
                pars = pars[labels]

            dm = self.make_dm(x=x, variable=variable)
            output = softplus_np(pars.dot(dm.T))

        output.columns = x
        output.columns.name = 'x'

        output = output.stack().to_frame(variable)
        output.columns.name = 'variable'

        return output

    @classmethod
    def get_sd_curve_stats(n_sd, groupby=[]):
        keys = ['x']

        if 'subject' in n_sd.index.names:
            keys.append('subject')

        if 'variable' in n_sd.index.names:
            keys.append('variable')

        keys += groupby

        sd_ci = n_sd.groupby(keys).apply(lambda d: pd.Series(hdi(d.values.ravel())))#, index=pd.Index(['hdi025', 'hdi975']))))
        sd_ci.columns = ['hdi025', 'hdi975']
        sd_mean = n_sd.groupby(keys).mean()

        return sd_mean.join(sd_ci)

    @classmethod
    def plot_sd_curve_stats(n_sd_stats, ylim=(0, 20), y=None, **kwargs):

        if y == None:
            y = n_sd_stats.columns[0]

        hue = 'variable' if 'variable' in n_sd_stats.index.names else None
        col = 'subject' if 'subject' in n_sd_stats.index.names else None

        g = sns.FacetGrid(n_sd_stats.reset_index(), hue=hue, col=col, col_wrap=3 if col is not None else None, sharex=False, sharey=False,
                        **kwargs)

        g.map_dataframe(plot_prediction, x='x', y=y)
        g.map_dataframe(plt.plot, 'x', y)

        g.set(ylim=ylim)
        g.fig.set_size_inches(6, 6)

        return g

    @staticmethod
    def get_sd_curve_stats(n_sd, groupby=[]):
        keys = ['x']

        if 'subject' in n_sd.index.names:
            keys.append('subject')

        if 'variable' in n_sd.index.names:
            keys.append('variable')

        keys += groupby

        sd_ci = n_sd.groupby(keys).apply(lambda d: pd.Series(hdi(d.values.ravel())))#, index=pd.Index(['hdi025', 'hdi975']))))
        sd_ci.columns = ['hdi025', 'hdi975']
        sd_mean = n_sd.groupby(keys).mean()

        return sd_mean.join(sd_ci)

    @staticmethod
    def plot_sd_curve_stats(n_sd_stats, ylim=(0, 20)):

        hue = 'variable' if 'variable' in n_sd_stats.index.names else None
        col = 'subject' if 'subject' in n_sd_stats.index.names else None

        g = sns.FacetGrid(n_sd_stats.reset_index(), hue=hue, col=col, col_wrap=3 if col is not None else None, sharex=False, sharey=False)

        g.map_dataframe(plot_prediction, x='x', y='sd')
        g.map_dataframe(plt.plot, 'x', 'sd')

        g.set(ylim=ylim)
        g.fig.set_size_inches(6, 6)

        return g

    def _get_paradigm(self, paradigm=None):

        paradigm_ = super()._get_paradigm(paradigm)

        paradigm_['n1'] = paradigm['n1'].values
        paradigm_['n2'] = paradigm['n2'].values

        return paradigm_

class FlexibleNoiseComparisonRegressionModel(RegressionModel, FlexibleNoiseComparisonModel):
    """FlexibleNoiseComparisonModel with patsy formula regression on noise spline coefficients."""

    def __init__(self, paradigm,
                 regressors,
                 fit_seperate_evidence_sd=True,
                 fit_prior=False,
                 polynomial_order=5,
                 memory_model='independent'):

        if (type(polynomial_order) is int) and fit_seperate_evidence_sd:
            polynomial_order = polynomial_order, polynomial_order


        for key in list(regressors.keys()):

            if key in ['evidence_sd', 'n1_evidence_sd', 'memory_noise', 'n2_evidence_sd', 'perceptual_noise']:

                if key in ['evidence_sd']:
                    po = polynomial_order
                elif key in ['n1_evidence_sd', 'memory_noise']:
                    po = polynomial_order[0]
                elif key in ['n2_evidence_sd', 'perceptual_noise']:
                    po = polynomial_order[1]

                warn(f'Found {key} in regressors, will add it for all {po} splines!')

                for i in range(1, po+1):
                    regressors[f'{key}_spline{i}'] = regressors[key]

                regressors.pop(key)


        RegressionModel.__init__(self, regressors)
        FlexibleNoiseComparisonModel.__init__(self, paradigm, fit_seperate_evidence_sd, fit_prior,
                                              polynomial_order, memory_model)
