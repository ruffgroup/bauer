import re
import pandas as pd
import pymc as pm
import numpy as np
from .utils import cumulative_normal, get_posterior, get_diff_dist
from .utils.math import inverse_softplus, softplus_np, inverse_softplus_np, logit_derivative, gaussian_pdf
from pymc.math import logit, invlogit
import pytensor.tensor as pt
from pytensor import scan    
from patsy import dmatrix
from .core import BaseModel, RegressionModel, LapseModel
from .utils.plotting import plot_prediction
from arviz import hdi
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


class MagnitudeComparisonModel(BaseModel):

    def __init__(self, data=None, fit_prior=False, fit_seperate_evidence_sd=True, save_trialwise_n_estimates=False):

        self.fit_prior = fit_prior
        self.fit_seperate_evidence_sd = fit_seperate_evidence_sd

        super().__init__(data, save_trialwise_n_estimates=save_trialwise_n_estimates)

    def get_model_inputs(self, parameters):

        model = pm.Model.get_context()

        model_inputs = {}

        if self.fit_prior:
            model_inputs['n1_prior_mu'] = parameters['prior_mu']
            model_inputs['n2_prior_mu'] = parameters['prior_mu']
        else:
            mean_prior = (pt.mean(pt.log(model['n1'])) + pt.mean(pt.log(model['n2']))) / 2.
            mean_std = (pt.std(pt.log(model['n1'])) + pt.std(pt.log(model['n2']))) / 2.

            model_inputs['n1_prior_mu'] = mean_prior
            model_inputs['n2_prior_mu'] = mean_prior

            model_inputs['n1_prior_std'] = mean_std
            model_inputs['n2_prior_std'] = mean_std

        model_inputs['n1_evidence_mu'] = model['log(n1)']
        model_inputs['n2_evidence_mu'] = model['log(n2)']

        model_inputs['threshold'] =  0.0

        if self.fit_seperate_evidence_sd:
            model_inputs['n1_evidence_sd'] = parameters['n1_evidence_sd']
            model_inputs['n2_evidence_sd'] = parameters['n2_evidence_sd']
        else:
            model_inputs['n1_evidence_sd'] = parameters['evidence_sd']
            model_inputs['n2_evidence_sd'] = parameters['evidence_sd']

        return model_inputs

    def get_free_parameters(self):

        free_parameters = {}

        if self.fit_seperate_evidence_sd:
            free_parameters['n1_evidence_sd'] = {'mu_intercept': -1., 'transform': 'softplus'}
            free_parameters['n2_evidence_sd'] = {'mu_intercept': -1., 'transform': 'softplus'}
        else:
            free_parameters['evidence_sd'] = {'mu_intercept': -1., 'transform': 'softplus'}

        if self.fit_prior:
            objective_mu = np.mean(np.log(np.stack((self.data['n1'], self.data['n2']))))
            objective_sd = np.mean(np.log(np.stack((self.data['n1'], self.data['n2']))))

            free_parameters['prior_mu'] = {'mu_intercept': objective_mu, 'transform': 'identity'}
            free_parameters['prior_sd'] = {'mu_intercept': objective_sd, 'transform': 'softplus'}


        return free_parameters

        
class MagnitudeComparisonLapseModel(LapseModel, MagnitudeComparisonModel):
    ...

class MagnitudeComparisonRegressionModel(RegressionModel, MagnitudeComparisonModel):
    def build_priors(self):

        super().build_priors()

        for key in ['n1_evidence_mu', 'n2_evidence_mu']:
            if key in self.regressors:
                self.build_hierarchical_nodes(key, mu_intercept=0.0, transform='identity')

class RiskModelProbabilityDistortion(BaseModel):

    def __init__(self, data=None, magnitude_prior_estimate='objective', save_trialwise_n_estimates=False, n_prospects=2,
                 p_grid_size=20, lapse_rate=0.01, distort_magnitudes=True, distort_probabilities=True,
                 fix_magnitude_prior_sd=False, fix_probabiliy_prior_sd=False,
                 estimate_magnitude_prior_mu=False):

        assert magnitude_prior_estimate in ['objective'], 'Only objective prior is currently supported'
        assert(n_prospects == 2), 'Only two prospects are currently supported'

        self.magnitude_prior_estimate = magnitude_prior_estimate
        self.n_prospects = n_prospects

        self.p_grid = np.linspace(1e-6, 1-1e-6, p_grid_size)
        self.lapse_rate = lapse_rate

        self.distort_magnitudes = distort_magnitudes
        self.distort_probabilities = distort_probabilities
        self.estimate_magnitude_prior_mu = estimate_magnitude_prior_mu

        self.fix_magnitude_prior_sd = fix_magnitude_prior_sd
        self.fix_probabiliy_prior_sd = fix_probabiliy_prior_sd

        if data is not None:
            for ix in range(self.n_prospects):
                assert(f'n{ix+1}' in data.columns), f'Data should contain columns n1, n2, ... n{self.n_prospects}'
                assert(f'p{ix+1}' in data.columns), f'Data should contain columns p1, p2, ... p{self.n_prospects}'

        super().__init__(data, save_trialwise_n_estimates=save_trialwise_n_estimates)


    def _get_paradigm(self, paradigm=None):

        if paradigm is None:
            paradigm = self.data

        paradigm_ = {}


        if np.in1d(paradigm['p1'], [0.0, 1.0]).any():
            raise ValueError('p1 contains 0 or 1, this is not supported by the current model (logodds(0) = -inf)\nHINT: You probably want to replace 0 by 1e-5 and 1 by 1-1e-5')

        if np.in1d(paradigm['p2'], [0.0, 1.0]).any():
            raise ValueError('p2 contains 0 or 1, this is not supported by the current model (logodds(0) = -inf)\nHINT: You probably want to replace 0 by 1e-5 and 1 by 1-1e-5')

        for ix in range(self.n_prospects):
            paradigm_[f'n{ix+1}'] = paradigm[f'n{ix+1}'].values
            paradigm_[f'p{ix+1}'] = paradigm[f'p{ix+1}'].values
            paradigm_[f'log(n{ix+1})'] = np.log(paradigm[f'n{ix+1}'].values)

        if 'subject' in paradigm.index.names:
            paradigm_['subject_ix'], _ = pd.factorize(paradigm.index.get_level_values('subject'))
        elif 'subject_ix' in paradigm.columns:
            paradigm_['subject_ix'], _ = pd.factorize(paradigm['subject_ix'])

        if 'choice' in paradigm.columns:
            paradigm_['choice'] = paradigm['choice'].values
        else:
            paradigm_['choice'] = np.zeros_like(paradigm['n1'].astype(bool))

        return paradigm_

    def _get_choice_predictions(self, model_inputs):

        def logodds_dist_in_p(mu_logodds, sd_logodds, p_grid=self.p_grid, normalize=True):

            logodds_grid = logit(p_grid)[np.newaxis, :]

            p_logodds = gaussian_pdf(logodds_grid, mu_logodds[:, np.newaxis], sd_logodds[:, np.newaxis]) * logit_derivative(p_grid)

            if normalize:
                p_logodds = p_logodds / pt.sum(p_logodds, 1, keepdims=True)

            return p_logodds * logit_derivative(p_grid)


        if self.distort_probabilities:
            posteriors = {}
            for ix in range(self.n_prospects):
                posteriors[f'p{ix+1}_posterior_mu'], posteriors[f'p{ix+1}_posterior_sd'] = get_posterior(model_inputs[f'p{ix+1}_evidence_mu'], model_inputs[f'p{ix+1}_evidence_sd'], model_inputs[f'p{ix+1}_prior_mu'], model_inputs[f'p{ix+1}_prior_sd'])

                if posteriors[f'p{ix+1}_posterior_sd'].ndim == 0:
                    posteriors[f'p{ix+1}_posterior_sd'] = posteriors[f'p{ix+1}_posterior_sd'][np.newaxis]

            ix = 0
            p_posterior1 = logodds_dist_in_p(posteriors[f'p{ix+1}_posterior_mu'], posteriors[f'p{ix+1}_posterior_sd'])
            ix = 1
            p_posterior2 = logodds_dist_in_p(posteriors[f'p{ix+1}_posterior_mu'], posteriors[f'p{ix+1}_posterior_sd'])

            # n x p1 x p2
            p_posterior_joint = p_posterior1[:, :,np.newaxis] * p_posterior2[:, np.newaxis, :]
            p_posterior_joint = p_posterior_joint / pt.sum(p_posterior_joint, (1, 2), keepdims=True)

        if self.distort_magnitudes:
            ix = 0
            n1_hat_mean, n1_hat_sd = get_posterior(model_inputs[f'n{ix+1}_evidence_mu'], model_inputs[f'n{ix+1}_evidence_sd'], model_inputs[f'n{ix+1}_prior_mu'], model_inputs[f'n{ix+1}_prior_sd'])

            ix = 1
            n2_hat_mean, n2_hat_sd = get_posterior(model_inputs[f'n{ix+1}_evidence_mu'], model_inputs[f'n{ix+1}_evidence_sd'], model_inputs[f'n{ix+1}_prior_mu'], model_inputs[f'n{ix+1}_prior_sd']) 

        if self.distort_magnitudes & self.distort_probabilities:

            ev1_hat_mean = n1_hat_mean[:, np.newaxis] + pt.log(self.p_grid)[np.newaxis, :]
            ev2_hat_mean = n2_hat_mean[:, np.newaxis] + pt.log(self.p_grid)[np.newaxis, :]

            ev_diff_mean = ev2_hat_mean[:, np.newaxis, :] - ev1_hat_mean[:, :, np.newaxis]

            ev_diff_sd = pt.sqrt(n1_hat_sd**2 + n2_hat_sd**2)

            if ev_diff_sd.ndim == 0:
                ev_diff_sd = ev_diff_sd[np.newaxis]

            p_choice = cumulative_normal(ev_diff_mean, 0.0, ev_diff_sd[:, np.newaxis, np.newaxis])
            p_choice = pt.sum(pt.sum(p_posterior_joint * p_choice, 1), 1)

        elif self.distort_magnitudes & (~self.distort_probabilities):
            p1 = invlogit(model_inputs[f'p1_evidence_mu'])
            p2 = invlogit(model_inputs[f'p2_evidence_mu'])

            ev1_hat_mean = n1_hat_mean + pt.log(p1)
            ev2_hat_mean = n2_hat_mean + pt.log(p2)

            ev_diff_mean = ev2_hat_mean - ev1_hat_mean

            ev_diff_sd = pt.sqrt(n1_hat_sd**2 + n2_hat_sd**2)

            p_choice = cumulative_normal(ev_diff_mean, 0.0, ev_diff_sd)

        elif (~self.distort_magnitudes) & self.distort_probabilities:

            n1 = model_inputs[f'n1_evidence_mu']
            n2 = model_inputs[f'n2_evidence_mu']

            ev1 = n1[:, np.newaxis, np.newaxis] + pt.log(self.p_grid)[np.newaxis, :, np.newaxis]
            ev2 = n2[:, np.newaxis, np.newaxis] + pt.log(self.p_grid)[np.newaxis, np.newaxis, :]

            p_choice = pt.clip(pt.sum((ev2 > ev1) * p_posterior_joint, (1,2)), 1e-6, 1-1e-6)

        else:
            raise NotImplementedError('At least probabilities or magnitudes should be distorted.')


        clip_range = self.lapse_rate / 2., 1 - self.lapse_rate / 2.
        p_choice = pt.clip(p_choice, clip_range[0], clip_range[1])

        return p_choice

    def get_model_inputs(self, parameters):

        model = pm.Model.get_context()

        model_inputs = {}

        for ix in range(self.n_prospects):
            # Magnitudes
            model_inputs[f'n{ix+1}_evidence_mu'] = model[f'log(n{ix+1})']
            model_inputs[f'p{ix+1}_evidence_mu'] = logit(model[f'p{ix+1}'])

            if self.distort_magnitudes:
                if self.fix_magnitude_prior_sd:
                    model_inputs[f'n{ix+1}_prior_sd'] = pt.std(model[f'log(n{ix+1})'])
                else:
                    model_inputs[f'n{ix+1}_prior_sd'] = parameters['magnitude_prior_sd']

                model_inputs[f'n{ix+1}_evidence_sd'] = parameters['magnitude_evidence_sd']

                if self.estimate_magnitude_prior_mu:
                    model_inputs[f'n{ix+1}_prior_mu'] = parameters['magnitude_prior_mu']
                else:
                    model_inputs[f'n{ix+1}_prior_mu'] = pt.mean(pt.log(pt.stack((model['n1'], model['n2']), 0)))

            if self.distort_probabilities:
                if self.fix_probabiliy_prior_sd:
                    model_inputs[f'p{ix+1}_prior_sd'] = pt.std(logit(model[f'p{ix+1}']))
                else:
                    model_inputs[f'p{ix+1}_prior_sd'] = parameters['probability_prior_sd']

                model_inputs[f'p{ix+1}_prior_mu'] = parameters['probability_prior_mu']
                model_inputs[f'p{ix+1}_evidence_sd'] = parameters['probability_evidence_sd']

        return model_inputs

    def get_free_parameters(self):

        free_parameters = {}

        if self.distort_magnitudes:
            free_parameters['magnitude_evidence_sd'] = {'mu_intercept': -1., 'transform': 'softplus'}

            if self.data is not None:
                prior_mu = np.log(np.stack([self.data[f'n{ix+1}'].values for ix in range(self.n_prospects)])).mean()
                prior_std = np.log(np.stack([self.data[f'n{ix+1}'].values for ix in range(self.n_prospects)])).std()
            else:
                prior_mu = np.log(10)
                prior_std = np.log(30)

            if self.estimate_magnitude_prior_mu:
                free_parameters['magnitude_prior_mu'] = {'mu_intercept': prior_mu, 'transform': 'softplus'}

            if not self.fix_magnitude_prior_sd:
                free_parameters['magnitude_prior_sd'] = {'mu_intercept': prior_std, 'transform': 'softplus'}

        if self.distort_probabilities:
            if not self.fix_probabiliy_prior_sd:
                free_parameters['probability_prior_sd'] = {'mu_intercept': -1., 'transform': 'softplus'}

            free_parameters['probability_evidence_sd'] = {'mu_intercept': -1., 'transform': 'softplus'}
            free_parameters['probability_prior_mu'] = {'mu_intercept': 0.0, 'transform': 'identity'}

        return free_parameters

class LossAversionModel(BaseModel):

    paradigm_keys = ['p1', 'p2', 'gain1', 'gain2', 'loss1', 'loss2']
    base_parameters = ['prior_mu_gains', 'prior_mu_losses', 'evidence_sd_gains', 'evidence_sd_losses', 'prior_sd_gains', 'prior_sd_losses']

    def __init__(self, data=None, save_trialwise_n_estimates=False, 
                 magnitude_grid=None,
                 ev_diff_grid=None,
                 lapse_rate=0.01, 
                 normalize_likelihoods=True,
                 fix_prior_sds=True):

        if magnitude_grid is None:
            self.magnitude_grid = np.linspace(1, 100, 50) 
        else:
            self.magnitude_grid = magnitude_grid

        if ev_diff_grid is None:
            self.ev_diff_grid = np.linspace(-50, 50, 50)
        else:
            self.ev_diff_grid = ev_diff_grid

        self.lapse_rate = lapse_rate
        self.fix_prior_sds = fix_prior_sds

        self.normalize_likelihoods = normalize_likelihoods

        super().__init__(data, save_trialwise_n_estimates=save_trialwise_n_estimates)


    def _get_paradigm(self, paradigm=None):

        if paradigm is None:
            paradigm = self.data

        paradigm_ = {}

        for key in self.paradigm_keys:
            paradigm_[key] = paradigm[key].values

        if 'subject' in paradigm.index.names:
            paradigm_['subject_ix'], _ = pd.factorize(paradigm.index.get_level_values('subject'))
        elif 'subject_ix' in paradigm.columns:
            paradigm_['subject_ix'], _ = pd.factorize(paradigm['subject_ix'])

        if 'choice' in paradigm.columns:
            paradigm_['choice'] = paradigm['choice'].values
        else:
            paradigm_['choice'] = np.zeros_like(paradigm['gain1'].astype(bool))

        return paradigm_

    def get_free_parameters(self):

        free_parameters = {}

        free_parameters['prior_mu_gains'] = {'mu_intercept': np.log(10.), 'sigma_intercept':3., 'transform': 'identity'}
        free_parameters['prior_mu_losses'] = {'mu_intercept': np.log(10.), 'sigma_intercept':3., 'transform': 'identity'}

        free_parameters['evidence_sd_gains'] = {'mu_intercept': -1., 'transform': 'softplus'}
        free_parameters['evidence_sd_losses'] = {'mu_intercept': -1., 'transform': 'softplus'}

        if not self.fix_prior_sds:
            free_parameters['prior_sd_gains'] = {'mu_intercept': 1., 'sigma_intercept':1., 'transform': 'softplus'}
            free_parameters['prior_sd_losses'] = {'mu_intercept': 1., 'sigma_intercept':1., 'transform': 'softplus'}

        return free_parameters


    def _get_choice_predictions(self, model_inputs):

        n_grid = pt.constant(self.magnitude_grid)
        n_grid_dx = n_grid[1] - n_grid[0]
        n_grid_log = pt.log(n_grid)

        ev_diff_grid = pt.constant(self.ev_diff_grid)
        ev_diff_grid_dx = ev_diff_grid[1] - ev_diff_grid[0]

        p1 = model_inputs['p1']
        p2 = model_inputs['p2']

        gains1 = model_inputs['gain1']
        losses1 = model_inputs['loss1']
        gains2 = model_inputs['gain2']
        losses2 = model_inputs['loss2']

        # Calculate the distributions of expectations in log space
        expectations_gains1_mu_log, expectations_gains1_sd_log = get_posterior(model_inputs['prior_mu_gains'], model_inputs['prior_sd_gains'], pt.log(gains1), model_inputs['evidence_sd_gains'])
        expectations_losses1_mu_log, expectations_losses1_sd_log = get_posterior(model_inputs['prior_mu_losses'], model_inputs['prior_sd_losses'], pt.log(losses1), model_inputs['evidence_sd_losses'])

        expectations_gains2_mu_log, expectations_gains2_sd_log = get_posterior(model_inputs['prior_mu_gains'], model_inputs['prior_sd_gains'], pt.log(gains2), model_inputs['evidence_sd_gains'])
        expectations_losses2_mu_log, expectations_losses2_sd_log = get_posterior(model_inputs['prior_mu_losses'], model_inputs['prior_sd_losses'], pt.log(losses2), model_inputs['evidence_sd_losses'])

        expectations_gains1_sd_log = pt.atleast_1d(expectations_gains1_sd_log)
        expectations_losses1_sd_log = pt.atleast_1d(expectations_losses1_sd_log)
        expectations_gains2_sd_log = pt.atleast_1d(expectations_gains2_sd_log)
        expectations_losses2_sd_log = pt.atleast_1d(expectations_losses2_sd_log)

        # Calculate the distributions of gains in natural space (n trials x n grid)
        # NOTE: These PDFs are normalized, they SUM to 1, not integrate to 1
        gains1_pdf = gaussian_pdf(n_grid_log[np.newaxis, :], expectations_gains1_mu_log[:, np.newaxis], expectations_gains1_sd_log[:, np.newaxis]) / n_grid * n_grid_dx
        losses1_pdf = gaussian_pdf(n_grid_log[np.newaxis, :], expectations_losses1_mu_log[:, np.newaxis], expectations_losses1_sd_log[:, np.newaxis]) / n_grid * n_grid_dx

        gains2_pdf = gaussian_pdf(n_grid_log[np.newaxis, :], expectations_gains2_mu_log[:, np.newaxis], expectations_gains2_sd_log[:, np.newaxis]) / n_grid * n_grid_dx
        losses2_pdf = gaussian_pdf(n_grid_log[np.newaxis, :], expectations_losses2_mu_log[:, np.newaxis], expectations_losses2_sd_log[:, np.newaxis]) / n_grid * n_grid_dx

        if self.normalize_likelihoods:
            gains1_pdf = gains1_pdf / pt.sum(gains1_pdf, 1, keepdims=True)
            losses1_pdf = losses1_pdf / pt.sum(losses1_pdf, 1, keepdims=True)
            gains2_pdf = gains2_pdf / pt.sum(gains2_pdf, 1, keepdims=True)
            losses2_pdf = losses2_pdf / pt.sum(losses2_pdf, 1, keepdims=True)

        # joint gain/loss distribution n_trials x n_grid (gains) x n_grid (losses)
        joint_pdf1 = gains1_pdf[:, :, np.newaxis] * losses1_pdf[:, np.newaxis, :]
        joint_pdf2 = gains2_pdf[:, :, np.newaxis] * losses2_pdf[:, np.newaxis, :]

        # ev_grids: n_trials x n_grid
        gains1_ev_grid = p1[:, np.newaxis]*n_grid[np.newaxis, :]
        losses1_ev_grid = (1-p1)[:, np.newaxis]*n_grid[np.newaxis, :]
        gains2_ev_grid = p2[:, np.newaxis]*n_grid[np.newaxis, :]
        losses2_ev_grid = (1-p2)[:, np.newaxis]*n_grid[np.newaxis, :]

        # ev_grids: n_trials x n_grid (gains) x n_grid (losses)
        evs1 = gains1_ev_grid[:, :, np.newaxis] - losses1_ev_grid[:, np.newaxis, :]
        evs2 = gains2_ev_grid[:, :, np.newaxis] - losses2_ev_grid[:, np.newaxis, :]

        # Discretize the joint_pdf1, it *SUMS* to one (not integral)
        # joint_pdf1 /= n_grid_dx**2
        # joint_pdf2 /= n_grid_dx**2

        # n_ev_diff_grid x n_trials x n_grid (gains) x n_grid (losses)
        ev1_diff_mapping, _ = scan(lambda bin_index, evs1, ev_diff_grid: (evs1 >= ev_diff_grid[bin_index]) & (evs1 < ev_diff_grid[bin_index+1]),
                                    sequences=[pt.arange(ev_diff_grid.shape[0]-1, dtype=int)],
                                    non_sequences=[evs1, ev_diff_grid])

        # Distribution o er expecrtations of the expected value of first option (n_trials x n_diff_grid)
        # ev1_pdf, _ = scan(lambda bin_index, evs1, ev_diff_grid, joint_pdf1: pt.sum(joint_pdf1 * ((evs1 >= ev_diff_grid[bin_index]) & (evs1 < ev_diff_grid[bin_index+1]) ), axis=[-2, -1]),
        #                 sequences=[pt.arange(len(ev_diff_grid)-1, dtype=int)], 
        #                 non_sequences=[evs1, ev_diff_grid, joint_pdf1])
        
        ev1_pdf, _ = scan(lambda ev_diff_mapping_, joint_pdf1: pt.sum(joint_pdf1 * ev_diff_mapping_, axis=[-2, -1]),
                        sequences=[ev1_diff_mapping],
                        non_sequences=[joint_pdf1])
        ev1_pdf = pt.transpose(ev1_pdf)

        # Distribution o er expecrtations of the expected value of first option (n_trials x n_diff_grid)
        ev2_diff_mapping, _ = scan(lambda bin_index, evs2, ev_diff_grid: (evs2 >= ev_diff_grid[bin_index]) & (evs2 < ev_diff_grid[bin_index+1]),
                                sequences=[pt.arange(ev_diff_grid.shape[0]-1, dtype=int)],
                                non_sequences=[evs2, ev_diff_grid])

        # ev2_pdf, _ = scan(lambda bin_index, evs2, ev_diff_grid, joint_pdf2: pt.sum(joint_pdf2 * ((evs2 >= ev_diff_grid[bin_index]) & (evs2 < ev_diff_grid[bin_index+1]) ), axis=[-2, -1]),
        #                 sequences=[pt.arange(len(ev_diff_grid)-1, dtype=int)], 
        #                 non_sequences=[evs2, ev_diff_grid, joint_pdf2])

        ev2_pdf, _ = scan(lambda ev_diff_mapping_, joint_pdf2: pt.sum(joint_pdf2 * ev_diff_mapping_, axis=[-2, -1]),
                        sequences=[ev2_diff_mapping],
                        non_sequences=[joint_pdf2])

        ev2_pdf = pt.transpose(ev2_pdf)

        # Joint distribution over ev1 and ev2 (n_trials x n_diff_grid x n_diff_grid)
        joint_ev_pdf = ev1_pdf[:, :, np.newaxis] * ev2_pdf[:, np.newaxis, :]

        # Calculate the probability of choosing the second option
        centers_of_ev_diff_bins = ev_diff_grid[:-1] + ev_diff_grid_dx/2
        choose2 = centers_of_ev_diff_bins[np.newaxis, :, np.newaxis] < centers_of_ev_diff_bins[np.newaxis, np.newaxis, :]

        # Integrate over unique_evs1
        p_choose2 = pt.clip(pt.sum(joint_ev_pdf * choose2, axis=[-2, -1]), 1e-6, 1-1e-6)

        return p_choose2

    def get_model_inputs(self, parameters):
        model = pm.Model.get_context()

        model_inputs = {}

        for key in self.base_parameters:
            if key in parameters:
                model_inputs[key] = parameters[key]
            
        if self.fix_prior_sds:
            for key in ['prior_sd_gains', 'prior_sd_losses']:
                if key not in parameters:
                    model_inputs['prior_sd_gains'] = 1.
                    model_inputs['prior_sd_losses'] = 1.
        
        for key in self.paradigm_keys:
            model_inputs[key] = model[key]

        return model_inputs

class RiskModel(BaseModel):

    def __init__(self, data, prior_estimate='objective', fit_seperate_evidence_sd=True, incorporate_probability='after_inference',
                 save_trialwise_n_estimates=False, memory_model='independent', n_prospects=2):

        assert prior_estimate in ['objective', 'shared', 'different', 'full', 'full_normed']

        self.fit_seperate_evidence_sd = fit_seperate_evidence_sd
        self.memory_model = memory_model
        self.prior_estimate = prior_estimate
        self.incorporate_probability = incorporate_probability

        if 'risky_first' not in data.columns:
            data['risky_first'] = data['p1'] != 1.0

        super().__init__(data, save_trialwise_n_estimates=save_trialwise_n_estimates)

    def _get_paradigm(self, data=None):


        paradigm = super()._get_paradigm(data)

        paradigm['p1'] = data['p1'].values
        paradigm['p2'] = data['p2'].values
        paradigm['risky_first'] = data['risky_first'].values.astype(bool)

        return paradigm

    def get_model_inputs(self):

        model = pm.Model.get_context()

        model_inputs = {}
        
        model_inputs['n1_evidence_mu'] = self.get_trialwise_variable('n1_evidence_mu', transform='identity') #at.log(model['n1'])
        model_inputs['n2_evidence_mu'] = self.get_trialwise_variable('n2_evidence_mu', transform='identity') #at.log(model['n2'])

        # Prob of choosing 2 should increase with p2
        if self.incorporate_probability == 'after_inference':
            model_inputs['threshold'] =  pt.log(model['p2'] / model['p1'])
        elif self.incorporate_probability == 'before_inference':
            model_inputs['threshold'] =  0.0
            model_inputs['n1_evidence_mu'] += pt.log(model['p1'])
            model_inputs['n2_evidence_mu'] += pt.log(model['p2'])
        else:
            raise ValueError('incorporate_probability should be either "after_inference" (default) or "before_inference"')

        if self.prior_estimate == 'objective':
            model_inputs['n1_prior_mu'] = pt.mean(pt.log(pt.stack((model['n1'], model['n2']), 0)))
            model_inputs['n1_prior_std'] = pt.std(pt.log(pt.stack((model['n1'], model['n2']), 0)))
            model_inputs['n2_prior_mu'] = model_inputs['n1_prior_mu']
            model_inputs['n2_prior_std'] = model_inputs['n1_prior_std']

        elif self.prior_estimate == 'shared':
            model_inputs['n1_prior_mu'] = self.get_trialwise_variable('prior_mu', transform='identity')
            model_inputs['n1_prior_std'] = self.get_trialwise_variable('prior_std', transform='softplus')
            model_inputs['n2_prior_mu'] = model_inputs['n1_prior_mu']
            model_inputs['n2_prior_std'] = model_inputs['n1_prior_std']

        elif self.prior_estimate == 'two_mus':

            risky_first = model['risky_first'].astype(bool)
            safe_n = pt.where(risky_first, model['n2'], model['n1'])
            safe_prior_std = pt.std(pt.log(safe_n))
            risky_prior_std = self.get_trialwise_variable('risky_prior_std', transform='softplus')

            model_inputs['n1_prior_mu'] = self.get_trialwise_variable('n1_prior_mu', transform='identity')
            model_inputs['n1_prior_std'] = pt.where(risky_first, risky_prior_std, safe_prior_std)
            model_inputs['n2_prior_mu'] = self.get_trialwise_variable('n2_prior_mu', transform='identity')
            model_inputs['n2_prior_std'] = pt.where(risky_first, safe_prior_std, risky_prior_std)

        elif self.prior_estimate == 'different':

            risky_first = model['risky_first'].astype(bool)

            safe_n = pt.where(risky_first, model['n2'], model['n1'])
            safe_prior_mu = pt.mean(pt.log(safe_n))
            safe_prior_std = pt.std(pt.log(safe_n))

            risky_prior_mu = self.get_trialwise_variable('risky_prior_mu', transform='identity')
            risky_prior_std = self.get_trialwise_variable('risky_prior_std', transform='softplus')

            model_inputs['n1_prior_mu'] = pt.where(risky_first, risky_prior_mu, safe_prior_mu)
            model_inputs['n1_prior_std'] = pt.where(risky_first, risky_prior_std, safe_prior_std)

            model_inputs['n2_prior_mu'] = pt.where(risky_first, safe_prior_mu, risky_prior_mu)
            model_inputs['n2_prior_std'] = pt.where(risky_first, safe_prior_std, risky_prior_std)

        elif self.prior_estimate in ['full', 'full_normed']:

            risky_first = model['risky_first'].astype(bool)

            risky_prior_mu = self.get_trialwise_variable('risky_prior_mu', transform='identity')
            risky_prior_std = self.get_trialwise_variable('risky_prior_std', transform='softplus')

            safe_prior_mu = self.get_trialwise_variable('safe_prior_mu', transform='identity')
            
            if self.prior_estimate == 'full_normed':
                safe_prior_std = 1.
            else:
                safe_prior_std = self.get_trialwise_variable('safe_prior_std', transform='softplus')

            model_inputs['n1_prior_mu'] = pt.where(risky_first, risky_prior_mu, safe_prior_mu)
            model_inputs['n1_prior_std'] = pt.where(risky_first, risky_prior_std, safe_prior_std)

            model_inputs['n2_prior_mu'] = pt.where(risky_first, safe_prior_mu, risky_prior_mu)
            model_inputs['n2_prior_std'] = pt.where(risky_first, safe_prior_std, risky_prior_std)

        if self.fit_seperate_evidence_sd:

            if self.memory_model == 'independent':
                model_inputs['n1_evidence_sd'] = self.get_trialwise_variable('n1_evidence_sd', transform='softplus')
                model_inputs['n2_evidence_sd'] = self.get_trialwise_variable('n2_evidence_sd', transform='softplus')
            elif self.memory_model == 'shared_perceptual_noise':
                perceptual_sd = self.get_trialwise_variable('perceptual_noise_sd', transform='softplus')
                memory_sd = self.get_trialwise_variable('memory_noise_sd', transform='softplus')

                model_inputs['n1_evidence_sd'] = perceptual_sd + memory_sd
                model_inputs['n2_evidence_sd'] = perceptual_sd
            else:
                raise ValueError('Unknown memory model: {}'.format(self.memory_model))

        else:
            model_inputs['n1_evidence_sd'] = self.get_trialwise_variable('evidence_sd', transform='softplus')
            model_inputs['n2_evidence_sd'] = self.get_trialwise_variable('evidence_sd', transform='softplus')

        return model_inputs

    def build_priors(self):

        if self.fit_seperate_evidence_sd:
            if self.memory_model == 'independent':
                self.build_hierarchical_nodes('n1_evidence_sd', mu_intercept=-1., transform='softplus')
                self.build_hierarchical_nodes('n2_evidence_sd', mu_intercept=-1., transform='softplus')
            elif self.memory_model == 'shared_perceptual_noise':
                self.build_hierarchical_nodes('perceptual_noise_sd', mu_intercept=-1., transform='softplus')
                self.build_hierarchical_nodes('memory_noise_sd', mu_intercept=-1., transform='softplus')
            else:
                raise ValueError('Unknown memory model: {}'.format(self.memory_model))
        else:
            self.build_hierarchical_nodes('evidence_sd', mu_intercept=-1., transform='softplus')

        model = pm.Model.get_context() 

        if self.prior_estimate == 'shared':
            prior_mu = np.mean(np.log(np.stack((self.data['n1'], self.data['n2']))))
            prior_std = np.mean(np.log(np.stack((self.data['n1'], self.data['n2']))))
            self.build_hierarchical_nodes('prior_mu', mu_intercept=prior_mu, transform='identity')
            self.build_hierarchical_nodes('prior_std', mu_intercept=prior_std, transform='softplus')

        elif self.prior_estimate == 'different':
            risky_n = np.where(self.data['risky_first'], self.data['n1'], self.data['n2'])

            risky_prior_mu = np.mean(np.log(risky_n))
            risky_prior_std = np.std(np.log(risky_n))

            self.build_hierarchical_nodes('risky_prior_mu', mu_intercept=risky_prior_mu, transform='identity')
            self.build_hierarchical_nodes('risky_prior_std', mu_intercept=risky_prior_std, transform='softplus')

        elif self.prior_estimate in ['full', 'full_normed']:
            risky_n = np.where(self.data['risky_first'], self.data['n1'], self.data['n2'])
            safe_n = np.where(self.data['risky_first'], self.data['n2'], self.data['n1'])

            risky_prior_mu = np.mean(np.log(risky_n))
            risky_prior_std = np.std(np.log(risky_n))

            self.build_hierarchical_nodes('risky_prior_mu', mu_intercept=risky_prior_mu, transform='identity')
            self.build_hierarchical_nodes('risky_prior_std', mu_intercept=risky_prior_std, transform='softplus')

            safe_prior_mu = np.mean(np.log(safe_n))

            self.build_hierarchical_nodes('safe_prior_mu', mu_intercept=safe_prior_mu, transform='identity')

            if self.prior_estimate == 'full':
                safe_prior_std = np.std(np.log(safe_n))
                self.build_hierarchical_nodes('safe_prior_std', mu_intercept=safe_prior_std, transform='softplus')

class RiskRegressionModel(RegressionModel, RiskModel):

    def __init__(self,  data, regressors, prior_estimate='objective', fit_seperate_evidence_sd=True, incorporate_probability='after_inference',
                 save_trialwise_n_estimates=False, memory_model='independent'):
        RegressionModel.__init__(self, data, regressors)
        RiskModel.__init__(self, data, prior_estimate, fit_seperate_evidence_sd, incorporate_probability=incorporate_probability,
                           save_trialwise_n_estimates=save_trialwise_n_estimates, memory_model=memory_model)

    def build_priors(self):

        super().build_priors()

        for key in ['n1_evidence_mu', 'n2_evidence_mu', 'evidence_sd_diff', 'evidence_mu_diff']:
            if key in self.regressors:
                self.build_hierarchical_nodes(key, mu_intercept=0.0, transform='identity')

        if (self.prior_estimate in ['full', 'full_normed']) and ('prior_mu' in self.regressors):
            print("***Warning, estimating both risky and safe priors, but (some) regressors affect both equally (via `prior_mu`)***")
            self.build_hierarchical_nodes('prior_mu', mu_intercept=0.0, transform='identity')

        if (self.fit_seperate_evidence_sd) and ('evidence_sd' in self.regressors):
            print("***Warning, estimating evidence_sd for both first and second option, but (some) regressors affect both equally (via `evidence_sd`)***")
            self.build_hierarchical_nodes('evidence_sd', mu_intercept=0.0, transform='identity')


    def get_trialwise_variable(self, key, transform='identity'):

        # Prior mean
        if (key == 'risky_prior_mu') and ('prior_mu' in self.regressors):
            return super().get_trialwise_variable('risky_prior_mu', transform='identity') + super().get_trialwise_variable('prior_mu', transform='identity')

        if (key == 'safe_prior_mu') and ('prior_mu' in self.regressors):
            return super().get_trialwise_variable('safe_prior_mu', transform='identity') + super().get_trialwise_variable('prior_mu', transform='identity')

        # Evidence SD
        if (key == 'n1_evidence_sd') and ('evidence_sd' in self.regressors):
            return pt.softplus(super().get_trialwise_variable('n1_evidence_sd', transform='identity') + super().get_trialwise_variable('evidence_sd', transform='identity'))

        if (key == 'n2_evidence_sd') and ('evidence_sd' in self.regressors):
            return pt.softplus(super().get_trialwise_variable('n2_evidence_sd', transform='identity') + super().get_trialwise_variable('evidence_sd', transform='identity'))

        if (key == 'n1_evidence_sd') and ('evidence_sd_diff' in self.regressors):
            return pt.softplus(super().get_trialwise_variable('n1_evidence_sd', transform='identity') + super().get_trialwise_variable('evidence_sd_diff', transform='identity'))

        if (key == 'n2_evidence_sd') and ('evidence_sd_diff' in self.regressors):
            return pt.softplus(super().get_trialwise_variable('n2_evidence_sd', transform='identity') - super().get_trialwise_variable('evidence_sd_diff', transform='identity'))

        if (key == 'n1_evidence_mu') and ('evidence_mu_diff' in self.regressors):
            return super().get_trialwise_variable('n1_evidence_mu', transform='identity') + super().get_trialwise_variable('evidence_mu_diff', transform='identity')

        if (key == 'n2_evidence_mu') and ('evidence_mu_diff' in self.regressors):
            return super().get_trialwise_variable('n2_evidence_mu', transform='identity') - super().get_trialwise_variable('evidence_mu_diff', transform='identity')

        return super().get_trialwise_variable(key=key, transform=transform)


class RiskLapseModel(RiskModel, LapseModel):

    def build_priors(self):
        RiskModel.build_priors(self)
        self.build_hierarchical_nodes('p_lapse', mu_intercept=-4, transform='logistic')

    def get_model_inputs(self):
        model_inputs = RiskModel.get_model_inputs(self)
        model_inputs['p_lapse'] = self.get_trialwise_variable('p_lapse', transform='logistic')
        return model_inputs

class RiskLapseRegressionModel(RegressionModel, RiskLapseModel):
    def __init__(self,  data, regressors, prior_estimate='objective', fit_seperate_evidence_sd=True):
        RegressionModel.__init__(self, data, regressors)
        RiskLapseModel.__init__(self, data, prior_estimate, fit_seperate_evidence_sd)


class RNPModel(BaseModel):

    def __init__(self, data, risk_neutral_p=0.55):
        self.risk_neutral_p = risk_neutral_p

        super().__init__(data)

    def get_model_inputs(self):

        model = pm.Model.get_context()

        model_inputs = {}
        model_inputs['n1_evidence_mu'] = self.get_trialwise_variable('n1_evidence_mu', transform='identity') #model['n1'])
        model_inputs['n2_evidence_mu'] = self.get_trialwise_variable('n2_evidence_mu', transform='identity')
        model_inputs['rnp'] = self.get_trialwise_variable('rnp', transform='logistic')
        model_inputs['gamma'] = self.get_trialwise_variable('gamma', transform='identity')

        return model_inputs

    def _get_paradigm(self, data=None):

        paradigm = super()._get_paradigm(data)

        paradigm['p1'] = data['p1'].values
        paradigm['p2'] = data['p2'].values
        paradigm['risky_first'] = data['risky_first'].values.astype(bool)

        return paradigm

    def _get_choice_predictions(self, model_inputs):

        model = pm.Model.get_context()

        rnp = model_inputs['rnp']
        slope = model_inputs['gamma']
        risky_first = model['risky_first'].astype(bool)

        intercept = -pt.log(rnp) * slope # More risk-seeking -> higher rnp, smaller intercept, more likely to choose option 2
        intercept = pt.where(risky_first, intercept, -intercept)
        n1 = model_inputs['n1_evidence_mu']
        n2 = model_inputs['n2_evidence_mu']

        p1, p2 =  model['p1'], model['p2']

        return cumulative_normal(intercept + slope*(n2-n1), 0.0, 1.0)

    def build_priors(self):
        self.build_hierarchical_nodes('gamma', mu_intercept=1.0, sigma_intercept=0.5, transform='identity')
        self.build_hierarchical_nodes('rnp', mu_intercept=0.0, sigma_intercept=1.0, transform='logistic')

class RNPRegressionModel(RegressionModel, RNPModel):
    def __init__(self,  data, regressors, risk_neutral_p=0.55):
        RegressionModel.__init__(self, data, regressors=regressors)
        RNPModel.__init__(self, data, risk_neutral_p)


class FlexibleSDComparisonModel(BaseModel):


    def __init__(self, data, fit_seperate_evidence_sd=True,
                 fit_n2_prior_mu=False,
                 polynomial_order=5,
                 bspline=False,
                 memory_model='independent'):

        self.fit_n2_prior_mu = fit_n2_prior_mu
        self.fit_seperate_evidence_sd = fit_seperate_evidence_sd
        
        if (type(polynomial_order) is int) and fit_seperate_evidence_sd:
            polynomial_order = polynomial_order, polynomial_order

        self.polynomial_order = polynomial_order
        self.max_polynomial_order = np.max(self.polynomial_order)
        self.bspline = bspline
        self.memory_model = memory_model

        super().__init__(data)


    def build_estimation_model(self, data=None, coords=None):
        coords = {'subject': self.unique_subjects, 'order':['first', 'second']}
        coords['poly_order'] = np.arange(self.max_polynomial_order)

        return BaseModel.build_estimation_model(self, data=data, coords=coords)

    def get_model_inputs(self):

        model = pm.Model.get_context()

        model_inputs = {}

        model_inputs['n1_prior_mu'] = pt.mean(model['n1'])
        model_inputs['n1_prior_std'] = pt.std(model['n1'])
        model_inputs['n2_prior_std'] = pt.std(model['n2'])
        model_inputs['threshold'] =  0.0

        model_inputs['n1_evidence_mu'] = self.get_trialwise_variable('n1_evidence_mu')
        model_inputs['n2_evidence_mu'] = self.get_trialwise_variable('n2_evidence_mu')

        if hasattr('self', 'fit_n2_prior_mu') and self.fit_n2_prior_mu:
            model_inputs['n2_prior_mu'] = self.get_trialwise_variable('n2_prior_mu', transform='identity')
        else:
            model_inputs['n2_prior_mu'] = pt.mean(model['n2'])

        model_inputs['n1_evidence_sd'] = self.get_trialwise_variable('n1_evidence_sd')
        model_inputs['n2_evidence_sd'] = self.get_trialwise_variable('n2_evidence_sd')

        return model_inputs


    def build_priors(self):

        model = pm.Model.get_context()

        if self.bspline:

            if self.fit_seperate_evidence_sd:
                mu_prior = [np.ones(po) * inverse_softplus_np(5.) for po in self.polynomial_order]
                std_prior = [np.ones(po) * 5 for po in self.polynomial_order]
            else:
                mu_prior = np.ones(self.polynomial_order) * inverse_softplus_np(5.)
                std_prior = np.ones(self.polynomial_order) * 5

            cauchy_sigma= .5
        else:
            mu_prior = np.concatenate([[10], np.zeros(self.polynomial_order-1)])
            std_prior = np.concatenate([[10], 10**(-np.arange(1, self.polynomial_order) - 1)])
            cauchy_sigma = 0.25

        if self.fit_seperate_evidence_sd:

            key1, key2 = self._get_evidence_sd_labels()

            for n in range(self.polynomial_order[0]):
                self.build_hierarchical_nodes(f'{key1}_poly{n}', mu_intercept=mu_prior[0][n], sigma_intercept=std_prior[0][n], cauchy_sigma_intercept=cauchy_sigma, transform='identity')

            for n in range(self.polynomial_order[1]):
                self.build_hierarchical_nodes(f'{key2}_poly{n}', mu_intercept=mu_prior[1][n], sigma_intercept=std_prior[1][n], cauchy_sigma_intercept=cauchy_sigma, transform='identity')

        else:

            n_evidence_sd_polypars = []

            for n in range(self.polynomial_order):
                e = self.build_hierarchical_nodes(f'evidence_sd_poly{n}', mu_intercept=mu_prior[n], sigma_intercept=std_prior[n], transform='identity')
                n_evidence_sd_polypars.append(e)

        if hasattr(self, 'fit_n2_prior_mu') and self.fit_n2_prior_mu:
            self.build_hierarchical_nodes('n2_prior_mu', mu_intercept=pt.mean(model['n2']), sigma_intercept=15., cauchy_sigma_intercept=.5, transform='identity')

    def _get_evidence_sd_labels(self):
        if self.memory_model == 'independent':
            key1 = 'n1_evidence_sd'
            key2 = 'n2_evidence_sd'
        elif self.memory_model == 'shared_perceptual_noise':
            key1 = 'perceptual_noise_sd'
            key2 = 'memory_noise_sd'

        return key1, key2

    def get_trialwise_variable(self, key, transform='identity'):
        
        if key in ['n1_evidence_mu', 'n2_evidence_mu', 'n1_evidence_sd', 'n2_evidence_sd', 'perceptual_noise_sd', 'memory_noise_sd', 'evidence_sd']:
            return self._get_trialwise_variable(key)
        else:
            return super().get_trialwise_variable(key, transform)


    def _get_trialwise_variable(self, key):

        model = pm.Model.get_context()

        if key == 'n1_evidence_mu':
            return model['n1']

        elif key == 'n2_evidence_mu':
            return model['n2']

        elif key == 'n1_evidence_sd':


            if self.memory_model == 'independent':
                
                if self.fit_seperate_evidence_sd:
                    dm = self.make_dm(x=self.data['n1'], variable='n1_evidence_sd')
                    n1_evidence_sd_poly_pars = pt.stack([self.get_trialwise_variable(f'n1_evidence_sd_poly{n}') for n in range(self.polynomial_order[0])], axis=1)
                    n1_evidence_sd = pt.softplus(pt.sum(n1_evidence_sd_poly_pars * dm, 1))
                else:
                    dm = self.make_dm(x=self.data['n1'], variable='evidence_sd')
                    n1_evidence_sd_poly_pars = pt.stack([self.get_trialwise_variable(f'evidence_sd_poly{n}') for n in range(self.polynomial_order)], axis=1)
                    n1_evidence_sd = pt.softplus(pt.sum(n1_evidence_sd_poly_pars * dm, 1))

            elif self.memory_model == 'shared_perceptual_noise':

                dm = self.make_dm(x=self.data['n1'], variable='perceptual_noise_sd')
                perceptual_noise_poly_pars = pt.stack([self.get_trialwise_variable(f'perceptual_noise_sd_poly{n}') for n in range(self.polynomial_order[0])], axis=1)
                perceptual_noise_sd = pt.softplus(pt.sum(perceptual_noise_poly_pars * dm, 1))

                dm = self.make_dm(x=self.data['n1'], variable='memory_noise_sd')
                memory_noise_poly_pars = pt.stack([self.get_trialwise_variable(f'memory_noise_sd_poly{n}') for n in range(self.polynomial_order[1])], axis=1)
                memory_noise_sd = pt.softplus(pt.sum(memory_noise_poly_pars * dm, 1))

                n1_evidence_sd = perceptual_noise_sd + memory_noise_sd

            return n1_evidence_sd

        elif key == 'n2_evidence_sd':

            if self.memory_model == 'independent':
                if self.fit_seperate_evidence_sd:
                    dm = self.make_dm(x=self.data['n2'], variable='n2_evidence_sd')
                    n2_evidence_sd_poly_pars = pt.stack([self.get_trialwise_variable(f'n2_evidence_sd_poly{n}') for n in range(self.polynomial_order[1])], axis=1)
                    n2_evidence_sd = pt.softplus(pt.sum(n2_evidence_sd_poly_pars * dm, 1))
                else:
                    dm = self.make_dm(x=self.data['n2'], variable='evidence_sd')
                    n2_evidence_sd_poly_pars = pt.stack([self.get_trialwise_variable(f'evidence_sd_poly{n}') for n in range(self.polynomial_order)], axis=1)
                    n2_evidence_sd = pt.softplus(pt.sum(n2_evidence_sd_poly_pars * dm, 1))

            elif self.memory_model == 'shared_perceptual_noise':

                dm = self.make_dm(x=self.data['n2'], variable='perceptual_noise_sd')
                perceptual_noise_poly_pars = pt.stack([self.get_trialwise_variable(f'perceptual_noise_sd_poly{n}') for n in range(self.polynomial_order[0])], axis=1)
                perceptual_noise_sd = pt.softplus(pt.sum(perceptual_noise_poly_pars * dm, 1))

                n2_evidence_sd = perceptual_noise_sd

            return n2_evidence_sd

        else:
            raise ValueError()

    def make_dm(self, x, variable='n1_evidence_sd'):



        if self.bspline:
            min_n, max_n = self.data[['n1', 'n2']].min().min(), self.data[['n1', 'n2']].max().max()

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

        else:
            model = pm.Model.get_context()
            exponents = np.arange(self.polynomial_order)
            dm = model[variable][:, np.newaxis]**exponents[np.newaxis, :]

        return dm

    @staticmethod
    def get_sd_curve(model, idata, x=None, variable='both', group=True):

        if x is None:
            x_min, x_max = model.data[['n1', 'n2']].min().min(), model.data[['n1', 'n2']].max().max()
            x = np.linspace(x_min, x_max, 100)


        if group:
            key = 'sd_poly{}_mu'
        else:
            key = 'sd_poly{}'

        if variable in ['n1', 'memory_noise_sd', 'perceptual_noise_sd', 'both']:
            print('yo1')
            if model.memory_model == 'independent':
                dm = np.asarray(model.make_dm(x=x, variable='n1_evidence_sd'))
                n1_sd = idata.posterior[[f'n1_evidence_{key.format(ix)}' for ix in range(0, model.polynomial_order[0])]].to_dataframe()
                n1_sd = softplus_np(n1_sd.dot(dm.T))
            elif model.memory_model == 'shared_perceptual_noise':
                print('yo2')
                perceptual_noise_sd = idata.posterior[[f'perceptual_noise_{key.format(ix)}' for ix in range(0, model.polynomial_order[0])]].to_dataframe()
                memory_noise_sd = idata.posterior[[f'memory_noise_{key.format(ix)}' for ix in range(0, model.polynomial_order[1])]].to_dataframe()

                dm = np.asarray(model.make_dm(x=x, variable='perceptual_noise_sd'))
                perceptual_noise_sd = softplus_np(perceptual_noise_sd.dot(dm.T))

                dm = np.asarray(model.make_dm(x=x, variable='memory_noise_sd'))
                memory_noise_sd = softplus_np(memory_noise_sd.dot(dm.T))
                n1_sd = memory_noise_sd + perceptual_noise_sd

                memory_noise_sd.columns = x
                memory_noise_sd.columns.name = 'x'
                perceptual_noise_sd.columns = x
                perceptual_noise_sd.columns.name = 'x'


            n1_sd.columns = x
            n1_sd.columns.name = 'x'

        if (variable == 'n2') or ((variable == 'both') & (model.memory_model == 'independent')):
            if model.memory_model == 'independent':
                dm = np.asarray(model.make_dm(x=x, variable='n2_evidence_sd'))
                n2_sd = idata.posterior[[f'n2_evidence_{key.format(ix)}' for ix in range(0, model.polynomial_order[1])]].to_dataframe()
                n2_sd = softplus_np(n2_sd.dot(dm.T))
            elif model.memory_model == 'shared_perceptual_noise':
                dm = np.asarray(model.make_dm(x=x, variable='perceptual_noise_sd'))
                perceptual_noise_sd = idata.posterior[[f'perceptual_noise_{key.format(ix)}' for ix in range(0, model.polynomial_order[0])]].to_dataframe()
                perceptual_noise_sd = softplus_np(perceptual_noise_sd.dot(dm.T))
                n2_sd = perceptual_noise_sd.dot(dm.T)
                perceptual_noise_sd.columns = x
                perceptual_noise_sd.columns.name = 'x'

            n2_sd.columns = x
            n2_sd.columns.name = 'x'

        if variable == 'n1':
            output = n1_sd
        elif variable == 'n2':
            output = n2_sd.stack().to_frame('sd')
        elif variable == 'perceptual_noise_sd':
            output = perceptual_noise_sd.stack().to_frame('sd')
        elif variable == 'memory_noise_sd':
            output = memory_noise_sd.stack().to_frame('sd')
        else:
            if model.memory_model == 'independent':
                output = pd.concat((n1_sd, n2_sd), axis=0, keys=['n1', 'n2'], names=['variable'])
            elif model.memory_model == 'shared_perceptual_noise':
                output = pd.concat((perceptual_noise_sd, memory_noise_sd), axis=0, keys=['perceptual_noise_sd', 'memory_noise_sd'], names=['variable'])

        return output.stack().to_frame('sd')


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

class FlexibleSDRiskModel(FlexibleSDComparisonModel, RiskModel):

    def __init__(self, data, prior_estimate='objective', fit_seperate_evidence_sd=True, save_trialwise_n_estimates=False, polynomial_order=5, bspline=False,
                 memory_model='independent'):

        if prior_estimate not in ['shared', 'full']:
            raise NotImplementedError('For now only with shared/full prior estimate')

        if (type(polynomial_order) is int) and fit_seperate_evidence_sd:
            polynomial_order = polynomial_order, polynomial_order

        self.polynomial_order = polynomial_order
        self.max_polynomial_order = np.max(self.polynomial_order)

        self.bspline = bspline

        RiskModel.__init__(self, data, save_trialwise_n_estimates=save_trialwise_n_estimates, fit_seperate_evidence_sd=fit_seperate_evidence_sd, prior_estimate=prior_estimate,
        memory_model=memory_model)


    def build_priors(self):
        
        # Risky/safe prior
        if self.prior_estimate == 'full':
            risky_n = np.where(self.data['risky_first'], self.data['n1'], self.data['n2'])
            safe_n = np.where(self.data['risky_first'], self.data['n2'], self.data['n1'])

            risky_prior_mu = np.mean(risky_n)
            risky_prior_std = np.std(risky_n)

            self.build_hierarchical_nodes('risky_prior_mu', mu_intercept=risky_prior_mu, sigma_intercept=15, cauchy_sigma_intercept=.5, cauchy_sigma_regressors=.5, transform='identity')
            self.build_hierarchical_nodes('risky_prior_std', mu_intercept=risky_prior_std, sigma_intercept=15, cauchy_sigma_intercept=.5, cauchy_sigma_regressors=.5, transform='softplus')

            safe_prior_mu = np.mean(safe_n)

            self.build_hierarchical_nodes('safe_prior_mu', mu_intercept=safe_prior_mu, sigma_intercept=15, cauchy_sigma_intercept=.5, cauchy_sigma_regressors=.5, transform='identity')

            safe_prior_std = np.std(safe_n)
            self.build_hierarchical_nodes('safe_prior_std', mu_intercept=safe_prior_std, sigma_intercept=15, cauchy_sigma_intercept=.5, cauchy_sigma_regressors=.5, transform='softplus')

            FlexibleSDComparisonModel.build_priors(self)
        else:
            prior_mu = (np.mean(self.data['n1']) + np.mean(self.data['n2']))/2.
            prior_std = (np.std(self.data['n1']) + np.std(self.data['n2']))/2.

            self.build_hierarchical_nodes('prior_mu', mu_intercept=prior_mu, sigma_intercept=15, cauchy_sigma_intercept=0.5, cauchy_sigma_regressors=0.5, transform='identity')
            self.build_hierarchical_nodes('prior_std', mu_intercept=prior_std, sigma_intercept=15, cauchy_sigma_intercept=0.5, cauchy_sigma_regressors=0.5, transform='softplus')

            FlexibleSDComparisonModel.build_priors(self)

    def _get_paradigm(self, data=None):
        return RiskModel._get_paradigm(self, data)

    def get_model_inputs(self):
        model = pm.Model.get_context()

        model_inputs = {}
        
        model_inputs['n1_evidence_mu'] = self.get_trialwise_variable('n1_evidence_mu', transform='identity')
        model_inputs['n2_evidence_mu'] = self.get_trialwise_variable('n2_evidence_mu', transform='identity')


        if self.prior_estimate == 'full':
            risky_first = model['risky_first'].astype(bool)

            risky_prior_mu = self.get_trialwise_variable('risky_prior_mu', transform='identity')
            risky_prior_std = self.get_trialwise_variable('risky_prior_std', transform='softplus')

            safe_prior_mu = self.get_trialwise_variable('safe_prior_mu', transform='identity')
            
            if self.prior_estimate == 'full_normed':
                safe_prior_std = 1.
            else:
                safe_prior_std = self.get_trialwise_variable('safe_prior_std', transform='softplus')

            model_inputs['n1_prior_mu'] = pt.where(risky_first, risky_prior_mu, safe_prior_mu)
            model_inputs['n1_prior_std'] = pt.where(risky_first, risky_prior_std, safe_prior_std)

            model_inputs['n2_prior_mu'] = pt.where(risky_first, safe_prior_mu, risky_prior_mu)
            model_inputs['n2_prior_std'] = pt.where(risky_first, safe_prior_std, risky_prior_std)

        elif self.prior_estimate == 'shared':
            model_inputs['n1_prior_mu'] = self.get_trialwise_variable('prior_mu', transform='identity')
            model_inputs['n1_prior_std'] = self.get_trialwise_variable('prior_std', transform='softplus')

            model_inputs['n2_prior_mu'] = self.get_trialwise_variable('prior_mu', transform='identity')
            model_inputs['n2_prior_std'] = self.get_trialwise_variable('prior_std', transform='softplus')


        model_inputs['n1_evidence_sd'] = self.get_trialwise_variable('n1_evidence_sd', transform='softplus')
        model_inputs['n2_evidence_sd'] = self.get_trialwise_variable('n2_evidence_sd', transform='softplus')

        model_inputs['p1'], model_inputs['p2'] = model['p1'], model['p2']

        return model_inputs

    def _get_choice_predictions(self, model_inputs):
        post_n1_mu, post_n1_sd = get_posterior(model_inputs['n1_prior_mu'], 
                                               model_inputs['n1_prior_std'], 
                                               model_inputs['n1_evidence_mu'], 
                                               model_inputs['n1_evidence_sd']
                                               )

        post_n2_mu, post_n2_sd = get_posterior(model_inputs['n2_prior_mu'],
                                               model_inputs['n2_prior_std'],
                                               model_inputs['n2_evidence_mu'], 
                                               model_inputs['n2_evidence_sd'])

        diff_mu, diff_sd = get_diff_dist(post_n2_mu * model_inputs['p2'], post_n2_sd * model_inputs['p2'],
                                         post_n1_mu * model_inputs['p1'], post_n1_sd * model_inputs['p1'])

        if self.save_trialwise_n_estimates:
            pm.Deterministic('n1_hat', post_n1_mu)
            pm.Deterministic('n2_hat', post_n2_mu)

        return cumulative_normal(0.0, diff_mu, diff_sd)


class ExpectedUtilityRiskModel(BaseModel):

    def __init__(self, data, save_trialwise_eu=False, probability_distortion=False, n_outcomes=1):
        self.save_trialwise_eu = save_trialwise_eu
        self.probability_distortion = probability_distortion
        self.n_outcomes = n_outcomes

        super().__init__(data)

    def _get_paradigm(self, data=None):

        if self.n_outcomes == 1:
            paradigm = super()._get_paradigm(data)

            paradigm['p1'] = data['p1'].values
            paradigm['p2'] = data['p2'].values
            paradigm['risky_first'] = data['risky_first'].values.astype(bool)

        else:
            paradigm = {}
            paradigm['subject_ix'], _ = pd.factorize(data.index.get_level_values('subject'))

            if 'choice' in data.columns:
                paradigm['choice'] = data.choice.values
            else:
                print('*** Warning: did not find choice column, assuming all choices are False ***')
                paradigm['choice'] = np.zeros_like(paradigm['n1']).astype(bool)

            for ix in range(1, self.n_outcomes+1):
                paradigm[f'n1.{ix}'] = data[f'n1.{ix}'].values
                paradigm[f'n2.{ix}'] = data[f'n2.{ix}'].values
                paradigm[f'p1.{ix}'] = data[f'p1.{ix}'].values
                paradigm[f'p2.{ix}'] = data[f'p2.{ix}'].values
                paradigm[f'log(n1.{ix})'] = np.log(data[f'n1.{ix}'].values)
                paradigm[f'log(n2.{ix})'] = np.log(data[f'n2.{ix}'].values)

        return paradigm

    def build_priors(self):
        self.build_hierarchical_nodes('alpha', mu_intercept=1., sigma_intercept=0.1, transform='softplus')
        self.build_hierarchical_nodes('sigma', mu_intercept=10., sigma_intercept=10,  transform='softplus')

        if self.probability_distortion:
            self.build_hierarchical_nodes('phi', mu_intercept=inverse_softplus(0.61), sigma_intercept=1.,  transform='softplus')

    def _get_choice_predictions(self, model_inputs):

        def prob_distortion(p, phi):
            return (p**phi) / ((p**phi + (1-p)**phi)**(1/phi))

        if self.n_outcomes == 1:
            if self.probability_distortion:
                p1 = prob_distortion(model_inputs['p1'], model_inputs['phi'])
                p2 = prob_distortion(model_inputs['p2'], model_inputs['phi'])

            else:
                p1 =  model_inputs['p1']
                p2 =  model_inputs['p2']

            eu1 = p1 * model_inputs['n1']**model_inputs['alpha']
            eu2 = p2 * model_inputs['n2']**model_inputs['alpha']

        else:
            eu1, eu2 = 0.0, 0.0

            for ix in range(1, self.n_outcomes+1):
                if self.probability_distortion:
                    eu1 += model_inputs[f'n1.{ix}']**model_inputs['alpha'] * prob_distortion(model_inputs[f'p1.{ix}'])
                    eu2 += model_inputs[f'n2.{ix}']**model_inputs['alpha'] * prob_distortion(model_inputs[f'p2.{ix}'])
                else:
                    eu1 += model_inputs[f'n1.{ix}']**model_inputs['alpha'] * model_inputs[f'p1.{ix}']
                    eu2 += model_inputs[f'n2.{ix}']**model_inputs['alpha'] * model_inputs[f'p2.{ix}']

        if self.save_trialwise_eu:
            pm.Deterministic('eu1', eu1)
            pm.Deterministic('eu2', eu2)

        return cumulative_normal(eu2 - eu1, 0.0, model_inputs['sigma'])

    def get_model_inputs(self):

        model = pm.Model.get_context()

        model_inputs = {}

        if self.n_outcomes == 1:

            model_inputs['n1'] = model['n1']
            model_inputs['n2'] = model['n2']
            model_inputs['p1'] = model['p1']
            model_inputs['p2'] = model['p2']
        else:
            for ix in range(1, self.n_outcomes+1):
                model_inputs[f'n1.{ix}'] = model[f'n1.{ix}']
                model_inputs[f'n2.{ix}'] = model[f'n2.{ix}']
                model_inputs[f'p1.{ix}'] = model[f'p1.{ix}']
                model_inputs[f'p2.{ix}'] = model[f'p2.{ix}']

        model_inputs['alpha'] = self.get_trialwise_variable('alpha', transform='softplus')
        model_inputs['sigma'] = self.get_trialwise_variable('sigma', transform='softplus')

        if self.probability_distortion:
            model_inputs['phi'] = self.get_trialwise_variable('phi', transform='softplus')

        return model_inputs

class FlexibleSDRiskRegressionModel(RegressionModel, FlexibleSDRiskModel):
    def __init__(self,  data, regressors, prior_estimate='full', fit_seperate_evidence_sd=True, 
                    save_trialwise_n_estimates=False, polynomial_order=5, bspline=False, memory_model='independent'):
        RegressionModel.__init__(self, data, regressors)
        FlexibleSDRiskModel.__init__(self, data, prior_estimate, fit_seperate_evidence_sd, 
                            save_trialwise_n_estimates=save_trialwise_n_estimates, polynomial_order=polynomial_order,
                             bspline=bspline, memory_model=memory_model)

    def get_trialwise_variable(self, key, transform='identity'):
        
        if key in ['n1_evidence_mu', 'n2_evidence_mu', 'n1_evidence_sd', 'n2_evidence_sd', 'perceptual_noise_sd', 'memory_noise_sd']:
            return self._get_trialwise_variable(key)
        else:
            return super().get_trialwise_variable(key, transform)

class ExpectedUtilityRiskRegressionModel(RegressionModel, ExpectedUtilityRiskModel):
    def __init__(self,  data, save_trialwise_eu, probability_distortion, regressors):
        RegressionModel.__init__(self, data, regressors=regressors)
        ExpectedUtilityRiskModel.__init__(self, data, save_trialwise_eu, probability_distortion=probability_distortion)