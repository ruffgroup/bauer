import re
import pandas as pd
import pymc as pm
import numpy as np
from .utils.bayes import cumulative_normal, get_posterior, get_diff_dist
from .utils.math import inverse_softplus, softplus_np, inverse_softplus_np, logit_derivative, gaussian_pdf
from pymc.math import logit, invlogit
import pytensor.tensor as pt
from pytensor import scan    
from patsy import dmatrix
from .core import BaseModel, LapseModel, RegressionModel
from .utils.plotting import plot_prediction
from arviz import hdi
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import warn

class PsychometricModel(BaseModel):

    paradigm_keys = ['x1', 'x2']
    base_parameters = ['nu', 'bias']

    def __init__(self, paradigm=None):
        super().__init__(paradigm)

    def get_free_parameters(self):
            
            free_parameters = {}

            free_parameters['nu'] = {'mu_intercept': inverse_softplus_np(1.), 'sigma_intercept': 10., 'transform': 'softplus'}
            free_parameters['bias'] = {'mu_intercept': 0, 'sigma_intercept': 10., 'transform': 'identity'}
    
            return free_parameters

    def _get_choice_predictions(self, model_inputs):

        mu1 = model_inputs['x1']
        mu2 = model_inputs['x2']
        sd1 = model_inputs['nu']
        sd2 = model_inputs['nu']

        diff_mu, diff_sd = get_diff_dist(mu2, sd2, mu1, sd1)

        return cumulative_normal(model_inputs['bias'], diff_mu, diff_sd)

class PsychometricLapseModel(LapseModel, PsychometricModel):
    ...

class PsychometricRegressionModel(RegressionModel, PsychometricModel):
    
        def __init__(self, paradigm, regressors, save_trialwise_estimates=False):
            RegressionModel.__init__(self, regressors)
            PsychometricModel.__init__(self, paradigm)

class PsychometricLapseRegressionModel(LapseModel, PsychometricRegressionModel):
    ...

class MagnitudeComparisonModel(BaseModel):

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
            objective_mu = np.mean(np.stack((self.paradigm['n1'], self.paradigm['n2'])))
            objective_sd = np.mean(np.stack((self.paradigm['n1'], self.paradigm['n2'])))

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

    def __init__(self, paradigm, regressors, fit_prior=False, fit_seperate_evidence_sd=True, memory_model = 'independent',save_trialwise_estimates=False):
        RegressionModel.__init__(self, regressors)
        MagnitudeComparisonModel.__init__(self, paradigm, fit_prior, fit_seperate_evidence_sd, memory_model, save_trialwise_estimates)

class MagnitudeComparisonLapseModel(LapseModel, MagnitudeComparisonModel):
    ...

class MagnitudeComparisonLapseRegressionModel(LapseModel, MagnitudeComparisonRegressionModel):
    ...

class RiskModelProbabilityDistortion(BaseModel):

    def __init__(self, paradigm=None, magnitude_prior_estimate='objective', save_trialwise_n_estimates=False, n_prospects=2,
                 p_grid_size=20, lapse_rate=0.01, distort_magnitudes=True, distort_probabilities=True,
                 fix_magnitude_prior_sd=False, fix_probabiliy_prior_sd=False,
                 estimate_magnitude_prior_mu=False):

        assert magnitude_prior_estimate in ['objective'], 'Only objective prior is currently supported' 

        self.magnitude_prior_estimate = magnitude_prior_estimate
        self.n_prospects = n_prospects

        self.p_grid = np.linspace(1e-6, 1-1e-6, p_grid_size)
        self.lapse_rate = lapse_rate

        self.distort_magnitudes = distort_magnitudes
        self.distort_probabilities = distort_probabilities
        self.estimate_magnitude_prior_mu = estimate_magnitude_prior_mu

        self.fix_magnitude_prior_sd = fix_magnitude_prior_sd
        self.fix_probabiliy_prior_sd = fix_probabiliy_prior_sd

        if paradigm is not None:
            for ix in range(self.n_prospects):
                assert(f'n{ix+1}' in paradigm.columns), f'paradigm should contain columns n1, n2, ... n{self.n_prospects}'
                assert(f'p{ix+1}' in paradigm.columns), f'paradigm should contain columns p1, p2, ... p{self.n_prospects}'

        super().__init__(paradigm, save_trialwise_n_estimates=save_trialwise_n_estimates)


    def _get_paradigm(self, paradigm=None):

        if paradigm is None:
            paradigm = self.paradigm

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

            if self.paradigm is not None:
                prior_mu = np.log(np.stack([self.paradigm[f'n{ix+1}'].values for ix in range(self.n_prospects)])).mean()
                prior_sd = np.log(np.stack([self.paradigm[f'n{ix+1}'].values for ix in range(self.n_prospects)])).std()
            else:
                prior_mu = np.log(10)
                prior_sd = np.log(30)

            if self.estimate_magnitude_prior_mu:
                free_parameters['magnitude_prior_mu'] = {'mu_intercept': prior_mu, 'transform': 'softplus'}

            if not self.fix_magnitude_prior_sd:
                free_parameters['magnitude_prior_sd'] = {'mu_intercept': prior_sd, 'transform': 'softplus'}

        if self.distort_probabilities:
            if not self.fix_probabiliy_prior_sd:
                free_parameters['probability_prior_sd'] = {'mu_intercept': -1., 'transform': 'softplus'}

            free_parameters['probability_evidence_sd'] = {'mu_intercept': -1., 'transform': 'softplus'}
            free_parameters['probability_prior_mu'] = {'mu_intercept': 0.0, 'transform': 'identity'}

        return free_parameters

class ProspectTheoryModel(BaseModel):

    paradigm_keys = ['gain', 'loss', 'prob_gain']
    base_parameters = ['alpha', 'beta', 'lambda']

    def __init__(self, paradigm, save_trialwise_n_estimates=False):
        super().__init__(paradigm, save_trialwise_n_estimates=save_trialwise_n_estimates)

    def get_free_parameters(self):

        free_parameters = {}
        free_parameters['lambda'] = {'mu_intercept': 1, 'sigma_intercept':.5, 'transform': 'identity'}
        free_parameters['alpha'] = {'mu_intercept': 0.5, 'sigma_intercept':.5, 'transform': 'identity'}
        free_parameters['beta'] = {'mu_intercept': 0.5, 'sigma_intercept':.5, 'transform': 'identity'}

        return free_parameters

    def _get_choice_predictions(self, model_inputs):

        p = model_inputs['prob_gain']
        gain = model_inputs['gain']
        loss = model_inputs['loss']
        
        utility = p * gain ** model_inputs['alpha'] - (1-p) * model_inputs['lambda'] * loss ** model_inputs['beta']
        
        p_choose = cumulative_normal(utility, 0.0, 1.0)

        return p_choose
    
    def get_model_inputs(self, parameters):
        model = pm.Model.get_context()

        model_inputs = {}

        for key in self.base_parameters:
            model_inputs[key] = parameters[key]
        
        for key in self.paradigm_keys:
            model_inputs[key] = model[key]

        return model_inputs

class LossAversionModel(BaseModel):

    base_parameters = ['prior_mu_gains', 'prior_mu_losses', 'evidence_sd_gains', 'evidence_sd_losses', 'prior_sd_gains', 'prior_sd_losses']

    def __init__(self, paradigm=None, save_trialwise_n_estimates=False, 
                 magnitude_grid=None,
                 ev_diff_grid=None,
                 lapse_rate=0.01, 
                 normalize_likelihoods=True,
                 paradigm_type='mixed_vs_mixed',# mixed_vs_mixed or mixed_vs_0
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

        if paradigm_type == 'mixed_vs_mixed':
            self.paradigm_keys = ['p1', 'p2', 'gain1', 'gain2', 'loss1', 'loss2']
        elif paradigm_type == 'mixed_vs_0':
            self.paradigm_keys = ['gain', 'loss']
        else:
            raise ValueError('paradigm_type should be either "mixed_vs_mixed" or "mixed_vs_0"')

        self.paradigm_type = paradigm_type

        super().__init__(paradigm, save_trialwise_n_estimates=save_trialwise_n_estimates)


    def get_free_parameters(self):

        free_parameters = {}

        free_parameters['prior_mu_gains'] = {'mu_intercept': np.log(10.), 'sigma_intercept':np.log(10)/2., 'transform': 'identity'}
        free_parameters['prior_mu_losses'] = {'mu_intercept': np.log(10.), 'sigma_intercept':np.log(10)/2., 'transform': 'identity'}

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

        if self.paradigm_type == 'mixed_vs_mixed':
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

        elif self.paradigm_type == 'mixed_vs_0':

            p = 0.5

            gains = model_inputs['gain']
            losses = model_inputs['loss']

            expectations_gains_mu_log, expectations_gains_sd_log = get_posterior(model_inputs['prior_mu_gains'], model_inputs['prior_sd_gains'], pt.log(gains), model_inputs['evidence_sd_gains'])
            expectations_losses_mu_log, expectations_losses_sd_log = get_posterior(model_inputs['prior_mu_losses'], model_inputs['prior_sd_losses'], pt.log(losses), model_inputs['evidence_sd_losses'])

            diff_mu, diff_sd = get_diff_dist(p * expectations_gains_mu_log, p * model_inputs['evidence_sd_gains'],
                                             (1-p) * expectations_losses_mu_log, (1-p) * model_inputs['evidence_sd_losses'])

            p_choose2 = cumulative_normal(0.0, diff_mu, diff_sd)

        elif self.paradigm_type == 'mixed_vs_0_approx':

            p = 0.5

            gains = model_inputs['gain']
            losses = model_inputs['loss']

            expectations_gains_mu_log, expectations_gains_sd_log = get_posterior(model_inputs['prior_mu_gains'], model_inputs['prior_sd_gains'], pt.log(gains), model_inputs['evidence_sd_gains'])
            expectations_losses_mu_log, expectations_losses_sd_log = get_posterior(model_inputs['prior_mu_losses'], model_inputs['prior_sd_losses'], pt.log(losses), model_inputs['evidence_sd_losses'])


            expectations_gains_sd_log = pt.atleast_1d(expectations_gains_sd_log)
            expectations_losses_sd_log = pt.atleast_1d(expectations_losses_sd_log)

            # Calculate the distributions of gains in natural space (n trials x n grid)
            # NOTE: These PDFs are normalized, they SUM to 1, not integrate to 1
            gains_pdf = gaussian_pdf(n_grid_log[np.newaxis, :], expectations_gains_mu_log[:, np.newaxis], expectations_gains_sd_log[:, np.newaxis]) / n_grid * n_grid_dx
            losses_pdf = gaussian_pdf(n_grid_log[np.newaxis, :], expectations_losses_mu_log[:, np.newaxis], expectations_losses_sd_log[:, np.newaxis]) / n_grid * n_grid_dx

            if self.normalize_likelihoods:
                gains_pdf = gains_pdf / pt.sum(gains_pdf, 1, keepdims=True)
                losses_pdf = losses_pdf / pt.sum(losses_pdf, 1, keepdims=True)

            # joint gain/loss distribution n_trials x n_grid (gains) x n_grid (losses)
            joint_pdf = gains_pdf[:, :, np.newaxis] * losses_pdf[:, np.newaxis, :]

            # ev_grids: n_grid
            gains_ev_grid = p*n_grid
            losses_ev_grid = (1-p)*n_grid
            
            # evs: n_grid (gains) x n_grid (losses)
            evs = gains_ev_grid[:, np.newaxis] - losses_ev_grid[np.newaxis, :]

            # Choose is boolean, True mean accept the gamble ('second option')
            choose = evs > 0.0

            p_choose2 = pt.sum(joint_pdf * choose[np.newaxis, :, :], axis=[-2, -1])

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

class LossAversionRegressionModel(RegressionModel, LossAversionModel):

    def __init__(self, paradigm=None, save_trialwise_n_estimates=False, magnitude_grid=None, ev_diff_grid=None, lapse_rate=0.01, normalize_likelihoods=True, paradigm_type='mixed_vs_mixed', fix_prior_sds=True, regressors=None):
        LossAversionModel.__init__(self, paradigm=paradigm, save_trialwise_n_estimates=save_trialwise_n_estimates, magnitude_grid=magnitude_grid, ev_diff_grid=ev_diff_grid, lapse_rate=lapse_rate, normalize_likelihoods=normalize_likelihoods, paradigm_type=paradigm_type, fix_prior_sds=fix_prior_sds)
        RegressionModel.__init__(self, regressors=regressors)

class RiskModel(BaseModel):

    paradigm_keys = ['n1', 'n2', 'p1', 'p2']

    def __init__(self, paradigm=None, prior_estimate='objective', fit_seperate_evidence_sd=True, incorporate_probability='after_inference',
                 save_trialwise_n_estimates=False, memory_model='independent', n_prospects=2):

        assert prior_estimate in ['objective', 'shared', 'different', 'full', 'full_normed', 'klw', 'fix_prior_sd']

        self.fit_seperate_evidence_sd = fit_seperate_evidence_sd
        self.memory_model = memory_model
        self.prior_estimate = prior_estimate
        self.incorporate_probability = incorporate_probability

        super().__init__(paradigm, save_trialwise_n_estimates=save_trialwise_n_estimates)

    def get_model_inputs(self, parameters):

        model = pm.Model.get_context()

        model_inputs = {}
        
        model_inputs['n1_evidence_mu'] = pt.log(model['n1']) #self.get_trialwise_variable('n1_evidence_mu', transform='identity') #at.log(model['n1'])
        model_inputs['n2_evidence_mu'] = pt.log(model['n2'])

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
            model_inputs['n1_prior_sd'] = pt.std(pt.log(pt.stack((model['n1'], model['n2']), 0)))
            model_inputs['n2_prior_mu'] = model_inputs['n1_prior_mu']
            model_inputs['n2_prior_sd'] = model_inputs['n1_prior_sd']

        elif self.prior_estimate == 'shared':
            model_inputs['n1_prior_mu'] = parameters['prior_mu']
            model_inputs['n1_prior_sd'] = parameters['prior_sd']
            model_inputs['n2_prior_mu'] = model_inputs['n1_prior_mu']
            model_inputs['n2_prior_sd'] = model_inputs['n1_prior_sd']

        elif self.prior_estimate == 'fix_prior_sd':
            model_inputs['n1_prior_sd'] = pt.std(pt.log(pt.stack((model['n1'], model['n2']), 0))) # fixed same prior sd
            model_inputs['n2_prior_sd'] = model_inputs['n1_prior_sd']

            model_inputs['n1_prior_mu'] = parameters['risky_prior_mu']
            model_inputs['n2_prior_mu'] = parameters['safe_prior_mu']

        elif self.prior_estimate == 'two_mus':

            risky_first = pt.where(model['p1'] < model['p2'], True, False)

            safe_n = pt.where(risky_first, model['n2'], model['n1'])
            safe_prior_sd = pt.std(pt.log(safe_n))
            risky_prior_sd = parameters['risky_prior_sd']

            model_inputs['n1_prior_mu'] = parameters['n1_prior_mu']
            model_inputs['n1_prior_sd'] = pt.where(risky_first, risky_prior_sd, safe_prior_sd)
            model_inputs['n2_prior_mu'] = parameters['n2_prior_mu']
            model_inputs['n2_prior_sd'] = pt.where(risky_first, safe_prior_sd, risky_prior_sd)

        elif self.prior_estimate == 'different':

            risky_first = model['p1'] < model['p2']

            safe_n = pt.where(risky_first, model['n2'], model['n1'])
            safe_prior_mu = pt.mean(pt.log(safe_n))
            safe_prior_sd = pt.std(pt.log(safe_n))

            risky_prior_mu = parameters['risky_prior_mu']
            risky_prior_sd = parameters['risky_prior_sd']

            model_inputs['n1_prior_mu'] = pt.where(risky_first, risky_prior_mu, safe_prior_mu)
            model_inputs['n1_prior_sd'] = pt.where(risky_first, risky_prior_sd, safe_prior_sd)

            model_inputs['n2_prior_mu'] = pt.where(risky_first, safe_prior_mu, risky_prior_mu)
            model_inputs['n2_prior_sd'] = pt.where(risky_first, safe_prior_sd, risky_prior_sd)

        elif self.prior_estimate in ['full', 'full_normed']:

            risky_first = model['p1'] < model['p2']

            risky_prior_mu = parameters['risky_prior_mu']
            risky_prior_sd = parameters['risky_prior_sd']

            safe_prior_mu = parameters['safe_prior_mu']
            
            if self.prior_estimate == 'full_normed':
                safe_prior_sd = 1.
            else:
                safe_prior_sd = parameters['safe_prior_sd']

            model_inputs['n1_prior_mu'] = pt.where(risky_first, risky_prior_mu, safe_prior_mu)
            model_inputs['n1_prior_sd'] = pt.where(risky_first, risky_prior_sd, safe_prior_sd)

            model_inputs['n2_prior_mu'] = pt.where(risky_first, safe_prior_mu, risky_prior_mu)
            model_inputs['n2_prior_sd'] = pt.where(risky_first, safe_prior_sd, risky_prior_sd)

        elif self.prior_estimate == 'klw':
            model_inputs['n1_prior_mu'] = pt.mean(pt.log(pt.stack((model['n1'], model['n2']), 0)))
            model_inputs['n2_prior_mu'] = model_inputs['n1_prior_mu']

            model_inputs['n1_prior_sd'] = parameters['prior_sd']
            model_inputs['n2_prior_sd'] = model_inputs['n1_prior_sd']


        if self.fit_seperate_evidence_sd:

            if self.memory_model == 'independent':
                model_inputs['n1_evidence_sd'] = parameters['n1_evidence_sd']
                model_inputs['n2_evidence_sd'] = parameters['n2_evidence_sd']
            elif self.memory_model == 'shared_perceptual_noise':
                perceptual_sd = parameters['perceptual_noise_sd']
                memory_sd = parameters['memory_noise_sd']

                model_inputs['n1_evidence_sd'] = perceptual_sd + memory_sd
                model_inputs['n2_evidence_sd'] = perceptual_sd
            else:
                raise ValueError('Unknown memory model: {}'.format(self.memory_model))

        else:
            model_inputs['n1_evidence_sd'] = parameters['evidence_sd']
            model_inputs['n2_evidence_sd'] = parameters['evidence_sd']

        return model_inputs

    def get_free_parameters(self):

        free_parameters = {}

        if self.fit_seperate_evidence_sd:
            if self.memory_model == 'independent':
                free_parameters['n1_evidence_sd'] = {'mu_intercept':-1., 'transform':'softplus'}
                free_parameters['n2_evidence_sd'] = {'mu_intercept':-1., 'transform':'softplus'}
            elif self.memory_model == 'shared_perceptual_noise':
                free_parameters['perceptual_noise_sd'] = {'mu_intercept':-1., 'transform':'softplus'}
                free_parameters['memory_noise_sd'] = {'mu_intercept':-1., 'transform':'softplus'}
            else:
                raise ValueError('Unknown memory model: {}'.format(self.memory_model))
        else:
            free_parameters['evidence_sd'] = {'mu_intercept':-1., 'transform':'softplus'}

        if self.prior_estimate == 'shared':
            if self.paradigm is not None:
                prior_mu = np.mean(np.log(np.stack((self.paradigm['n1'], self.paradigm['n2']))))
                prior_sd = np.mean(np.log(np.stack((self.paradigm['n1'], self.paradigm['n2']))))
            else:
                prior_mu = np.log(25)
                prior_sd = 2

            free_parameters['prior_mu'] = {'mu_intercept':prior_mu, 'transform':'identity'}
            free_parameters['prior_sd'] = {'mu_intercept':prior_sd, 'transform':'identity'}

        elif self.prior_estimate == 'different':

            if self.paradigm is not None:
                risky_n = np.where(self.paradigm['p1'] != 1.0, self.paradigm['n1'], self.paradigm['n2'])
                risky_prior_mu = np.mean(np.log(risky_n))
                risky_prior_sd = np.std(np.log(risky_n))
            else:
                risky_prior_mu = np.log(25)
                risky_prior_sd = 2

            free_parameters['risky_prior_mu'] = {'mu_intercept':risky_prior_mu, 'transform':'identity'}
            free_parameters['risky_prior_sd'] = {'mu_intercept':risky_prior_sd, 'transform':'identity'}

        elif self.prior_estimate in ['full', 'full_normed']:
            if self.paradigm is not None:
                risky_n = np.where(self.paradigm['p1'] != 1.0, self.paradigm['n1'], self.paradigm['n2'])
                safe_n = np.where(self.paradigm['p2'] != 1.0, self.paradigm['n2'], self.paradigm['n1'])

                risky_prior_mu = np.mean(np.log(risky_n))
                risky_prior_sd = np.std(np.log(risky_n))

                safe_prior_mu = np.mean(np.log(safe_n))
                safe_prior_sd = np.std(np.log(safe_n))

            else:
                risky_prior_mu = np.log(25)
                safe_prior_mu = np.log(25)

                risky_prior_sd = 2
                safe_prior_sd = 2

            free_parameters['risky_prior_mu'] = {'mu_intercept':risky_prior_mu, 'transform':'identity'}
            free_parameters['risky_prior_sd'] = {'mu_intercept':risky_prior_sd, 'transform':'softplus'}


            free_parameters['safe_prior_mu'] = {'mu_intercept':safe_prior_mu, 'transform':'identity'}

            if self.prior_estimate == 'full':
                free_parameters['safe_prior_sd'] = {'mu_intercept':safe_prior_sd, 'transform':'softplus'}

        elif self.prior_estimate == 'fix_prior_sd': # only mus estimated but prior sd fixed
            risky_n = np.where(self.paradigm['p1'] != 1.0, self.paradigm['n1'], self.paradigm['n2'])
            safe_n = np.where(self.paradigm['p2'] != 1.0, self.paradigm['n2'], self.paradigm['n1'])
            risky_prior_mu = np.mean(np.log(risky_n))
            safe_prior_mu = np.mean(np.log(safe_n))
            free_parameters['risky_prior_mu'] = {'mu_intercept':risky_prior_mu, 'transform':'identity'}
            free_parameters['safe_prior_mu'] = {'mu_intercept':safe_prior_mu, 'transform':'identity'}



        elif self.prior_estimate == 'klw':
            if hasattr(self, 'paradigm') and (self.paradigm is not None):
                prior_sd = np.std(np.log(np.stack((self.paradigm['n1'], self.paradigm['n2']))))
            else:
                prior_sd = 1.5

            free_parameters['prior_sd'] = {'mu_intercept':inverse_softplus_np(prior_sd), 'transform':'softplus'}

        return free_parameters

    def _get_example_paradigm(self):
        n_safe = [5., 7., 10, 14, 20, 28]
        risky_p = 0.55

        fractions = np.exp(np.linspace(np.log(0.25), np.log(4), 8, True))
        risky_first = [True, False]

        paradigm = pd.MultiIndex.from_product([n_safe, fractions, risky_first], names=['n_safe', 'fraction', 'risky_first']).to_frame(index=False)

        paradigm['n1'] = np.round(paradigm['fraction'] * paradigm['n_safe']).where(paradigm['risky_first'], paradigm['n_safe'])
        paradigm['n2'] = np.round(paradigm['fraction'] * paradigm['n_safe']).where(~paradigm['risky_first'], paradigm['n_safe'])

        paradigm['p1'] = np.where(paradigm['risky_first'], risky_p, 1.)
        paradigm['p2'] = np.where(paradigm['risky_first'], 1., risky_p)

        paradigm['log(risky/safe)'] = np.log(paradigm['fraction'])

        paradigm.set_index(pd.Index(np.arange(1, len(paradigm)+1), name='trial'), inplace=True)

        return paradigm

class RiskRegressionModel(RegressionModel, RiskModel):

    def __init__(self,  paradigm, regressors, prior_estimate='objective', fit_seperate_evidence_sd=True, incorporate_probability='after_inference',
                 save_trialwise_n_estimates=False, memory_model='independent'):
        RegressionModel.__init__(self, regressors)
        RiskModel.__init__(self, paradigm, prior_estimate, fit_seperate_evidence_sd, incorporate_probability=incorporate_probability,
                           save_trialwise_n_estimates=save_trialwise_n_estimates, memory_model=memory_model)

    def get_trialwise_variable(self, key):

        # Prior mean
        if (key == 'risky_prior_mu') and ('prior_mu' in self.regressors):
            return super().get_trialwise_variable('risky_prior_mu') + super().get_trialwise_variable('prior_mu', transform='identity')

        if (key == 'safe_prior_mu') and ('prior_mu' in self.regressors):
            return super().get_trialwise_variable('safe_prior_mu') + super().get_trialwise_variable('prior_mu', transform='identity')

        # Evidence SD
        # if (key == 'n1_evidence_sd') and ('evidence_sd' in self.regressors):
        #     return pt.softplus(super().get_trialwise_variable('n1_evidence_sd') + super().get_trialwise_variable('evidence_sd', transform='identity'))

        # if (key == 'n2_evidence_sd') and ('evidence_sd' in self.regressors):
        #     return pt.softplus(super().get_trialwise_variable('n2_evidence_sd') + super().get_trialwise_variable('evidence_sd', transform='identity'))

        if (key == 'n1_evidence_sd') and ('evidence_sd_diff' in self.regressors):
            return pt.softplus(super().get_trialwise_variable('n1_evidence_sd') + super().get_trialwise_variable('evidence_sd_diff', transform='identity'))

        if (key == 'n2_evidence_sd') and ('evidence_sd_diff' in self.regressors):
            return pt.softplus(super().get_trialwise_variable('n2_evidence_sd') - super().get_trialwise_variable('evidence_sd_diff', transform='identity'))

        if (key == 'n1_evidence_mu') and ('evidence_mu_diff' in self.regressors):
            return super().get_trialwise_variable('n1_evidence_mu') + super().get_trialwise_variable('evidence_mu_diff', transform='identity')

        if (key == 'n2_evidence_mu') and ('evidence_mu_diff' in self.regressors):
            return super().get_trialwise_variable('n2_evidence_mu') - super().get_trialwise_variable('evidence_mu_diff', transform='identity')

        return super().get_trialwise_variable(key=key)


class RiskLapseModel(LapseModel, RiskModel):
    ...

class RiskLapseRegressionModel(LapseModel, RiskRegressionModel):
    ...

class RNPModel(BaseModel):

    def __init__(self, paradigm, risk_neutral_p=0.55):
        self.risk_neutral_p = risk_neutral_p

        super().__init__(paradigm)

    def get_model_inputs(self):

        model = pm.Model.get_context()

        model_inputs = {}
        model_inputs['n1_evidence_mu'] = self.get_trialwise_variable('n1_evidence_mu', transform='identity') #model['n1'])
        model_inputs['n2_evidence_mu'] = self.get_trialwise_variable('n2_evidence_mu', transform='identity')
        model_inputs['rnp'] = self.get_trialwise_variable('rnp', transform='logistic')
        model_inputs['gamma'] = self.get_trialwise_variable('gamma', transform='identity')

        return model_inputs

    def _get_paradigm(self, paradigm=None):

        paradigm = super()._get_paradigm(paradigm)

        paradigm['p1'] = paradigm['p1'].values
        paradigm['p2'] = paradigm['p2'].values
        paradigm['risky_first'] = paradigm['risky_first'].values.astype(bool)

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
    def __init__(self,  paradigm, regressors, risk_neutral_p=0.55):
        RegressionModel.__init__(self, paradigm, regressors=regressors)
        RNPModel.__init__(self, paradigm, risk_neutral_p)


class FlexibleNoiseComparisonModel(BaseModel):
    """Magnitude-comparison model with a flexible, magnitude-dependent noise curve.

    Unlike log-space models, this model represents evidence in **natural (linear)
    space**: the Gaussian noise on the perceived magnitude n has standard deviation
    ν_k(n).  This noise curve is parameterised as a B-spline of natural-space n:

        ν_k(n) = softplus( Σ_j β_{k,j} · φ_j(n) )

    where φ_j(n) are B-spline basis functions evaluated at natural-space n (not log n).
    ν_k(n) is therefore **natural-space noise** (SD of the Gaussian on n, not on log n).

    **Weber's law** corresponds to a *linear* ν(n) ∝ n: noise grows proportionally
    with magnitude.  A flat ν(n) implies constant absolute noise, which violates
    Weber's law.  Use ``get_sd_curve()`` to extract posterior ν(n) curves.
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
                objective_mu = np.mean(np.log(np.stack((self.paradigm['n1'], self.paradigm['n2']))))
                objective_sd = np.mean(np.log(np.stack((self.paradigm['n1'], self.paradigm['n2']))))

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
                spline_pars2 = pt.stack([parameters[l1] for l1 in labels1], axis=1)

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

class AffineNoiseComparisonModel(FlexibleNoiseComparisonModel):
    """Magnitude-comparison model with affine (intercept + linear) noise.

    The noise curve is parameterised as:

        ν_k(n) = softplus(β_{k,0} + β_{k,1} · n̂)

    where n̂ = (n − n_min) / (n_max − n_min) normalises n to [0, 1] over the
    observed magnitude range.  This is an **affine** function of n:

    - β_{k,1} ≈ 0  →  constant absolute noise (no Weber scaling)
    - β_{k,0} small, β_{k,1} large  →  approximately linear / Weber-like
    - Both nonzero  →  noise floor + Weber scaling

    This model sits between :class:`MagnitudeComparisonModel` (log-space, pure
    Weber, 2 free noise params per option) and :class:`FlexibleNoiseComparisonModel`
    (arbitrary B-spline, 2k params):

        MCM  ⊂  AffineNoise (4 noise params)  ⊂  FlexNoise (2k noise params)

    Parameters
    ----------
    paradigm : pd.DataFrame
    fit_seperate_evidence_sd : bool
        If True (default) fit separate noise curves for n1 and n2.
    """

    def __init__(self, paradigm, fit_seperate_evidence_sd=True,
                 fit_prior=False, memory_model='independent'):
        super().__init__(paradigm,
                         fit_seperate_evidence_sd=fit_seperate_evidence_sd,
                         fit_prior=fit_prior,
                         polynomial_order=2,
                         memory_model=memory_model)

    def make_dm(self, x, variable='n1_evidence_sd'):
        min_n = self.paradigm[['n1', 'n2']].min().min()
        max_n = self.paradigm[['n1', 'n2']].max().max()
        x_norm = (np.asarray(x, dtype=float) - min_n) / (max_n - min_n)
        return np.column_stack([np.ones_like(x_norm), x_norm])


class FlexibleNoiseComparisonRegressionModel(RegressionModel, FlexibleNoiseComparisonModel):

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

class FlexibleNoiseRiskModel(FlexibleNoiseComparisonModel, RiskModel):
    """Risk model with a flexible, magnitude-dependent noise curve.

    Unlike the standard :class:`RiskModel` (which models noise in log space),
    this model represents payoffs in **natural (linear) space**.  The Gaussian noise
    on the perceived magnitude has standard deviation ν_k(n), parameterised as a
    B-spline of natural-space n:

        ν_k(n) = softplus( Σ_j β_{k,j} · φ_j(n) )

    where φ_j are B-spline basis functions evaluated at the *natural-space*
    magnitude n (not log n).  ν_k(n) is therefore **natural-space noise**
    (SD of the Gaussian on n, not on log n).

    **Weber's law** (scale invariance) corresponds to a *linear* ν(n) ∝ n:
    noise grows proportionally with magnitude.  A flat ν(n) implies constant
    absolute noise, violating Weber's law.

    Use ``get_sd_curve()`` to extract posterior ν(n) curves (natural-space noise).

    Parameters
    ----------
    polynomial_order : int or (int, int)
        Number of B-spline basis functions per noise curve.  A single int is
        used for both options; a 2-tuple sets (n1_order, n2_order) separately.
    representational_noise : {'payoff', 'ev'}
        Whether the noise curve is indexed by raw payoff magnitude or by
        expected value.
    """

    def __init__(self, paradigm, prior_estimate='full',
                 fit_seperate_evidence_sd=True, save_trialwise_n_estimates=False, polynomial_order=5,
                 representational_noise='payoff',
                 memory_model='independent'):

        if prior_estimate not in ['shared', 'full', 'fix_prior_sd']:
            raise NotImplementedError('For now only with shared/full/fix_prior_sd prior estimate')

        if representational_noise not in ['payoff', 'ev']:
            raise ValueError(f'Unknown representational noise: {representational_noise} (should be "payoff" or "ev")')

        if (type(polynomial_order) is int) and fit_seperate_evidence_sd:
            polynomial_order = polynomial_order, polynomial_order

        self.polynomial_order = polynomial_order
        self.max_polynomial_order = np.max(self.polynomial_order)
        self.representational_noise = representational_noise

        RiskModel.__init__(self, paradigm, save_trialwise_n_estimates=save_trialwise_n_estimates,
                           fit_seperate_evidence_sd=fit_seperate_evidence_sd,
                           prior_estimate=prior_estimate,
                           memory_model=memory_model)

    def get_model_inputs(self, parameters):

        model = pm.Model.get_context()
        model_inputs = {}

        if self.prior_estimate == 'full':

            risky_first = model['p1'] < model['p2']

            risky_prior_mu = parameters['risky_prior_mu']
            risky_prior_sd = parameters['risky_prior_sd']
            safe_prior_mu = parameters['safe_prior_mu']
            
            if self.prior_estimate == 'full_normed':
                safe_prior_sd = 1.
            else:
                safe_prior_sd = parameters['safe_prior_sd']

            model_inputs['n1_prior_mu'] = pt.where(risky_first, risky_prior_mu, safe_prior_mu)
            model_inputs['n1_prior_sd'] = pt.where(risky_first, risky_prior_sd, safe_prior_sd)

            model_inputs['n2_prior_mu'] = pt.where(risky_first, safe_prior_mu, risky_prior_mu)
            model_inputs['n2_prior_sd'] = pt.where(risky_first, safe_prior_sd, risky_prior_sd)
        elif self.prior_estimate == 'fix_prior_sd':
            model_inputs['n1_prior_sd'] = pt.std(pt.log(pt.stack((model['n1'], model['n2']), 0))) # fixed same prior sd
            model_inputs['n2_prior_sd'] = model_inputs['n1_prior_sd']

            model_inputs['n1_prior_mu'] = parameters['risky_prior_mu']
            model_inputs['n2_prior_mu'] = parameters['safe_prior_mu']

        else:
            model_inputs['n1_prior_mu'] = parameters['prior_mu']
            model_inputs['n1_prior_sd'] = parameters['prior_sd']
            model_inputs['n2_prior_mu'] = model_inputs['n1_prior_mu']
            model_inputs['n2_prior_sd'] = model_inputs['n1_prior_sd']

        model_inputs['n1_evidence_mu'] = model['n1']
        model_inputs['n2_evidence_mu'] = model['n2']

        model_inputs['n1_evidence_sd'] = self._get_trialwise_evidence_sd('n1_evidence_sd', parameters)
        model_inputs['n2_evidence_sd'] = self._get_trialwise_evidence_sd('n2_evidence_sd', parameters)

        model_inputs['p1'] = model['p1']
        model_inputs['p2'] = model['p2']

        return model_inputs

    def _get_choice_predictions(self, model_inputs):

        if self.representational_noise == 'payoff':
            post_n1_mu, post_n1_sd = get_posterior(model_inputs['n1_prior_mu'], 
                                                model_inputs['n1_prior_sd'], 
                                                model_inputs['n1_evidence_mu'], 
                                                model_inputs['n1_evidence_sd']
                                                )

            post_n2_mu, post_n2_sd = get_posterior(model_inputs['n2_prior_mu'],
                                                model_inputs['n2_prior_sd'],
                                                model_inputs['n2_evidence_mu'], 
                                                model_inputs['n2_evidence_sd'])

                                               
            diff_mu, diff_sd = get_diff_dist(post_n2_mu * model_inputs['p2'], model_inputs['n2_evidence_sd'],
                                             post_n1_mu * model_inputs['p1'], model_inputs['n1_evidence_sd'])

        elif self.representational_noise == 'ev':

            post_n1_mu, post_n1_sd = get_posterior(model_inputs['n1_prior_mu'], 
                                                model_inputs['n1_prior_sd'], 
                                                model_inputs['n1_evidence_mu'], 
                                                model_inputs['n1_evidence_sd']
                                                )

            post_n2_mu, post_n2_sd = get_posterior(model_inputs['n2_prior_mu'],
                                                model_inputs['n2_prior_sd'],
                                                model_inputs['n2_evidence_mu'], 
                                                model_inputs['n2_evidence_sd'])

                                               
            diff_mu, diff_sd = get_diff_dist(post_n2_mu * model_inputs['p2'], model_inputs['n2_evidence_sd'] * model_inputs['p2'],
                                             post_n1_mu * model_inputs['p1'], model_inputs['n1_evidence_sd'] * model_inputs['p1'])


        if self.save_trialwise_n_estimates:
            pm.Deterministic('n1_hat', post_n1_mu)
            pm.Deterministic('n2_hat', post_n2_mu)

        return cumulative_normal(0.0, diff_mu, diff_sd)

    def get_free_parameters(self):

        self.fit_prior = False
        free_parameters = FlexibleNoiseComparisonModel.get_free_parameters(self)
        self.fit_prior = True

        if self.prior_estimate == 'full':
            if self.paradigm is not None:
                risky_n = np.where(self.paradigm['p1'] != 1.0, self.paradigm['n1'], self.paradigm['n2'])
                safe_n = np.where(self.paradigm['p2'] != 1.0, self.paradigm['n2'], self.paradigm['n1'])

                risky_prior_mu = np.mean(risky_n)
                risky_prior_sd = np.std(risky_n)

                safe_prior_mu = np.mean(safe_n)
                safe_prior_sd = np.std(safe_n)

            else:
                risky_prior_mu = 25
                safe_prior_mu = 25

                risky_prior_sd = 50
                safe_prior_sd = 50

            free_parameters['risky_prior_mu'] = {'mu_intercept':risky_prior_mu, 'sigma_intercept':25., 'transform':'identity'}
            free_parameters['risky_prior_sd'] = {'mu_intercept':risky_prior_sd, 'sigma_intercept':25., 'transform':'softplus'}
            free_parameters['safe_prior_mu'] = {'mu_intercept':safe_prior_mu, 'sigma_intercept':25., 'transform':'identity'}

            if self.prior_estimate == 'full':
                free_parameters['safe_prior_sd'] = {'mu_intercept':safe_prior_sd, 'transform':'softplus'}
        
        elif self.prior_estimate == 'fix_prior_sd': # only mus estimated but prior sd fixed
            risky_n = np.where(self.paradigm['p1'] != 1.0, self.paradigm['n1'], self.paradigm['n2'])
            safe_n = np.where(self.paradigm['p2'] != 1.0, self.paradigm['n2'], self.paradigm['n1'])
            risky_prior_mu = np.mean(np.log(risky_n))
            safe_prior_mu = np.mean(np.log(safe_n))
            free_parameters['risky_prior_mu'] = {'mu_intercept':risky_prior_mu, 'transform':'identity'}
            free_parameters['safe_prior_mu'] = {'mu_intercept':safe_prior_mu, 'transform':'identity'}

        elif self.prior_estimate == 'shared':

            if self.paradigm is not None:
                prior_mu = np.mean(np.stack((self.paradigm['n1'], self.paradigm['n2'])))
                prior_sd = np.mean(np.stack((self.paradigm['n1'], self.paradigm['n2'])))
            else:
                prior_mu = 25
                prior_sd = 25

            free_parameters['prior_mu'] = {'mu_intercept':prior_mu, 'sigma_intercept':25., 'transform':'identity'}
            free_parameters['prior_sd'] = {'mu_intercept':prior_sd, 'sigma_intercept':25., 'transform':'softplus'}

        return free_parameters

    def _get_paradigm(self, paradigm=None, subject_mapping=None):
        return RiskModel._get_paradigm(self, paradigm, subject_mapping=subject_mapping)

class AffineNoiseRiskModel(FlexibleNoiseRiskModel):
    """Risky-choice model with affine (intercept + linear) noise.

    The noise curve is parameterised as:

        ν_k(n) = softplus(β_{k,0} + β_{k,1} · n̂)

    where n̂ = (n − n_min) / (n_max − n_min).  See
    :class:`AffineNoiseComparisonModel` for the magnitude-comparison variant.
    """

    def __init__(self, paradigm, prior_estimate='full',
                 fit_seperate_evidence_sd=True, save_trialwise_n_estimates=False,
                 memory_model='independent'):
        super().__init__(paradigm, prior_estimate=prior_estimate,
                         fit_seperate_evidence_sd=fit_seperate_evidence_sd,
                         save_trialwise_n_estimates=save_trialwise_n_estimates,
                         polynomial_order=2,
                         memory_model=memory_model)

    def make_dm(self, x, variable='n1_evidence_sd'):
        min_n = self.paradigm[['n1', 'n2']].min().min()
        max_n = self.paradigm[['n1', 'n2']].max().max()
        x_norm = (np.asarray(x, dtype=float) - min_n) / (max_n - min_n)
        return np.column_stack([np.ones_like(x_norm), x_norm])


class FlexibleNoiseRiskRegressionModel(RegressionModel, FlexibleNoiseRiskModel):

    def __init__(self, paradigm,
                 regressors,
                 prior_estimate='full',
                 fit_seperate_evidence_sd=True, save_trialwise_n_estimates=False, polynomial_order=5, 
                 representational_noise='payoff',
                 memory_model='independent'):

        if (type(polynomial_order) is int) and fit_seperate_evidence_sd:
            polynomial_order = polynomial_order, polynomial_order 

        
        for key in list(regressors.keys()):

            if key in ['evidence_sd', 'n1_evidence_sd', 'memory_noise_sd', 'n2_evidence_sd', 'perceptual_noise_sd']:

                if key in ['evidence_sd']:
                    po = polynomial_order
                elif key in ['n1_evidence_sd', 'memory_noise_sd']:
                    po = polynomial_order[0]
                elif key in ['n2_evidence_sd', 'perceptual_noise_sd']:
                    po = polynomial_order[1]

                for i in range(1, po+1):
                    regressors[f'{key}_spline{i}'] = regressors[key]

                regressors.pop(key)
            

        RegressionModel.__init__(self, regressors)
        FlexibleNoiseRiskModel.__init__(self, paradigm, prior_estimate, fit_seperate_evidence_sd, save_trialwise_n_estimates,
                                        polynomial_order, representational_noise, memory_model)


    def get_sd_curve(self, conditions, idata=None, pars=None, x=None, variable='n1_evidence_sd', group=True, hierarchical=True, data=None):

        conditionwise_parameters = self.get_conditionwise_parameters(idata, conditions, group=group)
        conditionwise_parameters = conditionwise_parameters.stack(list(range(conditions.columns.nlevels))).unstack('parameter')


        if self.memory_model == 'independent':
            possible_variables = ['n1_evidence_sd', 'n2_evidence_sd', 'both']
        elif self.memory_model == 'shared_perceptual_noise':
            possible_variables = ['memory_noise_sd', 'perceptual_noise_sd', 'both']
        assert(variable in possible_variables), f'variable must be one of {possible_variables}'

        if variable == 'both':
            key1, key2 = self._get_evidence_sd_labels()

            n1_evidence_sd = self.get_sd_curve(conditions, idata=idata, pars=pars, x=x, variable=key1, group=group, hierarchical=hierarchical, data=data)
            n2_evidence_sd = self.get_sd_curve(conditions, idata=idata, pars=pars, x=x, variable=key2, group=group, hierarchical=hierarchical, data=data)

            return n1_evidence_sd.join(n2_evidence_sd)


        if pars is not None:
            raise NotImplementedError('pars argument is not implemented yet')

        if idata is None:
            raise ValueError('idata argument is mandatory')

        if x is None:
            assert((data is not None) or (hasattr(self, 'data'))), "If x is not provided, data must be provided."

            if data is None:
                data = self.data

            x_min, x_max = data[['n1', 'n2']].min().min(), data[['n1', 'n2']].max().max()
            x = np.linspace(x_min, x_max, 100)

        labels1, labels2 = self._get_evidence_sd_spline_par_labels()
        if variable in ['n1_evidence_sd', 'memory_noise_sd']:
            labels = labels1
        else:
            labels = labels2

        if group:
            labels1 = [f'{l}_mu' for l in labels1]
            labels2 = [f'{l}_mu' for l in labels2]

        dm = self.make_dm(x=x, variable=variable)

        pars = conditionwise_parameters[labels]
        output = softplus_np(pars.dot(dm.T))
        
        output.columns = x
        output.columns.name = 'x'

        output = output.stack().to_frame(variable)
        output.columns.name = 'variable'

        return output
    
class ExpectedUtilityRiskModel(BaseModel):

    paradigm_keys = ['n1', 'n2', 'p1', 'p2']

    def __init__(self, paradigm, save_trialwise_eu=False, probability_distortion=False, n_outcomes=1):
        self.save_trialwise_eu = save_trialwise_eu
        self.probability_distortion = probability_distortion
        self.n_outcomes = n_outcomes

        super().__init__(paradigm)

    def _get_paradigm(self, paradigm=None):

        if self.n_outcomes == 1:
            paradigm_ = super()._get_paradigm(paradigm)

            paradigm_['p1'] = paradigm['p1'].values
            paradigm_['p2'] = paradigm['p2'].values
            paradigm_['n1'] = paradigm['n1'].values
            paradigm_['n2'] = paradigm['n2'].values

        else:
            paradigm_ = {}
            paradigm_['subject_ix'], _ = pd.factorize(paradigm.index.get_level_values('subject'))

            if 'choice' in paradigm.columns:
                paradigm_['choice'] = paradigm.choice.values
            else:
                print('*** Warning: did not find choice column, assuming all choices are False ***')
                paradigm_['choice'] = np.zeros_like(paradigm['n1']).astype(bool)

            for ix in range(1, self.n_outcomes+1):
                paradigm_[f'n1.{ix}'] = paradigm[f'n1.{ix}'].values
                paradigm_[f'n2.{ix}'] = paradigm[f'n2.{ix}'].values
                paradigm_[f'p1.{ix}'] = paradigm[f'p1.{ix}'].values
                paradigm_[f'p2.{ix}'] = paradigm[f'p2.{ix}'].values
                paradigm_[f'log(n1.{ix})'] = np.log(paradigm[f'n1.{ix}'].values)
                paradigm_[f'log(n2.{ix})'] = np.log(paradigm[f'n2.{ix}'].values)

        return paradigm_

    def get_free_parameters(self):
        free_parameters = {}

        free_parameters['alpha'] = {'mu_intercept': 1., 'sigma_intercept': 0.1, 'transform': 'softplus'}
        free_parameters['sigma'] = {'mu_intercept': 10., 'sigma_intercept': 10., 'transform': 'softplus'}

        if self.probability_distortion:
            free_parameters['phi'] = {'mu_intercept': inverse_softplus(0.61), 'sigma_intercept': 1., 'transform': 'softplus'}

        return free_parameters

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

    def get_model_inputs(self, parameters):

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

        model_inputs['alpha'] = parameters['alpha']
        model_inputs['sigma'] = parameters['sigma']

        if self.probability_distortion:
            model_inputs['phi'] = parameters['phi']

        return model_inputs

class ExpectedUtilityRiskRegressionModel(RegressionModel, ExpectedUtilityRiskModel):
    def __init__(self,  paradigm, save_trialwise_eu, probability_distortion, regressors):
        RegressionModel.__init__(self, regressors=regressors)
        ExpectedUtilityRiskModel.__init__(self, paradigm, save_trialwise_eu, probability_distortion=probability_distortion)