import pandas as pd
import pymc as pm
import numpy as np
from .utils import cumulative_normal, get_posterior, get_diff_dist
from .utils.math import inverse_softplus, softplus_np
import pytensor.tensor as pt
from patsy import dmatrix
from .core import BaseModel, RegressionModel, LapseModel
from .utils.plotting import plot_prediction
from arviz import hdi
import seaborn as sns
import matplotlib.pyplot as plt

from .models import RegressionModel
from .core import BaseModel

class Risk_NLC_EU_Model(BaseModel):

    def __init__(self, data, save_trialwise_eu=False, probability_distortion=False,
                 prior_estimate='objective', fit_seperate_evidence_sd=True, incorporate_probability='after_inference',
                 save_trialwise_n_estimates=False, memory_model='independent'):
        
        prior_estimate='full' # did not work with normal  argument input.. got overwritten to True, no idea why
        probability_distortion=False

        print(f'prior:{prior_estimate}, prob distortion:{probability_distortion}')
        assert prior_estimate in ['objective', 'shared', 'different', 'full', 'full_normed']

        self.fit_seperate_evidence_sd = fit_seperate_evidence_sd
        self.memory_model = memory_model
        self.prior_estimate = prior_estimate
        self.incorporate_probability = incorporate_probability

        self.save_trialwise_eu = save_trialwise_eu
        self.probability_distortion = probability_distortion

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
#____________
#        model_inputs['n1'] = model['n1']
#        model_inputs['n2'] = model['n2']
        model_inputs['p1'] = model['p1']
        model_inputs['p2'] = model['p2']
#________
        # EU model
        model_inputs['alpha'] = self.get_trialwise_variable('alpha', transform='softplus')
        #model_inputs['sigma'] = self.get_trialwise_variable('sigma', transform='softplus')

        if self.probability_distortion:
            model_inputs['phi'] = self.get_trialwise_variable('phi', transform='softplus')



        return model_inputs
    

    def build_priors(self):

       # NLC model 
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
 
       # EU model
        self.build_hierarchical_nodes('alpha', mu_intercept=1., sigma_intercept=0.1, transform='softplus')
        #self.build_hierarchical_nodes('sigma', mu_intercept=10., sigma_intercept=10,  transform='softplus')

        if self.probability_distortion:
            self.build_hierarchical_nodes('phi', mu_intercept=inverse_softplus(0.61), sigma_intercept=1.,  transform='softplus')


    def _get_choice_predictions(self, model_inputs):

        if self.probability_distortion:

            def prob_distortion(p, phi):
                return (p**phi) / ((p**phi + (1-p)**phi)**(1/phi))

            p1 = prob_distortion(model_inputs['p1'], model_inputs['phi'])
            p2 = prob_distortion(model_inputs['p2'], model_inputs['phi'])

        else:
            p1 =  model_inputs['p1']
            p2 =  model_inputs['p2']

        post_n1_mu, post_n1_sd = get_posterior(model_inputs['n1_prior_mu'], 
                                               model_inputs['n1_prior_std'], 
                                               model_inputs['n1_evidence_mu'], 
                                               model_inputs['n1_evidence_sd']
                                               )

        post_n2_mu, post_n2_sd = get_posterior(model_inputs['n2_prior_mu'],
                                               model_inputs['n2_prior_std'],
                                               model_inputs['n2_evidence_mu'], 
                                               model_inputs['n2_evidence_sd'])

        eu1 = p1 * post_n1_mu**model_inputs['alpha']
        eu2 = p2 * post_n2_mu**model_inputs['alpha']

        diff_mu, diff_sd = get_diff_dist(eu2, post_n2_sd, eu1, post_n1_sd)

        #cumulative_normal(eu2 - eu1, 0.0, model_inputs['sigma'])
        #cumulative_normal(model_inputs['threshold'], diff_mu, diff_sd)
        return cumulative_normal(diff_mu, 0.0, diff_sd)


class Risk_NLC_EU_RegressionModel(RegressionModel, Risk_NLC_EU_Model):                             
    def __init__(self,  data, regressors, 
                 probability_distortion, # EU model
                 prior_estimate='objective', fit_seperate_evidence_sd=True, incorporate_probability='after_inference', # NLC
                 save_trialwise_n_estimates=False, memory_model='independent'):

        RegressionModel.__init__(self, data, regressors)
        Risk_NLC_EU_Model.__init__(self, data, 
                                    probability_distortion, # EU model
                                    prior_estimate, fit_seperate_evidence_sd, incorporate_probability=incorporate_probability,
                                    save_trialwise_n_estimates=save_trialwise_n_estimates, memory_model=memory_model)
