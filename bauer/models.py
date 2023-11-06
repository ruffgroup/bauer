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


class MagnitudeComparisonModel(BaseModel):

    def __init__(self, data, fit_prior=False, fit_seperate_evidence_sd=True, save_trialwise_n_estimates=False):

        self.fit_prior = fit_prior
        self.fit_seperate_evidence_sd = fit_seperate_evidence_sd


        super().__init__(data, save_trialwise_n_estimates=save_trialwise_n_estimates)

    def get_model_inputs(self):

        model = pm.Model.get_context()

        model_inputs = {}

        if self.fit_prior:
            model_inputs['n1_prior_mu'] = self.get_trialwise_variable('prior_mu', transform='identity')
            model_inputs['n2_prior_mu'] = self.get_trialwise_variable('prior_mu', transform='identity')

            model_inputs['n1_prior_std'] = self.get_trialwise_variable('prior_sd', transform='softplus')
            model_inputs['n2_prior_std'] = self.get_trialwise_variable('prior_sd', transform='softplus')

        else:
            mean_prior = (pt.mean(pt.log(model['n1'])) + pt.mean(pt.log(model['n2']))) / 2.
            mean_std = (pt.std(pt.log(model['n1'])) + pt.std(pt.log(model['n2']))) / 2.

            model_inputs['n1_prior_mu'] = mean_prior
            model_inputs['n2_prior_mu'] = mean_prior

            model_inputs['n1_prior_std'] = mean_std
            model_inputs['n2_prior_std'] = mean_std

        model_inputs['threshold'] =  0.0

        model_inputs['n1_evidence_mu'] = self.get_trialwise_variable('n1_evidence_mu', transform='identity') #model['n1'])
        model_inputs['n2_evidence_mu'] = self.get_trialwise_variable('n2_evidence_mu', transform='identity')

        if self.fit_seperate_evidence_sd:
            model_inputs['n1_evidence_sd'] = self.get_trialwise_variable('n1_evidence_sd', transform='softplus')
            model_inputs['n2_evidence_sd'] = self.get_trialwise_variable('n2_evidence_sd', transform='softplus')
        else:
            model_inputs['n1_evidence_sd'] = self.get_trialwise_variable('evidence_sd', transform='softplus')
            model_inputs['n2_evidence_sd'] = self.get_trialwise_variable('evidence_sd', transform='softplus')

        return model_inputs


    def build_priors(self):

        if self.fit_seperate_evidence_sd:
            self.build_hierarchical_nodes('n1_evidence_sd', mu_intercept=-1., transform='softplus')
            self.build_hierarchical_nodes('n2_evidence_sd', mu_intercept=-1., transform='softplus')
        else:
            self.build_hierarchical_nodes('evidence_sd', mu_intercept=-1., transform='softplus')

        if self.fit_prior:
            objective_mu = np.mean(np.log(np.stack((self.data['n1'], self.data['n2']))))
            objective_sd = np.mean(np.log(np.stack((self.data['n1'], self.data['n2']))))

            self.build_hierarchical_nodes('prior_mu', mu_intercept=objective_mu, transform='identity')
            self.build_hierarchical_nodes('prior_sd', mu_intercept=objective_sd, transform='softplus')
        
class MagnitudeComparisonLapseModel(LapseModel, MagnitudeComparisonModel):
    ...

class MagnitudeComparisonRegressionModel(RegressionModel, MagnitudeComparisonModel):
    def build_priors(self):

        super().build_priors()

        for key in ['n1_evidence_mu', 'n2_evidence_mu']:
            if key in self.regressors:
                self.build_hierarchical_nodes(key, mu_intercept=0.0, transform='identity')


class RiskModel(BaseModel):

    def __init__(self, data, prior_estimate='objective', fit_seperate_evidence_sd=True, incorporate_probability='after_inference',
                 save_trialwise_n_estimates=False, memory_model='independent'):

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

    def create_data(self):

        data = pd.MultiIndex.from_product([self.unique_subjects,
                                           np.exp(np.linspace(-1.5, 1.5, 25)),
                                           self.base_numbers, 
                                           [False, True]],
                                               names=['subject', 'frac', 'n1', 'risky_first']).to_frame().reset_index(drop=True)

        data['n1'] = data['n1'].values.astype(int)
        data['n2'] = (data['frac'] * data['n1']).round().values.astype(int)
        data['trial_nr'] = data.groupby('subject').cumcount() + 1
        data['p1'] = data['risky_first'].map({True:0.55, False:1.0})
        data['p2'] = data['risky_first'].map({True:1.0, False:0.55})

        return data.set_index(['subject', 'trial_nr'])

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
                mu_prior = [np.zeros(po) for po in self.polynomial_order]
                std_prior = [np.ones(po) * 100 for po in self.polynomial_order]
            else:
                mu_prior = np.zeros(self.polynomial_order)
                std_prior = np.ones(self.polynomial_order) * 100

            cauchy_sigma= 1
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
            self.build_hierarchical_nodes('n2_prior_mu', mu_intercept=pt.mean(model['n2']), sigma_intercept=100., cauchy_sigma_intercept=1., transform='identity')

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

                dm = self.make_dm(x=self.data['n1'], variable='perceptual_noise_sd')
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

            self.build_hierarchical_nodes('risky_prior_mu', mu_intercept=risky_prior_mu, sigma_intercept=100, cauchy_sigma_intercept=1.0, cauchy_sigma_regressors=1.0, transform='identity')
            self.build_hierarchical_nodes('risky_prior_std', mu_intercept=risky_prior_std, sigma_intercept=100, cauchy_sigma_intercept=1.0, cauchy_sigma_regressors=1.0, transform='softplus')

            safe_prior_mu = np.mean(safe_n)

            self.build_hierarchical_nodes('safe_prior_mu', mu_intercept=safe_prior_mu, sigma_intercept=100, cauchy_sigma_intercept=1.0, cauchy_sigma_regressors=1.0, transform='identity')

            safe_prior_std = np.std(safe_n)
            self.build_hierarchical_nodes('safe_prior_std', mu_intercept=safe_prior_std, sigma_intercept=100, cauchy_sigma_intercept=1.0, cauchy_sigma_regressors=1.0, transform='softplus')

            FlexibleSDComparisonModel.build_priors(self)
        else:
            prior_mu = (np.mean(self.data['n1']) + np.mean(self.data['n2']))/2.
            prior_std = (np.std(self.data['n1']) + np.std(self.data['n2']))/2.

            self.build_hierarchical_nodes('prior_mu', mu_intercept=prior_mu, sigma_intercept=100, cauchy_sigma_intercept=1.0, cauchy_sigma_regressors=1.0, transform='identity')
            self.build_hierarchical_nodes('prior_std', mu_intercept=prior_std, sigma_intercept=100, cauchy_sigma_intercept=1.0, cauchy_sigma_regressors=1.0, transform='softplus')

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

    def create_data(self):
        return RiskModel.create_data(self)

class ExpectedUtilityRiskModel(BaseModel):

    def __init__(self, data, save_trialwise_eu=False, probability_distortion=False):
        self.save_trialwise_eu = save_trialwise_eu
        self.probability_distortion = probability_distortion

        super().__init__(data)

    def _get_paradigm(self, data=None):

        paradigm = super()._get_paradigm(data)

        paradigm['p1'] = data['p1'].values
        paradigm['p2'] = data['p2'].values
        paradigm['risky_first'] = data['risky_first'].values.astype(bool)

        return paradigm

    def build_priors(self):
        self.build_hierarchical_nodes('alpha', mu_intercept=1., sigma_intercept=0.1, transform='softplus')
        self.build_hierarchical_nodes('sigma', mu_intercept=10., sigma_intercept=10,  transform='softplus')

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

        eu1 = p1 * model_inputs['n1']**model_inputs['alpha']
        eu2 = p2 * model_inputs['n2']**model_inputs['alpha']

        if self.save_trialwise_eu:
            pm.Deterministic('eu1', eu1)
            pm.Deterministic('eu2', eu2)

        return cumulative_normal(eu2 - eu1, 0.0, model_inputs['sigma'])

    def get_model_inputs(self):

        model = pm.Model.get_context()

        model_inputs = {}

        model_inputs['n1'] = model['n1']
        model_inputs['n2'] = model['n2']
        model_inputs['p1'] = model['p1']
        model_inputs['p2'] = model['p2']

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