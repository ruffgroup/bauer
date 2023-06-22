import pandas as pd
import pymc as pm
import numpy as np
from .utils import cumulative_normal, get_diff_dist, get_posterior
from .utils.math import inverse_softplus
import aesara.tensor as at
from patsy import dmatrix
from .core import BaseModel, RegressionModel, LapseModel


class MagnitudeComparisonModel(BaseModel):

    def __init__(self, data, fit_n2_prior_mu=True, fit_seperate_evidence_sd=True, save_trialwise_n_estimates=False):
        self.fit_n2_prior_mu = fit_n2_prior_mu
        self.fit_seperate_evidence_sd = fit_seperate_evidence_sd


        super().__init__(data, save_trialwise_n_estimates=save_trialwise_n_estimates)

    def get_model_inputs(self):

        model = pm.Model.get_context()

        model_inputs = {}
        model_inputs['n1_prior_mu'] = at.mean(at.log(model['n1']))
        model_inputs['n1_prior_std'] = at.std(at.log(model['n1']))
        model_inputs['n2_prior_std'] = at.std(at.log(model['n2']))
        model_inputs['threshold'] =  0.0

        model_inputs['n1_evidence_mu'] = self.get_trialwise_variable('n1_evidence_mu', transform='identity') #model['n1'])
        model_inputs['n2_evidence_mu'] = self.get_trialwise_variable('n2_evidence_mu', transform='identity')

        if self.fit_n2_prior_mu:
            model_inputs['n2_prior_mu'] = self.get_trialwise_variable('n2_prior_mu', transform='identity')
        else:
            model_inputs['n2_prior_mu'] = at.mean(at.log(model['n2']))

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

        if self.fit_n2_prior_mu:
            self.build_hierarchical_nodes('n2_prior_mu', mu_intercept=0.0, transform='identity')
        
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

        assert prior_estimate in ['objective', 'shared', 'different', 'full']

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
            model_inputs['threshold'] =  at.log(model['p2'] / model['p1'])
        elif self.incorporate_probability == 'before_inference':
            model_inputs['threshold'] =  0.0
            model_inputs['n1_evidence_mu'] += at.log(model['p1'])
            model_inputs['n2_evidence_mu'] += at.log(model['p2'])
        else:
            raise ValueError('incorporate_probability should be either "after_inference" (default) or "before_inference"')

        if self.prior_estimate == 'objective':
            model_inputs['n1_prior_mu'] = at.mean(at.log(at.stack((model['n1'], model['n2']), 0)))
            model_inputs['n1_prior_std'] = at.std(at.log(at.stack((model['n1'], model['n2']), 0)))
            model_inputs['n2_prior_mu'] = model_inputs['n1_prior_mu']
            model_inputs['n2_prior_std'] = model_inputs['n1_prior_std']

        elif self.prior_estimate == 'shared':
            model_inputs['n1_prior_mu'] = self.get_trialwise_variable('prior_mu', transform='identity')
            model_inputs['n1_prior_std'] = self.get_trialwise_variable('prior_std', transform='softplus')
            model_inputs['n2_prior_mu'] = model_inputs['n1_prior_mu']
            model_inputs['n2_prior_std'] = model_inputs['n1_prior_std']

        elif self.prior_estimate == 'two_mus':

            risky_first = model['risky_first'].astype(bool)
            safe_n = at.where(risky_first, model['n2'], model['n1'])
            safe_prior_std = at.std(at.log(safe_n))
            risky_prior_std = self.get_trialwise_variable('risky_prior_std', transform='softplus')

            model_inputs['n1_prior_mu'] = self.get_trialwise_variable('n1_prior_mu', transform='identity')
            model_inputs['n1_prior_std'] = at.where(risky_first, risky_prior_std, safe_prior_std)
            model_inputs['n2_prior_mu'] = self.get_trialwise_variable('n2_prior_mu', transform='identity')
            model_inputs['n2_prior_std'] = at.where(risky_first, safe_prior_std, risky_prior_std)

        elif self.prior_estimate == 'different':

            risky_first = model['risky_first'].astype(bool)

            safe_n = at.where(risky_first, model['n2'], model['n1'])
            safe_prior_mu = at.mean(at.log(safe_n))
            safe_prior_std = at.std(at.log(safe_n))

            risky_prior_mu = self.get_trialwise_variable('risky_prior_mu', transform='identity')
            risky_prior_std = self.get_trialwise_variable('risky_prior_std', transform='softplus')

            model_inputs['n1_prior_mu'] = at.where(risky_first, risky_prior_mu, safe_prior_mu)
            model_inputs['n1_prior_std'] = at.where(risky_first, risky_prior_std, safe_prior_std)

            model_inputs['n2_prior_mu'] = at.where(risky_first, safe_prior_mu, risky_prior_mu)
            model_inputs['n2_prior_std'] = at.where(risky_first, safe_prior_std, risky_prior_std)

        elif self.prior_estimate == 'full':

            risky_first = model['risky_first'].astype(bool)

            risky_prior_mu = self.get_trialwise_variable('risky_prior_mu', transform='identity')
            risky_prior_std = self.get_trialwise_variable('risky_prior_std', transform='softplus')

            safe_prior_mu = self.get_trialwise_variable('safe_prior_mu', transform='identity')
            safe_prior_std = self.get_trialwise_variable('safe_prior_std', transform='softplus')

            model_inputs['n1_prior_mu'] = at.where(risky_first, risky_prior_mu, safe_prior_mu)
            model_inputs['n1_prior_std'] = at.where(risky_first, risky_prior_std, safe_prior_std)

            model_inputs['n2_prior_mu'] = at.where(risky_first, safe_prior_mu, risky_prior_mu)
            model_inputs['n2_prior_std'] = at.where(risky_first, safe_prior_std, risky_prior_std)

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
                self.build_hierarchical_nodes('n1_evidence_sd', mu_intercept=-1., transform='softplus')
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

        elif self.prior_estimate == 'full':
            risky_n = np.where(self.data['risky_first'], self.data['n1'], self.data['n2'])
            safe_n = np.where(self.data['risky_first'], self.data['n2'], self.data['n1'])

            risky_prior_mu = np.mean(np.log(risky_n))
            risky_prior_std = np.std(np.log(risky_n))

            self.build_hierarchical_nodes('risky_prior_mu', mu_intercept=risky_prior_mu, transform='identity')
            self.build_hierarchical_nodes('risky_prior_std', mu_intercept=risky_prior_std, transform='softplus')

            safe_prior_mu = np.mean(np.log(safe_n))
            safe_prior_std = np.std(np.log(safe_n))

            self.build_hierarchical_nodes('safe_prior_mu', mu_intercept=safe_prior_mu, transform='identity')
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

        for key in ['n1_evidence_mu', 'n2_evidence_mu']:
            if key in self.regressors:
                self.build_hierarchical_nodes(key, mu_intercept=0.0, transform='identity')

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

        intercept = -at.log(rnp) * slope # More risk-seeking -> higher rnp, smaller intercept, more likely to choose option 2
        intercept = at.where(risky_first, intercept, -intercept)
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
                 polynomial_order=2):
        self.fit_n2_prior_mu = fit_n2_prior_mu
        self.fit_seperate_evidence_sd = fit_seperate_evidence_sd
        self.polynomial_order = polynomial_order

        super().__init__(data)


    def build_estimation_model(self, data=None, coords=None):
        coords = {'subject': self.unique_subjects, 'order':['first', 'second']}
        coords['poly_order'] = np.arange(self.polynomial_order)

        return super().build_estimation_model(data=data, coords=coords)

    def get_model_inputs(self):

        model = pm.Model.get_context()

        model_inputs = {}

        model_inputs['n1_prior_mu'] = at.mean(model['n1'])
        model_inputs['n1_prior_std'] = at.std(model['n1'])
        model_inputs['n2_prior_std'] = at.std(model['n2'])
        model_inputs['threshold'] =  0.0

        model_inputs['n1_evidence_mu'] = self.get_trialwise_variable('n1_evidence_mu')
        model_inputs['n2_evidence_mu'] = self.get_trialwise_variable('n2_evidence_mu')

        if self.fit_n2_prior_mu:
            model_inputs['n2_prior_mu'] = self.get_trialwise_variable('n2_prior_mu', transform='identity')
        else:
            model_inputs['n2_prior_mu'] = at.mean(model['n2'])

        model_inputs['n1_evidence_sd'] = self.get_trialwise_variable('n1_evidence_sd')
        model_inputs['n2_evidence_sd'] = self.get_trialwise_variable('n2_evidence_sd')

        return model_inputs


    def build_priors(self):

        if self.fit_seperate_evidence_sd:
            n1_evidence_sd_polypars = []
            n2_evidence_sd_polypars = []

            for n in range(self.polynomial_order):
                if n == 0:
                    mu_intercept, sigma_intercept = 10, 10
                else:
                    mu_intercept, sigma_intercept = 0, 10**(-n+1)

                e1 = self.build_hierarchical_nodes(f'n1_evidence_sd_poly{n}', mu_intercept=mu_intercept, sigma_intercept=sigma_intercept, transform='identity')
                e2 = self.build_hierarchical_nodes(f'n2_evidence_sd_poly{n}', mu_intercept=mu_intercept, sigma_intercept=sigma_intercept, transform='identity')

                n1_evidence_sd_polypars.append(e1)
                n2_evidence_sd_polypars.append(e2)

            pm.Deterministic('n1_evidence_sd_poly', var=at.stack(n1_evidence_sd_polypars, axis=1),
                             dims=('subject', 'poly_order'))
            pm.Deterministic('n2_evidence_sd_poly', var=at.stack(n2_evidence_sd_polypars, axis=1),
                             dims=('subject', 'poly_order'))

        else:

            n_evidence_sd_polypars = []

            for n in range(self.polynomial_order):
                if n == 0:
                    mu_intercept, sigma_intercept = 10, 10
                else:
                    mu_intercept, sigma_intercept = 0, 1

                e = self.build_hierarchical_nodes(f'n_evidence_sd_poly{n}', mu_intercept=mu_intercept, sigma_intercept=sigma_intercept, transform='identity')
                n_evidence_sd_polypars.append(e)

            n_evidence_sd_polypars = pm.Deterministic('evidence_sd_poly', var=at.stack(n_evidence_sd_polypars, axis=1),
                             dims=('subject', 'poly_order'))
            pm.Deterministic('n1_evidence_sd_poly', var=n_evidence_sd_polypars,
                             dims=('subject', 'poly_order'))
            pm.Deterministic('n2_evidence_sd_poly', var=n_evidence_sd_polypars,
                             dims=('subject', 'poly_order'))

        if self.fit_n2_prior_mu:
            self.build_hierarchical_nodes('n2_prior_mu', mu_intercept=0.0, transform='identity')

    def get_trialwise_variable(self, key, transform='identity'):
        
        if key in ['n1_evidence_mu', 'n2_evidence_mu', 'n1_evidence_sd', 'n2_evidence_sd']:
            return self._get_trialwise_variable(key)
        else:
            super().get_trialwise_variable(key, transform)


    def _get_trialwise_variable(self, key):

        model = pm.Model.get_context()

        exponents = np.arange(self.polynomial_order)

        if key == 'n1_evidence_mu':
            return model['n1']

        elif key == 'n2_evidence_mu':
            return model['n2']

        elif key == 'n1_evidence_sd':
            n1_evidence_sd = at.sum(model['n1_evidence_sd_poly'][model['subject_ix'], :] *\
                                          model['n1'][:, np.newaxis]**exponents[np.newaxis, :], 1)
            return at.softplus(n1_evidence_sd)


        elif key == 'n2_evidence_sd':
            n2_evidence_sd = at.sum(model['n2_evidence_sd_poly'][model['subject_ix'], :] *\
                                          model['n2'][:, np.newaxis]**exponents[np.newaxis, :], 1)
            return at.softplus(n2_evidence_sd)
        else:
            raise ValueError()



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

class ExpectedUtilityRiskRegressionModel(RegressionModel, ExpectedUtilityRiskModel):
    def __init__(self,  data, save_trialwise_eu, probability_distortion, regressors):
        RegressionModel.__init__(self, data, regressors=regressors)
        ExpectedUtilityRiskModel.__init__(self, data, save_trialwise_eu, probability_distortion=probability_distortion)