import pandas as pd
import pymc as pm
import numpy as np
from .utils import cumulative_normal, get_diff_dist, get_posterior
import aesara.tensor as at
from patsy import dmatrix
from .core import BaseModel, RegressionModel


class MagnitudeComparisonModel(BaseModel):

    def __init__(self, data, fit_n2_prior_mu=True, fit_seperate_evidence_sd=True):
        self.fit_n2_prior_mu = fit_n2_prior_mu
        self.fit_seperate_evidence_sd = fit_seperate_evidence_sd


        super().__init__(data)

    def get_model_inputs(self):

        model = pm.Model.get_context()

        model_inputs = {}
        model_inputs['n1_prior_mu'] = at.mean(at.log(model['n1']))
        model_inputs['n1_prior_std'] = at.std(at.log(model['n1']))
        model_inputs['n2_prior_std'] = at.std(at.log(model['n2']))
        model_inputs['threshold'] =  0.0

        model_inputs['n1_evidence_mu'] = at.log(model['n1'])
        model_inputs['n2_evidence_mu'] = at.log(model['n2'])

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
        

class MagnitudeComparisonRegressionModel(RegressionModel, MagnitudeComparisonModel):

    ...

class RiskModel(BaseModel):

    def __init__(self, data, prior_estimate='objective', fit_seperate_evidence_sd=True):

        assert prior_estimate in ['objective', 'shared', 'different', 'full']

        self.fit_seperate_evidence_sd = fit_seperate_evidence_sd
        self.prior_estimate = prior_estimate

        if 'risky_first' not in data.columns:
            data['risky_first'] = data['p1'] != 1.0

        super().__init__(data)

    def _get_paradigm(self, data=None):

        paradigm = super()._get_paradigm(data)

        paradigm['p1'] = data['p1'].values
        paradigm['p2'] = data['p2'].values
        paradigm['risky_first'] = data['risky_first'].values.astype(bool)

        return paradigm

    def get_model_inputs(self):

        model = pm.Model.get_context()

        model_inputs = {}
        
        model_inputs['n1_evidence_mu'] = at.log(model['n1'])
        model_inputs['n2_evidence_mu'] = at.log(model['n2'])

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


        # Prob of choosing 2 should increase with p2
        model_inputs['threshold'] =  at.log(model['p2'] / model['p1'])

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

    def __init__(self,  data, regressors, prior_estimate='objective', fit_seperate_evidence_sd=True):
        RegressionModel.__init__(self, data, regressors)
        RiskModel.__init__(self, data, prior_estimate, fit_seperate_evidence_sd)
