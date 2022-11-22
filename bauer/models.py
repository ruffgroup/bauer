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

        self.fit_seperate_evidence_sd = fit_seperate_evidence_sd
        self.prior_estimate = prior_estimate

        super().__init__(data)

    def _get_paradigm(self, data=None):

        paradigm = super()._get_paradigm(data)

        paradigm['p1'] = data['p1'].values
        paradigm['p2'] = data['p2'].values
        paradigm['risky_first'] = (data['p1'] != 1.0).values.astype(bool)

        return paradigm

    def get_model_inputs(self):

        model = pm.Model.get_context()

        model_inputs = {}
        
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

        elif self.prior_estimate == 'different':

            risky_first = model['risky_first'].astype(bool)

            safe_prior_mu = at.mean(at.log(at.stack((model['n1'][~risky_first], model['n2'][risky_first]), 0)))
            safe_prior_std = at.std(at.log(at.stack((model['n1'][~risky_first], model['n2'][risky_first]), 0)))
            risky_prior_mu = self.get_trialwise_variable('risky_prior_mu', transform='identity')
            risky_prior_std = self.get_trialwise_variable('risky_prior_std', transform='softplus')

            model_inputs['n1_prior_mu'] = at.where(risky_first, risky_prior_mu, safe_prior_mu)
            model_inputs['n1_prior_std'] = at.where(risky_first, risky_prior_std, safe_prior_std)

            model_inputs['n2_prior_mu'] = at.where(risky_first, safe_prior_mu, risky_prior_mu)
            model_inputs['n2_prior_std'] = at.where(risky_first, safe_prior_std, risky_prior_std)

        # Prob of choosing 1 should decrease when risky option comes first
        model_inputs['threshold'] =  at.log(model['p1'] / model['p2'])

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
        ndim = model['n1_evidence_sd'].ndim

        if ndim == 1:
            pm.Deterministic('evidence_sd', at.stack((model['n1_evidence_sd'], model['n2_evidence_sd']), 1), dims=('subject', 'order'))

        if self.prior_estimate == 'shared':
            self.build_hierarchical_nodes('prior_mu', mu_intercept=.3, transform='identity')
            self.build_hierarchical_nodes('prior_std', mu_intercept=-1., transform='softplus')

        elif self.prior_estimate == 'different':
            self.build_hierarchical_nodes('risky_prior_mu', mu_intercept=.3, transform='identity')
            self.build_hierarchical_nodes('risky_prior_std', mu_intercept=-1., transform='softplus')

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