import pandas as pd
import pymc as pm
import numpy as np
from .utils import cumulative_normal, get_diff_dist, get_posterior
import aesara.tensor as at
from patsy import dmatrix

class BaseModel(object):

    def __init__(self, data):
        self.data = data
        self.unique_subjects = self.data.index.unique(level='subject')
        self.n_subjects = len(self.unique_subjects) 
        self.base_numbers = self.data['n1'].unique()

    def _get_paradigm(self, data=None):

        if data is None:
            data = self.data

        paradigm = {}
        paradigm['n1'] = data['n1'].values
        paradigm['n2'] = data['n2'].values
        paradigm['subject_ix'], _ = pd.factorize(data.index.get_level_values('subject'))

        for key, value in paradigm.items():
            paradigm[key] = pm.MutableData(key, value)

        if 'choice' in data.columns:
            paradigm['choice'] = pm.MutableData('choice', data.choice)
        else:
            paradigm['choice'] = None

        return paradigm

    def build_likelihood(self, model, paradigm):
        model_inputs = self.get_model_inputs(model, paradigm)
        post_n1_mu, post_n1_sd = get_posterior(model_inputs['n1_prior_mu'], 
                                               model_inputs['n1_prior_std'], 
                                               at.log(paradigm['n1']),
                                               model_inputs['n1_evidence_sd']
                                               )

        post_n2_mu, post_n2_sd = get_posterior(model_inputs['n2_prior_mu'],
                                               model_inputs['n2_prior_std'],
                                               at.log(paradigm['n2']),
                                               model_inputs['n2_evidence_sd'])

        diff_mu, diff_sd = get_diff_dist(post_n1_mu, post_n1_sd, post_n2_mu, post_n2_sd)
        p = pm.Deterministic('p', var=cumulative_normal(model_inputs['threshold'], diff_mu, diff_sd))
        pm.Bernoulli('ll_bernoulli', p=p, observed=paradigm['choice'])

    def build_estimation_model(self, data=None):

        if data is None:
            data = self.data

        coords = {'subject': self.unique_subjects,
        'order':['first', 'second']}
                                              
        with pm.Model(coords=coords) as self.estimation_model:
            paradigm = self._get_paradigm(data=data)
            self.build_priors(self.estimation_model, paradigm)
            self.build_likelihood(self.estimation_model, paradigm)

    def build_prediction_model(self, parameters):
        parameters = parameters.reset_index()
        coords = {'subject': self.unique_subjects}

        data = self.create_data()

        with pm.Model(coords=coords) as self.pred_model:

            paradigm = self._get_paradigm(data)

            for key in self.free_parameters.keys():
                pm.MutableData(key, parameters[key], dims='subject')

        self.build_likelihood(paradigm, self.pred_model)

    def predict(self, parameters):

        if ('subject' not in parameters.columns) & (parameters.index.name != 'subject'):
            parameters['subject'] = range(1, len(parameters)+1)

        if not hasattr(self, 'pred_model'):
            self.build_prediction_model(parameters)

        for key in self.free_parameters.keys():
            self.pred_model.set_data(key, parameters[key].values)

        data = self.pred_model['p'].eval()

        return data

    def sample(self, draws=1000, tune=1000, target_accept=0.8):
        
        with self.estimation_model:
            self.trace = pm.sample(draws, tune=tune, target_accept=target_accept, return_inferencedata=True)
        
        return self.trace            

    def ppc(self, data=None, trace=None, var_names=['p', 'll_bernoulli']):

        if data is None:
            data = self.create_data()

        if trace is None:
            trace = self.trace

        self.build_estimation_model(data=data)

        with self.estimation_model:
            idata = pm.sample_posterior_predictive(trace, var_names=var_names)

        pred = [idata['posterior_predictive'][key].to_dataframe() for key in var_names]
        pred = pd.concat(pred, axis=1, keys=var_names, names=['variable'])
        pred = pred.unstack(['chain', 'draw']).droplevel(1, axis=1)
        pred.index = data.index
        pred = pred.set_index(pd.MultiIndex.from_frame(data), append=True)
        pred = pred.stack('variable')
        pred = pred.reorder_levels(np.roll(pred.index.names, 1)).sort_index()

        return pred

    def create_data(self):

        data = pd.MultiIndex.from_product([self.unique_subjects,
                                           np.exp(np.linspace(-1.5, 1.5, 25)),
                                           self.base_numbers],
                                               names=['subject', 'frac', 'n1']).to_frame().reset_index(drop=True)

        data['n1'] = data['n1'].values
        data['n2'] = ((1./data['frac']) * data['n1']).round().values
        data['trial_nr'] = data.groupby('subject').cumcount() + 1

        return data.set_index(['subject', 'trial_nr'])


def build_hierarchical_nodes_bounded(name, mu=0.0, sigma=.5):
    group_mu = pm.Normal(f"{name}_mu_untransformed", 
                                    mu=mu, 
                                    sigma=sigma)

    pm.Deterministic(name=f'{name}_mu', var=at.softplus(group_mu))
    
    group_sd = pm.HalfCauchy(f'{name}_sd', .25)
    subject_offset = pm.Normal(f'{name}_offset', mu=0, sigma=1, dims=('subject',))

    subjectwise_untrans = pm.Deterministic(f'{name}_untransformed', group_mu + group_sd * subject_offset, dims=('subject',))
    
    pm.Deterministic(name=name, var=at.softplus(subjectwise_untrans), dims=('subject',))

class RegressionModel(object):

    def __init__(self, data, regression):
        self.data = data
        self.unique_subjects = self.data.index.unique(level='subject')
        self.n_subjects = len(self.unique_subjects) 
        self.base_numbers = self.data['n1'].unique()