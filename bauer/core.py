import pandas as pd
import pymc as pm
import numpy as np
from .utils import cumulative_normal, get_diff_dist, get_posterior
import aesara.tensor as at
from patsy import dmatrix

class BaseModel(object):

    def __init__(self, data):
        """
        data should contain ['n1', 'n2'] and 'choice'.
        The latter should be a boolean that indicates whether the *second*
        option was chosen.
        
        """
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

        if 'choice' in data.columns:
            paradigm['choice'] = data.choice.values
        else:
            paradigm['choice'] = np.zeros_like(paradigm['n1']).astype(bool)

        return paradigm

    def set_paradigm(self, paradigm):
        pm.set_data(paradigm)

    def build_likelihood(self):

        model = pm.Model.get_context()
        model_inputs = self.get_model_inputs()
        print(model_inputs)
        post_n1_mu, post_n1_sd = get_posterior(model_inputs['n1_prior_mu'], 
                                               model_inputs['n1_prior_std'], 
                                               at.log(model['n1']),
                                               model_inputs['n1_evidence_sd']
                                               )

        post_n2_mu, post_n2_sd = get_posterior(model_inputs['n2_prior_mu'],
                                               model_inputs['n2_prior_std'],
                                               at.log(model['n2']),
                                               model_inputs['n2_evidence_sd'])

        diff_mu, diff_sd = get_diff_dist(post_n2_mu, post_n2_sd, post_n1_mu, post_n1_sd)
        p = pm.Deterministic('p', var=cumulative_normal(model_inputs['threshold'], diff_mu, diff_sd))
        pm.Bernoulli('ll_bernoulli', p=p, observed=model['choice'])

    def build_estimation_model(self, data=None):

        if data is None:
            data = self.data

        coords = {'subject': self.unique_subjects, 'order':['first', 'second']}
                                              
        with pm.Model(coords=coords) as self.estimation_model:
            paradigm = self._get_paradigm(data=data)

            for key, value in paradigm.items():
                pm.Data(key, value, mutable=True)

            self.free_parameters = []
            self.build_priors()
            self.build_likelihood()

    def build_prediction_model(self, parameters):
        parameters = parameters.reset_index()
        coords = {'subject': self.unique_subjects}

        data = self.create_data()

        with pm.Model(coords=coords) as self.pred_model:

            paradigm = self._get_paradigm(data)

            for key in self.free_parameters.keys():
                pm.MutableData(key, parameters[key], dims='subject')

        self.build_likelihood()

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

        paradigm = self._get_paradigm(data)

        with self.estimation_model:
            self.set_paradigm(paradigm)
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

        data['n1'] = data['n1'].values.astype(int)
        data['n2'] = (data['frac'] * data['n1']).round().values.astype(int)
        data['trial_nr'] = data.groupby('subject').cumcount() + 1

        return data.set_index(['subject', 'trial_nr'])

    def get_trialwise_variable(self, key, transform='identity'):

        model = pm.Model.get_context()

        if transform not in ['identy', 'softplus']:
            Exception()

        if transform == 'identity':
            trialwise_pars = model[f'{key}'][model['subject_ix']]

        elif transform == 'softplus':
            trialwise_pars = at.softplus(model[f'{key}_untransformed'][model['subject_ix']])

        return trialwise_pars

    def build_hierarchical_nodes(self, name, mu_intercept=0.0, sigma_intercept=.5, transform='identity'):

        self.free_parameters.append(name)

        if transform == 'identity':
            group_mu = pm.Normal(f"{name}_mu", 
                                            mu=mu_intercept, 
                                            sigma=sigma_intercept)
        else:
            group_mu = pm.Normal(f"{name}_mu_untransformed", 
                                            mu=mu_intercept, 
                                            sigma=sigma_intercept)

            pm.Deterministic(name=f'{name}_mu', var=at.softplus(group_mu))
            
        group_sd = pm.HalfCauchy(f'{name}_sd', .25)
        subject_offset = pm.Normal(f'{name}_offset', mu=0, sigma=1, dims=('subject',))

        if transform == 'identity':
             pm.Deterministic(f'{name}', group_mu + group_sd * subject_offset, dims=('subject',))
        else:
            subjectwise_untrans = pm.Deterministic(f'{name}_untransformed', group_mu + group_sd * subject_offset, dims=('subject',))
        
            if transform == 'softplus':
                pm.Deterministic(name=name, var=at.softplus(subjectwise_untrans), dims=('subject',))
            else:
                raise Exception

class RegressionModel(BaseModel):

    def __init__(self, data, regressors=None):

        super().__init__(data)

        if regressors is None:
            self.regressors = {}
        else:
            self.regressors = regressors

        self.design_matrices = {}
        
    
    def _get_paradigm(self, data=None):

        paradigm = super()._get_paradigm(data)

        if data is None:
            data = self.data

        for key, dm in self.design_matrices.items():
            paradigm[f'_dm_{key}'] = np.asarray(dm)

        return paradigm

    def build_design_matrix(self, data, parameter):
        if parameter not in self.regressors:
            self.regressors[parameter] = '1'
        
        return dmatrix(self.regressors[parameter], data)

    def get_trialwise_variable(self, key, transform='identity'):

        model = pm.Model.get_context()

        if transform not in ['identy', 'softplus']:
            Exception()

        dm = model[f'_dm_{key}']

        if transform == 'identity':
            trialwise_pars = at.sum(model[key][model['subject_ix']] * dm, 1)
        elif transform == 'softplus':
            trialwise_pars = at.softplus(at.sum(model[key][model['subject_ix']] * dm, 1))

        return trialwise_pars

    def build_estimation_model(self, data=None):

        if data is None:
            data = self.data

        coords = {'subject': self.unique_subjects,
        'order':['first', 'second']}

        with pm.Model(coords=coords) as self.estimation_model:

            self.free_parameters = []
            self.build_priors()

            for key in self.free_parameters:
                self.estimation_model.coords[key+'_regressors'] = self.design_matrices[key].design_info.column_names

            paradigm = self._get_paradigm(data=data)

            for key, value in paradigm.items():
                pm.Data(key, value, mutable=True)

            self.build_likelihood()


    def build_hierarchical_nodes(self, name, mu_intercept=0.0, sigma_intercept=1., transform='identity'):

        self.free_parameters.append(name)
        self.design_matrices[name] = self.build_design_matrix(self.data, name)
        
        mu = np.zeros(self.design_matrices[name].shape[1])
        sigma = np.ones(self.design_matrices[name].shape[1])

        if self.design_matrices[name].design_info.column_names[0] == 'Intercept':
            if transform == 'identity':
                mu[0] = mu_intercept
                sigma[0] = sigma_intercept
            # Possibly use inverse of softplus

        group_mu = pm.Normal(f"{name}_mu", 
                                        mu=mu, 
                                        sigma=sigma,
                                        dims=(f'{name}_regressors',))

        
        group_sd = pm.HalfCauchy(f'{name}_sd', .25, dims=(f'{name}_regressors',))
        subject_offset = pm.Normal(f'{name}_offset', mu=0, sigma=1, dims=('subject', f'{name}_regressors'))

        pm.Deterministic(name, group_mu + group_sd * subject_offset, dims=('subject', f'{name}_regressors'))

