import warnings
import pandas as pd
import pymc as pm
import numpy as np
from .utils import cumulative_normal, get_diff_dist, get_posterior, logistic
import pytensor.tensor as pt
from patsy import dmatrix

class BaseModel(object):

    def __init__(self, data, save_trialwise_n_estimates=False):
        """
        data should contain ['n1', 'n2'] and 'choice'.
        The latter should be a boolean that indicates whether the *second*
        option was chosen.
        
        """
        self.data = data
        self.save_trialwise_n_estimates = save_trialwise_n_estimates
        self.free_parameters = self.get_free_parameters()

    def _get_paradigm(self, paradigm=None):

        if paradigm is None:
            paradigm = self.data

        paradigm_ = {}
        paradigm_['n1'] = paradigm['n1'].values
        paradigm_['n2'] = paradigm['n2'].values
        paradigm_['log(n1)'] = np.log(paradigm['n1'].values)
        paradigm_['log(n2)'] = np.log(paradigm['n2'].values)
        
        if 'subject' in paradigm.index.names:
            paradigm_['subject_ix'], _ = pd.factorize(paradigm.index.get_level_values('subject'))
        elif 'subject' in paradigm.columns:
            paradigm_['subject_ix'], _ = pd.factorize(paradigm.subject.values)

        if 'choice' in paradigm.columns:
            paradigm_['choice'] = paradigm.choice.values
        else:
            paradigm_['choice'] = np.zeros_like(paradigm['n1']).astype(bool)

        return paradigm_

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

        diff_mu, diff_sd = get_diff_dist(post_n2_mu, post_n2_sd, post_n1_mu, post_n1_sd)

        if self.save_trialwise_n_estimates:
            pm.Deterministic('n1_hat', post_n1_mu)
            pm.Deterministic('n2_hat', post_n2_mu)

        return cumulative_normal(model_inputs['threshold'], diff_mu, diff_sd)

    def build_likelihood(self, parameters, save_p_choice=False):
        model = pm.Model.get_context()
        model_inputs = self.get_model_inputs(parameters)
        
        if save_p_choice:
            p = pm.Deterministic('p', var=self._get_choice_predictions(model_inputs))
        else:
            p = self._get_choice_predictions(model_inputs)

        pm.Bernoulli('ll_bernoulli', p=p, observed=model['choice'])

    def get_parameter_values(self, n_trials=None):

        parameters = {}

        for key, info in self.free_parameters.items():
            parameters[key] = self.get_trialwise_variable(key, transform=info['transform'], n_trials=n_trials)

        return parameters

    def build_estimation_model(self, data=None, coords=None, hierarchical=True):

        if data is None:
            data = self.data

        if hierarchical and (coords is None):
            assert('subject' in data.index.names), "Hierarchical estimation requires a multi-index with a 'subject' level."
            coords = {'subject': data.index.unique(level='subject')}
                                              
        with pm.Model(coords=coords) as self.estimation_model:
            self.set_paradigm(data)
            self.build_priors(hierarchical=hierarchical)
            parameters = self.get_parameter_values(n_trials=len(data))
            self.build_likelihood(parameters, save_p_choice=False)

    def build_priors(self, hierarchical=True):

        if hierarchical:
            for key, info in self.free_parameters.items():
                self.build_hierarchical_nodes(key, **info)
        else:
            for key, info in self.free_parameters.items():
                self.build_prior(key, **info)

    def build_prior(self, name, mu_intercept=None, sigma_intercept=None, transform='identity'):

        model = pm.Model.get_context()

        if mu_intercept is None:
            mu_intercept = 0.0

        if sigma_intercept is None:
            sigma_intercept = .5

        if transform == 'identity':
            pm.Normal(name, mu=mu_intercept, sigma=sigma_intercept)
        elif transform == 'softplus':
            pm.Normal(name+'_untransformed', mu=mu_intercept, sigma=sigma_intercept)
            pm.Deterministic(name, var=pt.softplus(model[name+'_untransformed']))
        elif transform == 'logistic':
            pm.Normal(name+'_untransformed', mu=mu_intercept, sigma=sigma_intercept)
            pm.Deterministic(name, var=logistic(model[name+'_untransformed']))
        else:
            raise NotImplementedError


    def set_paradigm(self, paradigm=None):
        if paradigm is None:
            paradigm = self.data

        paradigm = self._get_paradigm(paradigm=paradigm)

        for key, value in paradigm.items():
            pm.Data(key, value, mutable=True)

    def build_prediction_model(self, paradigm, parameters):

        assert(isinstance(parameters, dict))

        if paradigm is None:
            paradigm = self.data
                                              
        with pm.Model() as self.prediction_model:
            self.set_paradigm(paradigm)

            # Make parameters flexible
            for key, value in parameters.items():
                pm.Data(key, value, mutable=True)

            self.build_likelihood(parameters=parameters, save_p_choice=True)

    def predict(self, paradigm, parameters):

        self.build_prediction_model(paradigm, parameters)
        data = self.prediction_model['p'].eval()
        data = pd.DataFrame(data, index=paradigm.index, columns=['p_choice'])
        data = data.join(paradigm)

        return data

    def simulate(self, paradigm, parameters, n_samples=1):

        self.build_prediction_model(paradigm, parameters)

        with self.prediction_model:
            data = pm.draw(self.prediction_model['ll_bernoulli'], draws=n_samples)

        if n_samples == 1:
            data = data[np.newaxis, :]

        if not paradigm.index.name:
            paradigm.index.name = 'trial'

        data = pd.DataFrame(data.T, index=paradigm.index, columns=pd.Index(np.arange(n_samples)+1, name='sample'))
        data = data.stack().to_frame('simulated_choice').astype(bool)
        data = data.join(paradigm)

        return data

    def sample(self, draws=1000, tune=1000, target_accept=0.8, **kwargs):
        
        with self.estimation_model:
            self.idata = pm.sample(draws, tune=tune, target_accept=target_accept, return_inferencedata=True, **kwargs)
        
        return self.idata            

    def ppc(self, data=None, idata=None, var_names=['p', 'll_bernoulli']):

        if data is None:
            if self.data is None:
                data = self.create_data()
            else:
                data = self.data

        if idata is None:
            idata = self.idata

        paradigm = self._get_paradigm(data)

        with self.estimation_model:
            self.set_paradigm(paradigm)
            idata = pm.sample_posterior_predictive(idata, var_names=var_names)

        pred = [idata['posterior_predictive'][key].to_dataframe() for key in var_names]
        pred = pd.concat(pred, axis=1, keys=var_names, names=['variable'])
        pred = pred.unstack(['chain', 'draw']).droplevel(1, axis=1)
        pred.index = data.index
        pred = pred.set_index(pd.MultiIndex.from_frame(data), append=True)
        pred = pred.stack('variable')
        pred = pred.reorder_levels(np.roll(pred.index.names, 1)).sort_index()

        return pred

    def get_trialwise_variable(self, key, transform, n_trials=None):

        model = pm.Model.get_context()

        if key == 'n1_evidence_mu':
            return model['log(n1)']

        if key == 'n2_evidence_mu':
            return model['log(n2)']

        if transform not in ['identy', 'softplus', 'logistic']:
            Exception()

        if transform == 'identity':
            transform_ = lambda x: x
        elif transform == 'softplus':
            transform_ = pt.softplus
        elif transform == 'logistic':
            transform_ = logistic

        if model[f'{key}'].ndim == 1:
            return transform_(model[f'{key}'][model['subject_ix']])
        elif model[f'{key}'].ndim == 0:
            return pt.tile(transform_(model[f'{key}']), n_trials)

    def build_hierarchical_nodes(self, name, mu_intercept=None, sigma_intercept=None, cauchy_sigma_intercept=None, transform='identity', **kwargs):

        if mu_intercept is None:
            mu_intercept = 0.0

        if sigma_intercept is None:
            sigma_intercept = .5
        
        if cauchy_sigma_intercept is None:
            cauchy_sigma_intercept = 0.25

        if transform == 'identity':
            group_mu = pm.Normal(f"{name}_mu", 
                                            mu=mu_intercept, 
                                            sigma=sigma_intercept)
        elif transform == 'softplus':
            group_mu = pm.Normal(f"{name}_mu_untransformed", 
                                            mu=mu_intercept, 
                                            sigma=sigma_intercept)

            pm.Deterministic(name=f'{name}_mu', var=pt.softplus(group_mu))
        elif transform == 'logistic':
            group_mu = pm.Normal(f"{name}_mu_untransformed", 
                                            mu=mu_intercept, 
                                            sigma=sigma_intercept)

            pm.Deterministic(name=f'{name}_mu', var=logistic(group_mu))
        else:
            raise NotImplementedError

            
        group_sd = pm.HalfCauchy(f'{name}_sd', cauchy_sigma_intercept)
        subject_offset = pm.Normal(f'{name}_offset', mu=0, sigma=1, dims=('subject',))

        if transform == 'identity':
             return pm.Deterministic(f'{name}', group_mu + group_sd * subject_offset, dims=('subject',))
        else:
            subjectwise_untrans = pm.Deterministic(f'{name}_untransformed', group_mu + group_sd * subject_offset, dims=('subject',))
        
            if transform == 'softplus':
                return pm.Deterministic(name=name, var=pt.softplus(subjectwise_untrans), dims=('subject',))
            elif transform == 'logistic':
                return pm.Deterministic(name=name, var=logistic(subjectwise_untrans), dims=('subject',))
            else:
                raise Exception

    def get_subjectwise_parameter_estimates(self, idata=None, parameters=None):

        idata = self._get_idata(idata)
        parameters = self._get_parameters(parameters).keys()

        pars = idata.posterior[parameters].to_dataframe()
        pars.columns.name = 'parameter'

        return pars

    def get_groupwise_parameter_estimates(self, idata=None, parameters=None, include_sd=True):

        idata = self._get_idata(idata)
        parameters = self._get_parameters(parameters)

        mu_pars = pd.concat([idata.posterior[par+'_mu'].to_dataframe() for par in self.free_parameters], axis=1, keys=parameters, names=['parameter']).droplevel(1, axis=1)

        if include_sd:
            sd_pars = pd.concat([idata.posterior[par+'_sd'].to_dataframe() for par in self.free_parameters], axis=1, keys=self.free_parameters, names=['parameter']).droplevel(1, axis=1)
            pars = pd.concat((mu_pars, sd_pars), keys=['mu', 'sd'], names=['type'], axis=1)

        else:
            pars = pd.concat((mu_pars,), keys=['mu'], names=['type'], axis=1)

        pars.columns.name = 'parameter'

        return pars

    def _get_idata(self, idata):
        if idata is None:
            idata = self.idata
            if idata is None:
                raise ValueError('No idata found. Please run sample() first.')

        return idata

    def _get_parameters(self, parameters):
        if parameters is None:
            parameters = self.free_parameters

        return parameters




class RegressionModel(BaseModel):

    def __init__(self, data, regressors=None):

        BaseModel.__init__(self, data)

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

        if transform not in ['identy', 'softplus', 'logistic']:
            Exception()


        if key in ['n1_evidence_mu', 'n2_evidence_mu']:
            if key in self.design_matrices.keys():
                dm = model[f'_dm_{key}']
                trialwise_pars = model[f'log({key[:2]})'] + pt.sum(model[key][model['subject_ix']] * dm, 1)
            else:
                if key == 'n1_evidence_mu':
                    return model['log(n1)']

                if key == 'n2_evidence_mu':
                    return model['log(n2)']


        else:
            dm = model[f'_dm_{key}']
            if transform == 'identity':
                trialwise_pars = pt.sum(model[key][model['subject_ix']] * dm, 1)
            elif transform == 'softplus':
                trialwise_pars = pt.softplus(pt.sum(model[key][model['subject_ix']] * dm, 1))
            elif transform == 'logistic':
                trialwise_pars = logistic(pt.sum(model[key][model['subject_ix']] * dm, 1))

        return trialwise_pars

    def build_estimation_model(self, data=None):

        if data is None:
            data = self.data

        coords = {'subject': self.unique_subjects }

        with pm.Model(coords=coords) as self.estimation_model:

            self.free_parameters = []
            self.build_priors()

            paradigm = self._get_paradigm(data=data)

            for key, value in paradigm.items():
                pm.Data(key, value, mutable=True)

            self.build_likelihood()


    def build_hierarchical_nodes(self, name, mu_intercept=0.0, sigma_intercept=1., cauchy_sigma_intercept=0.25, sigma_regressors=1., cauchy_sigma_regressors=0.25, transform='identity'):

        self.design_matrices[name] = self.build_design_matrix(self.data, name)

        model = pm.Model.get_context()
        model.add_coord(f'{name}_regressors', self.design_matrices[name].design_info.column_names)
        
        mu = np.zeros(self.design_matrices[name].shape[1])
        sigma = np.ones(self.design_matrices[name].shape[1]) * sigma_regressors
        cauchy_sigma = np.ones(self.design_matrices[name].shape[1]) * cauchy_sigma_regressors
        cauchy_sigma[0] = cauchy_sigma_intercept

        if self.design_matrices[name].design_info.column_names[0] == 'Intercept':

            if name in ['n1_evidence_mu', 'n2_evidence_mu']:
                warnings.warn(f'{name} has an Intercept-regressors. This is most likely NOT a good idea.')

            if transform == 'identity':
                mu[0] = mu_intercept
                sigma[0] = sigma_intercept
            elif transform == 'logistic':
                mu[0] = mu_intercept
                sigma[0] = sigma_intercept
            # Possibly use inverse of softplus

        group_mu = pm.Normal(f"{name}_mu", 
                                        mu=mu, 
                                        sigma=sigma,
                                        dims=(f'{name}_regressors',))

        
        group_sd = pm.HalfCauchy(f'{name}_sd', cauchy_sigma, dims=(f'{name}_regressors',))
        subject_offset = pm.Normal(f'{name}_offset', mu=0, sigma=1, dims=('subject', f'{name}_regressors'))

        return pm.Deterministic(name, group_mu + group_sd * subject_offset, dims=('subject', f'{name}_regressors'))

class LapseModel(BaseModel):

    def build_priors(self):
        super().build_priors()
        self.build_hierarchical_nodes('p_lapse', mu_intercept=-4, transform='logistic')

    def build_likelihood(self):
        model = pm.Model.get_context()
        model_inputs = self.get_model_inputs()

        p_choice = self._get_choice_predictions(model_inputs)
        p = pm.Deterministic('p', var=(1-model_inputs['p_lapse']) * p_choice + (model_inputs['p_lapse'] * 0.5))

        pm.Bernoulli('ll_bernoulli', p=p, observed=model['choice'])

    def get_model_inputs(self):
        model_inputs = super().get_model_inputs()
        model_inputs['p_lapse'] = self.get_trialwise_variable('p_lapse', transform='logistic')
        return model_inputs