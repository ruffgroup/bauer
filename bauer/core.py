import warnings
import pandas as pd
import pymc as pm
import numpy as np
from .utils import cumulative_normal, get_diff_dist, get_posterior
from .utils.math import logistic, softplus_np, logistic_np, logit_np, inverse_softplus_np
import pytensor.tensor as pt
from patsy import dmatrix

class BaseModel(object):

    paradigm_keys = []

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
        paradigm_['_data_n'] = len(paradigm)

        for key in self.paradigm_keys:
            paradigm_[key] = paradigm[key].values

        if 'subject' in paradigm.index.names:
            paradigm_['subject_ix'], _ = pd.factorize(paradigm.index.get_level_values('subject'))
        elif 'subject' in paradigm.columns:
            paradigm_['subject_ix'], _ = pd.factorize(paradigm['subject'])

        if 'choice' in paradigm.columns:
            paradigm_['choice'] = paradigm['choice'].values

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

    def get_parameter_values(self):

        parameters = {}

        for key in self.free_parameters.keys():
            parameters[key] = self.get_trialwise_variable(key)

        return parameters

    def build_estimation_model(self, data=None, coords=None, hierarchical=True, save_p_choice=False):

        if data is None:
            data = self.data


        if hierarchical and (coords is None):
            assert('subject' in data.index.names), "Hierarchical estimation requires a multi-index with a 'subject' level."
            coords = {'subject': data.index.unique(level='subject')}
                                              
        with pm.Model(coords=coords) as self.estimation_model:
            paradigm = self._get_paradigm(paradigm=data)
            self.set_paradigm(paradigm)
            self.build_priors(hierarchical=hierarchical)
            parameters = self.get_parameter_values()
            self.build_likelihood(parameters, save_p_choice=save_p_choice)

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
        for key, value in paradigm.items():
            pm.Data(key, value, mutable=True)

    def update_paradigm(self, paradigm):
        raise NotImplementedError

    def build_prediction_model(self, paradigm, parameters,):

        assert(isinstance(parameters, dict) or isinstance(parameters, pd.DataFrame)), "Parameters should be a dictionary or a DataFrame."

        if paradigm is None:
            paradigm = self.data

        if isinstance(parameters, pd.DataFrame):
            parameter_subjects = parameters['subject'] if 'subject' in parameters else parameters.index.get_level_values('subject')

            # Make sure that the unique levels in 'subject' (either index or column) of the paradigm align with the subjects in the parameters
            if 'subject' in paradigm.index.names:
                assert(np.array_equal(paradigm.index.unique(level='subject'), parameter_subjects)), "The unique subjects in the paradigm do not match the subjects in the parameters."
            elif 'subject' in paradigm.columns:
                assert(np.array_equal(paradigm.subject.unique(), parameter_subjects)), "The unique subjects in the paradigm do not match the subjects in the parameters."

            parameters = parameters.to_dict(orient='list')

                                              
        with pm.Model() as self.prediction_model:
            paradigm = self._get_paradigm(paradigm=paradigm)
            self.set_paradigm(paradigm)

            # Make parameters flexible
            for key, value in parameters.items():
                pm.Data(key, value, mutable=True)

            parameters = self.get_parameter_values()
            model_inputs = self.get_model_inputs(parameters)
        
            p = pm.Deterministic('p', var=self._get_choice_predictions(model_inputs))
            pm.Bernoulli('ll_bernoulli', p=p)

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

        if ('subject' in paradigm.index.names) or ('subject' in paradigm.columns):
            data = data.set_index('subject', append=True)
            data = data.reorder_levels(['subject'] + list(data.index.names)[:-1])

        return data

    def sample(self, draws=1000, tune=1000, target_accept=0.8, **kwargs):
        
        with self.estimation_model:
            self.idata = pm.sample(draws, tune=tune, target_accept=target_accept, return_inferencedata=True, **kwargs)
        
        return self.idata            

    def fit_map(self, filter_pars=True, **kwargs):
        with self.estimation_model:
            pars = pm.find_MAP(**kwargs)

        if filter_pars:
            pars = {key: pars[key] for key in self.free_parameters}

        if 'subject' in self.estimation_model.coords:
            pars = pd.DataFrame(pars, index=self.estimation_model.coords['subject'])
            pars.columns.name = 'parameter'
        
        return pars

    def ppc(self, paradigm, idata, var_names=['ll_bernoulli']):

        with self.estimation_model:
            idata = pm.sample_posterior_predictive(idata, var_names=var_names)

        pred = [idata['posterior_predictive'][key].to_dataframe() for key in var_names]
        pred = pd.concat(pred, axis=1, keys=var_names, names=['variable'])
        pred = pred.unstack(['chain', 'draw']).droplevel(1, axis=1)
        pred.index = paradigm.index
        pred = pred.set_index(pd.MultiIndex.from_frame(paradigm), append=True)
        pred = pred.stack('variable')
        pred = pred.reorder_levels(np.roll(pred.index.names, 1)).sort_index()

        return pred

    def get_trialwise_variable(self, key):


        model = pm.Model.get_context()
        
        n_trials = pt.cast(model['_data_n'], int)

        if key == 'n1_evidence_mu':
            return model['log(n1)']

        if key == 'n2_evidence_mu':
            return model['log(n2)']

        if model[f'{key}'].ndim == 1:
            return model[f'{key}'][model['subject_ix']]
        elif model[f'{key}'].ndim == 0:
            return pt.tile(model[f'{key}'], n_trials)

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

    def sample_parameters_from_prior(self, n_subjects=None):
        samples_pars = {}
        for key, pars in self.free_parameters.items():

            if 'sigma_intercept' not in pars.keys():
                pars['sigma_intercept'] = 1.
            if 'transform' not in pars.keys():
                pars['transform'] = 'identity'

            samples_pars[key] = np.random.normal(pars['mu_intercept'], pars['sigma_intercept'], n_subjects)
            if pars['transform'] == 'softplus':
                samples_pars[key] = softplus_np(samples_pars[key])
            elif pars['transform'] == 'logistic':
                samples_pars[key] = logistic_np(samples_pars[key])

        if n_subjects:
            return pd.DataFrame(samples_pars, index=pd.Index(np.arange(1, n_subjects+1), name='subject'))
        else:
            return samples_pars

    def forward_transform(self, data, parameter):
        transform = self.free_parameters[parameter]['transform']

        if transform == 'identity':
            return data
        elif transform == 'softplus':
            return softplus_np(data)
        elif transform == 'logistic':
            return logistic_np(data)

    def backward_transform(self, data, parameter):
        transform = self.free_parameters[parameter]['transform']

        if transform == 'identity':
            return data
        elif transform == 'softplus':
            return inverse_softplus_np(data)
        elif transform == 'logistic':
            return logit_np(data)

class RegressionModel(BaseModel):

    def __init__(self, regressors=None):

        if regressors is None:
            self.regressors = {}
        else:
            self.regressors = regressors

        self.design_matrices = {}
        
    def _get_paradigm(self, paradigm=None):
        paradigm_ = super()._get_paradigm(paradigm)


        for key in self.free_parameters.keys():
            dm = self.build_design_matrix(paradigm, key)
            self.design_matrices[key] = dm
            paradigm_[f'_dm_{key}'] = np.asarray(dm)

        return paradigm_

    def build_design_matrix(self, data, parameter):
        if parameter not in self.regressors:
            self.regressors[parameter] = '1'
        
        return dmatrix(self.regressors[parameter], data)

    def get_trialwise_variable(self, key):

        model = pm.Model.get_context()

        if 'transform' in self.free_parameters[key]:
            transform = self.free_parameters[key]['transform']
        else:
            transform = 'identity'

        dm = model[f'_dm_{key}']

        if key in ['n1_evidence_mu', 'n2_evidence_mu']:
            if key in self.design_matrices.keys():
                trialwise_pars = model[f'log({key[:2]})'] + pt.sum(model[key][model['subject_ix']] * dm, 1)
            else:
                if key == 'n1_evidence_mu':
                    return model['log(n1)']

                if key == 'n2_evidence_mu':
                    return model['log(n2)']

        else:
            pars = model[key]
            if pars.ndim == 1:
                trialwise_pars = pt.sum(pars * dm, 1)
            elif model[key].ndim == 2:
                trialwise_pars = pt.sum(pars[model['subject_ix']] * dm, 1)
            else:
                raise ValueError(f'Unknown dimensionality of {key}')

            if transform == 'softplus':
                trialwise_pars = pt.softplus(trialwise_pars)
            elif transform == 'logistic':
                trialwise_pars = logistic(trialwise_pars)

        return trialwise_pars


    def build_prior(self, name, mu_intercept=None, sigma_intercept=None, sigma_regressors=1.,  transform='identity'):

        model = pm.Model.get_context()
        model.add_coord(f'{name}_regressors', self.design_matrices[name].design_info.column_names)
        
        mu = np.zeros(self.design_matrices[name].shape[1])
        sigma = np.ones(self.design_matrices[name].shape[1]) * sigma_regressors

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

        pm.Normal(name, mu=mu, sigma=sigma, dims=(f'{name}_regressors',))


    def build_hierarchical_nodes(self, name, mu_intercept=0.0, sigma_intercept=1., cauchy_sigma_intercept=0.25, sigma_regressors=1., cauchy_sigma_regressors=0.25, transform='identity'):

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

    def fit_map(self, filter_pars=True, **kwargs):

        def convert_map(map_parameters):
            df = []

            hierarchical = 'subject' in self.estimation_model.coords

            if hierarchical:
                subjects = self.estimation_model.coords['subject']

            parameters = self.design_matrices.keys()
            for parameter, dm in self.design_matrices.items():
                if hierarchical:
                    df.append(pd.DataFrame(map_parameters[parameter], 
                                            index=pd.Index(subjects, name='subject'), columns=pd.Index(dm.design_info.column_names, name='dm_key')))
                else:
                    df.append(pd.Series(map_parameters[parameter], 
                                            index=pd.Index(dm.design_info.column_names, name='dm_key')))

            df = pd.concat(df, keys=parameters, axis=1)

            if not hierarchical:
                df = df.stack().to_frame().T.swaplevel(0, 1, axis=1).sort_index(axis=1)

            df.columns.names = ['parameter', 'dm_key']

            return df

        with self.estimation_model:
            pars = pm.find_MAP(**kwargs)
        
        return convert_map(pars)


    def build_prediction_model(self, paradigm, parameters,):

        if isinstance(parameters, pd.DataFrame):
            hierarchical = True
        elif isinstance(parameters, dict):
            hierarchical = False
        else:
            raise ValueError("Parameters should be a dictionary or a DataFrame.")
        
        with pm.Model() as self.prediction_model:
            paradigm = self._get_paradigm(paradigm=paradigm)
            self.set_paradigm(paradigm)
            
            if hierarchical:
                n_subjects = np.unique(self.prediction_model['subject_ix'].eval()).shape[0]
                assert(len(parameters) == n_subjects), f'Number of subjects in data ({n_subjects}) does not match number of subjects in parameters ({len(parameters)})'

            for key in self.free_parameters.keys():
                dm_keys = self.design_matrices[key].design_info.column_names
                if hierarchical:
                    assert parameters[key].columns.tolist() == dm_keys, f'Parameter {key} needs the following columns: {dm_keys} (in that order)'
                    print(key, parameters[key].shape)
                    pm.Data(key, parameters[key], mutable=True)
                else:
                    value = [parameters[(key, dm_key)] for dm_key in dm_keys]
                    pm.Data(key, value, mutable=True)

            parameters = self.get_parameter_values()
            model_inputs = self.get_model_inputs(parameters)
        
            p = pm.Deterministic('p', var=self._get_choice_predictions(model_inputs))
            pm.Bernoulli('ll_bernoulli', p=p)

    def sample_parameters_from_prior(self, paradigm, n_subjects=None):

        samples_pars = {}

        for par_key, pars in self.free_parameters.items():
            dm = self.build_design_matrix(paradigm, par_key)

            dm_keys = dm.design_info.column_names

            for dm_key in dm_keys:
                key = (par_key, dm_key)

                if dm_keys == 'Intercept':
                    if 'sigma_intercept' not in pars.keys():
                        pars['sigma_intercept'] = 1.
                    samples_pars[key] = np.random.normal(pars['mu_intercept'], pars['sigma_intercept'], n_subjects)

                else:
                    samples_pars[key] = np.random.normal(0, 1, n_subjects)

        if n_subjects:
            return pd.DataFrame(samples_pars, index=pd.Index(np.arange(1, n_subjects+1), name='subject'))
        else:
            return samples_pars