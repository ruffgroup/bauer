import pandas as pd
import pymc as pm
import numpy as np
from .utils import cumulative_normal, get_diff_dist, get_posterior
import aesara.tensor as at


class MagnitudeComparisonModel(object):
    
    def __init__(self, data):
        self.data = data
        self.subject_ix, self.unique_subjects = pd.factorize(self.data.index.get_level_values('subject'))
        self.n_subjects = len(self.unique_subjects) 
            
    def build_model(self):

        base_numbers = self.data.n1.unique()

        # True means choose 2
        choices = self.data.choice
        
        mean_n1 = np.mean(np.log(base_numbers)) # same as np.mean(np.log(d['n1']))
        std_n1 = np.std(np.log(base_numbers))
        mean_n2 = np.mean(np.log(self.data['n2']))
        std_n2 = np.std(np.log(self.data['n2']))
        
        n1_mu = np.log(self.data['n1'])
        n2_mu = np.log(self.data['n2'])

        self.coords = {
            "subject": self.unique_subjects,
            "order":['first', 'second']}

                                              
        with pm.Model(coords=self.coords) as self.model:
                
            def build_hierarchical_nodes(name, mu=0.0, sigma=.5):
                nodes = {}

                nodes[f'{name}_mu_untransformed'] = pm.Normal(f"{name}_mu_untransformed", 
                                              mu=mu, 
                                              sigma=sigma)

                nodes[f'{name}_mu'] = pm.Deterministic(name=f'{name}_mu', var=at.softplus(nodes[f'{name}_mu_untransformed']))
                
                nodes[f'{name}_sd'] = pm.HalfCauchy(f'{name}_sd', .25)
                nodes[f'{name}_offset'] = pm.Normal(f'{name}_offset', mu=0, sigma=1, dims=('subject',))
                nodes[f'{name}_untransformed'] = pm.Deterministic(f'{name}_untransformed', nodes[f'{name}_mu_untransformed'] + nodes[f'{name}_sd'] * nodes[f'{name}_offset'],
                                              dims=('subject',))
                
                nodes[f'{name}'] = pm.Deterministic(name=f'{name}',
                                                              var=at.softplus(nodes[f'{name}_untransformed']),
                                                   dims=('subject',))
                
                return nodes

            # Hyperpriors for group nodes
            
            nodes = {}
            
            n1_mu = pm.MutableData('n1_mu', n1_mu)
            n2_mu = pm.MutableData('n2_mu', n2_mu)
            
            nodes.update(build_hierarchical_nodes('evidence_sd1'), mu=-1.)
            nodes.update(build_hierarchical_nodes('evidence_sd2'), mu=-1.)
            nodes.update(build_hierarchical_nodes('n2_prior_mu'), mu=mean_n2)
            
            evidence_sd = pm.Deterministic(name='evidence_sd',
                                           var=at.stack((nodes['evidence_sd1'], nodes['evidence_sd2']), axis=1),
                                           dims=('subject', 'order'))

            n1_prior_mu = pm.MutableData('n1_prior_mu', mean_n1)
            n1_prior_sd = pm.MutableData('std_n1', std_n1)
            n2_prior_sd = pm.MutableData('std_n2', std_n2)
            
            
            post_n1_mu, post_n1_sd = get_posterior(n1_prior_mu, 
                                                    n1_prior_sd, 
                                                    n1_mu,
                                                    evidence_sd[self.subject_ix, 0])
            post_n2_mu, post_n2_sd = get_posterior(nodes['n2_prior_mu'][self.subject_ix],#.eval(), #n2_prior_mu, 
                                                    n2_prior_sd,
                                                    n2_mu,
                                                    evidence_sd[self.subject_ix, 1])

            diff_mu, diff_sd = get_diff_dist(post_n1_mu, post_n1_sd, post_n2_mu, post_n2_sd)
            p = cumulative_normal(at.log(1), diff_mu, diff_sd)
            ll = pm.Bernoulli('ll_bernoulli', p=p, observed=choices)
            
    def sample(self, draws=1000, tune=1000, target_accept=0.8):
        
        with self.model:
            self.trace = pm.sample(draws, tune=tune, target_accept=target_accept, return_inferencedata=True)
        
        return self.trace            