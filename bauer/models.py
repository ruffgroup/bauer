import pandas as pd
import pymc as pm
import numpy as np
from .utils import cumulative_normal, get_diff_dist, get_posterior
import aesara.tensor as at
from patsy import dmatrix
from .core import BaseModel, build_hierarchical_nodes_bounded


class MagnitudeComparisonModel(BaseModel):

    def __init__(self, data, fit_n2_prior_mu=True):

        self.fit_n2_prior_mu = fit_n2_prior_mu

        super().__init__(data)

    def get_model_inputs(self, model, paradigm):

        model_inputs = {}
        model_inputs['n1_prior_mu'] = at.mean(at.log(paradigm['n1']))
        model_inputs['n1_prior_std'] = at.std(at.log(paradigm['n1']))

        if self.fit_n2_prior_mu:
            model_inputs['n2_prior_mu'] = model['n2_prior_mu'][paradigm['subject_ix']]
        else:
            model_inputs['n2_prior_mu'] = at.mean(at.log(paradigm['n2']))

        model_inputs['n2_prior_std'] = at.std(at.log(paradigm['n2']))

        model_inputs['threshold'] =  0.0

        model_inputs['n1_evidence_sd'] = model['evidence_sd1'][paradigm['subject_ix']]
        model_inputs['n2_evidence_sd'] = model['evidence_sd2'][paradigm['subject_ix']]

        return model_inputs

    def build_priors(self, model, paradigm):
        build_hierarchical_nodes_bounded('evidence_sd1', mu=-1.)
        build_hierarchical_nodes_bounded('evidence_sd2', mu=-1.)

        pm.Deterministic('evidence_sd', at.stack((model['evidence_sd1'], model['evidence_sd2']), axis=1), dims=('subject', 'order'))

        if self.fit_n2_prior_mu:
            build_hierarchical_nodes_bounded('n2_prior_mu', mu=at.mean(at.log(paradigm['n2'])))

class RiskModel(BaseModel):
    ...