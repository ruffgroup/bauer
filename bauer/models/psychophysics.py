import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pandas as pd
from ..core import BaseModel, LapseModel, RegressionModel
from ..utils.bayes import cumulative_normal, get_diff_dist
from ..utils.math import inverse_softplus_np

class PsychometricModel(BaseModel):
    """Psychometric model for two-alternative forced choice with a sensitivity and bias parameter.

    Parameters ``nu`` (discrimination sensitivity, softplus-transformed) and ``bias``
    (decision criterion) describe the probability of choosing option 2 given stimuli x1 and x2.
    Paradigm requires columns ``x1``, ``x2``, and ``choice``.
    """

    paradigm_keys = ['x1', 'x2']
    base_parameters = ['nu', 'bias']

    def __init__(self, paradigm=None):
        super().__init__(paradigm)

    def get_free_parameters(self):

            free_parameters = {}

            free_parameters['nu'] = {'mu_intercept': inverse_softplus_np(1.), 'sigma_intercept': 10., 'transform': 'softplus'}
            free_parameters['bias'] = {'mu_intercept': 0, 'sigma_intercept': 10., 'transform': 'identity'}

            return free_parameters

    def _get_choice_predictions(self, model_inputs):

        mu1 = model_inputs['x1']
        mu2 = model_inputs['x2']
        sd1 = model_inputs['nu']
        sd2 = model_inputs['nu']

        diff_mu, diff_sd = get_diff_dist(mu2, sd2, mu1, sd1)

        return cumulative_normal(model_inputs['bias'], diff_mu, diff_sd)

class PsychometricLapseModel(LapseModel, PsychometricModel):
    """PsychometricModel extended with a lapse rate parameter."""
    ...

class PsychometricRegressionModel(RegressionModel, PsychometricModel):
    """PsychometricModel with patsy formula regression on nu and/or bias."""

    def __init__(self, paradigm, regressors, save_trialwise_estimates=False):
        RegressionModel.__init__(self, regressors)
        PsychometricModel.__init__(self, paradigm)

class PsychometricLapseRegressionModel(LapseModel, PsychometricRegressionModel):
    """PsychometricModel with both a lapse rate and patsy formula regression."""
    ...
