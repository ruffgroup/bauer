import warnings

from ..core import BaseModel, LapseModel, RegressionModel
from ..utils.bayes import cumulative_normal, get_diff_dist
from ..utils.math import inverse_softplus_np


class PsychophysicalModel(BaseModel):
    """Psychophysical model for two-alternative forced choice with a sensitivity and bias parameter.

    Parameters ``nu`` (discrimination sensitivity, softplus-transformed) and ``bias``
    (decision criterion) describe the probability of choosing option 2 given stimuli x1 and x2.
    Paradigm requires columns ``x1``, ``x2``, and ``choice``.
    """

    paradigm_keys = ['x1', 'x2']
    base_parameters = ['nu', 'bias']

    def __init__(self, paradigm=None):
        super().__init__(paradigm)

    def get_free_parameters(self):
        return {
            'nu':   {'mu_intercept': inverse_softplus_np(1.), 'sigma_intercept': 10., 'transform': 'softplus'},
            'bias': {'mu_intercept': 0,                       'sigma_intercept': 10., 'transform': 'identity'},
        }

    def _get_choice_predictions(self, model_inputs):
        mu1, mu2 = model_inputs['x1'], model_inputs['x2']
        sd = model_inputs['nu']
        diff_mu, diff_sd = get_diff_dist(mu2, sd, mu1, sd)
        return cumulative_normal(model_inputs['bias'], diff_mu, diff_sd)


class PsychophysicalLapseModel(LapseModel, PsychophysicalModel):
    """PsychophysicalModel extended with a lapse rate parameter."""
    ...


class PsychophysicalRegressionModel(RegressionModel, PsychophysicalModel):
    """PsychophysicalModel with patsy formula regression on nu and/or bias."""

    def __init__(self, paradigm, regressors, save_trialwise_estimates=False):
        RegressionModel.__init__(self, regressors)
        PsychophysicalModel.__init__(self, paradigm)


class PsychophysicalLapseRegressionModel(LapseModel, PsychophysicalRegressionModel):
    """PsychophysicalModel with both a lapse rate and patsy formula regression."""
    ...


# ─── Deprecated aliases ───────────────────────────────────────────────────────
# The old "Psychometric*" names are kept as subclasses so existing code keeps
# working; using them emits a ``DeprecationWarning``. Prefer the
# ``Psychophysical*`` names in new code.

def _deprecated_alias(old_cls, new_cls):
    class _Aliased(new_cls):
        def __init__(self, *args, **kwargs):
            warnings.warn(
                f'{old_cls} is a deprecated alias for {new_cls.__name__}; '
                f'use {new_cls.__name__} instead.',
                DeprecationWarning, stacklevel=2,
            )
            super().__init__(*args, **kwargs)
    _Aliased.__name__ = old_cls
    _Aliased.__qualname__ = old_cls
    _Aliased.__doc__ = (
        f'Deprecated alias for :class:`{new_cls.__name__}`.\n\n'
        f'Will be removed in a future release. Use ``{new_cls.__name__}``.'
    )
    return _Aliased


PsychometricModel = _deprecated_alias('PsychometricModel', PsychophysicalModel)
PsychometricLapseModel = _deprecated_alias('PsychometricLapseModel', PsychophysicalLapseModel)
PsychometricRegressionModel = _deprecated_alias('PsychometricRegressionModel', PsychophysicalRegressionModel)
PsychometricLapseRegressionModel = _deprecated_alias('PsychometricLapseRegressionModel',
                                                     PsychophysicalLapseRegressionModel)
