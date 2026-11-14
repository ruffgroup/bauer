import warnings

from ..core import BaseModel, LapseModel, RegressionModel
from ..utils.bayes import cumulative_normal, get_diff_dist
from ..utils.math import inverse_softplus_np


class PsychophysicalModel(BaseModel):
    """Psychophysical model for two-alternative forced choice with a sensitivity and bias parameter.

    Parameters ``nu`` (discrimination sensitivity, softplus-transformed) and ``bias``
    (decision criterion) describe the probability of choosing option 2 given stimuli x1 and x2.
    Paradigm requires columns ``x1``, ``x2``, and ``choice``.

    ``nu`` and ``bias`` are in the raw units of ``x``, while the default priors
    (notably the HalfNormal(0.25) group-SD prior) assume order-1 parameters.
    **Standardize x so that x2 - x1 is of order 1** (e.g. divide by its SD);
    otherwise hierarchical shrinkage is near-total (large units) or absent
    (small units). A warning is raised when this looks off.
    """

    paradigm_keys = ['x1', 'x2']
    base_parameters = ['nu', 'bias']
    # nu and bias are in the raw units of x: warn if x2 - x1 is far from
    # order 1 (the default priors would then mis-regularize; see
    # BaseModel._check_input_scales).
    scale_sensitive_keys = ('x1', 'x2')

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

    def __init__(self, paradigm, regressors=None, save_trialwise_estimates=False,
                 fixed_regressors=None, random_regressors=None):
        RegressionModel.__init__(self, regressors, fixed_regressors=fixed_regressors,
                                 random_regressors=random_regressors)
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
