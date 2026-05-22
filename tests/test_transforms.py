"""Round-trip tests for BaseModel.forward_transform / backward_transform.

Every bauer model declares per-parameter transforms (identity/softplus/logistic).
Posterior samples are stored in natural (transformed) scale, but several
helpers (sample_parameters_from_prior, fit_map_individual) need to round-trip
through the untransformed scale. These tests confirm the two transforms are
mutually inverse.
"""
import pytest
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


@pytest.fixture
def paradigm():
    n_trials, n_subj = 6, 2
    return pd.DataFrame({
        'subject': np.repeat([1, 2], n_trials),
        'run': 1,
        'trial_nr': np.tile(np.arange(n_trials), n_subj),
        'n1': np.tile([5, 7, 10, 14, 20, 28], n_subj),
        'n2': np.tile([4, 5, 8, 12, 22, 30], n_subj),
        'choice': np.tile([True, False, True, False, True, False], n_subj),
        'rt': 0.5 + np.linspace(0, 1, n_trials * n_subj),
    }).set_index(['subject', 'run', 'trial_nr'])


@pytest.fixture
def model(paradigm):
    """A model with a mix of identity / softplus / logistic / softplus-with-floor
    transforms — built once, parameters annotated on construction."""
    from bauer.models import RiskModel
    return RiskModel(paradigm=paradigm, prior_estimate='shared',
                     fit_seperate_evidence_sd=True)


def test_transforms_known_for_every_free_parameter(model):
    """Every free parameter has an explicit transform — either set or defaulted."""
    for name, spec in model.free_parameters.items():
        spec.setdefault('transform', 'identity')
        assert spec['transform'] in {'identity', 'softplus', 'logistic'}, name


def test_identity_transform_is_passthrough(model):
    """Identity round-trip is exact."""
    # prior_mu is identity-transformed
    id_params = [k for k, v in model.free_parameters.items()
                 if v.get('transform', 'identity') == 'identity']
    assert id_params, "Need at least one identity-transformed parameter."
    for p in id_params:
        x = np.array([-3.0, 0.0, 1.5])
        np.testing.assert_array_equal(model.forward_transform(x, p), x)
        np.testing.assert_array_equal(model.backward_transform(x, p), x)


def test_softplus_transform_roundtrip(model):
    """forward(backward(y)) ≈ y for any positive y, on a softplus parameter."""
    sp_params = [k for k, v in model.free_parameters.items()
                 if v.get('transform', 'identity') == 'softplus']
    assert sp_params, "Need at least one softplus-transformed parameter."
    p = sp_params[0]
    y = np.array([0.01, 0.5, 2.0, 10.0])
    untrans = model.backward_transform(y, p)
    retrans = model.forward_transform(untrans, p)
    np.testing.assert_allclose(retrans, y, atol=1e-6)


def test_logistic_transform_roundtrip():
    """logit→logistic round-trip on a (0,1)-bounded parameter."""
    from bauer.models import RiskLapseModel
    df = pd.DataFrame({
        'subject': [1, 1, 1, 1],
        'run': 1, 'trial_nr': range(4),
        'n1': [10., 12., 15., 20.], 'n2': [12., 10., 18., 14.],
        'p1': [0.55, 1.0, 0.55, 1.0], 'p2': [1.0, 0.55, 1.0, 0.55],
        'choice': [True, False, True, False],
    }).set_index(['subject', 'run', 'trial_nr'])
    m = RiskLapseModel(paradigm=df, prior_estimate='shared')
    assert m.free_parameters['p_lapse']['transform'] == 'logistic'
    p = np.array([0.02, 0.1, 0.5, 0.9])
    untrans = m.backward_transform(p, 'p_lapse')
    retrans = m.forward_transform(untrans, 'p_lapse')
    np.testing.assert_allclose(retrans, p, atol=1e-10)


def test_sample_parameters_from_prior_shape(model):
    """sample_parameters_from_prior(n_subjects=N) returns one row per subject,
    one column per free parameter, all in valid natural scale."""
    pars = model.sample_parameters_from_prior(n_subjects=5)
    assert pars.shape == (5, len(model.free_parameters))
    for name, spec in model.free_parameters.items():
        transform = spec.get('transform', 'identity')
        col = pars[name].values
        if transform == 'softplus':
            assert (col > 0).all(), f"{name} (softplus) produced non-positive values"
        elif transform == 'logistic':
            assert ((col > 0) & (col < 1)).all(), f"{name} (logistic) outside (0,1)"
        # identity unconstrained
