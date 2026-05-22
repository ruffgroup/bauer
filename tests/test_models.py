"""Smoke tests for bauer model classes.

These tests do NOT run MCMC. They only verify that:
- each model class can build_estimation_model() on a tiny synthetic paradigm
- each prior_estimate option produces sensible free-parameter sets
- DDM/RDM models correctly require rt + choice columns
"""
import pytest
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


@pytest.fixture
def paradigm_magnitude():
    """Tiny multi-subject magnitude-comparison paradigm."""
    n1 = [5, 7, 10, 14, 20, 28]
    n2 = [4, 5, 8, 12, 22, 30]
    n_trials = 12
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'subject': np.repeat([1, 2, 3], n_trials),
        'run': 1,
        'trial_nr': np.tile(np.arange(n_trials), 3),
        'n1': np.tile(n1 + n1, 3),
        'n2': np.tile(n2 + n2, 3),
        'choice': rng.random(n_trials * 3) > 0.5,
        'rt': 0.5 + rng.random(n_trials * 3),
    }).set_index(['subject', 'run', 'trial_nr'])


@pytest.fixture
def paradigm_risk():
    """Tiny multi-subject risky-choice paradigm."""
    rng = np.random.default_rng(0)
    n_trials = 16
    n_subj = 3
    # n_trials must equal 2 * len(safe_n) so that safe_n.repeat(2) has the
    # right length (was previously 6 vs 16 — broadcast error).
    safe_n = np.array([5, 7, 10, 14, 18, 22, 26, 30])
    return pd.DataFrame({
        'subject': np.repeat([1, 2, 3], n_trials),
        'run': 1,
        'trial_nr': np.tile(np.arange(n_trials), n_subj),
        'n1': np.tile(safe_n.repeat(2) * (1 + 0.5 * rng.random(n_trials)), n_subj).round(),
        'n2': np.tile(safe_n.repeat(2), n_subj),
        'p1': np.tile([0.55, 1.0] * (n_trials // 2), n_subj),
        'p2': np.tile([1.0, 0.55] * (n_trials // 2), n_subj),
        'choice': rng.random(n_trials * n_subj) > 0.5,
        'rt': 0.5 + rng.random(n_trials * n_subj),
    }).set_index(['subject', 'run', 'trial_nr'])


def test_magnitude_model_builds(paradigm_magnitude):
    from bauer.models import MagnitudeComparisonModel
    m = MagnitudeComparisonModel(paradigm=paradigm_magnitude,
                                  fit_seperate_evidence_sd=True, fit_prior=True)
    m.build_estimation_model(data=paradigm_magnitude, hierarchical=True)
    assert 'a' not in m.free_parameters
    assert 'n1_evidence_sd' in m.free_parameters
    assert 'n2_evidence_sd' in m.free_parameters
    assert 'prior_mu' in m.free_parameters


def test_flexible_noise_builds(paradigm_magnitude):
    from bauer.models import FlexibleNoiseComparisonModel
    # spline_order=4 (df>=4) — patsy requires df >= degree+1 when an Intercept
    # is included.
    m = FlexibleNoiseComparisonModel(
        paradigm=paradigm_magnitude, fit_seperate_evidence_sd=True,
        spline_order=4, fit_prior=True,
    )
    m.build_estimation_model(paradigm=paradigm_magnitude, hierarchical=True)
    assert any(k.startswith('n1_evidence_sd_spline') for k in m.free_parameters)


@pytest.mark.parametrize('prior_estimate', ['objective', 'shared', 'full', 'klw'])
def test_risk_model_prior_estimates(paradigm_risk, prior_estimate):
    from bauer.models import RiskModel
    m = RiskModel(paradigm=paradigm_risk, prior_estimate=prior_estimate,
                   fit_seperate_evidence_sd=True)
    m.build_estimation_model(data=paradigm_risk, hierarchical=True)
    pars = set(m.free_parameters)
    expected_priors = {
        'objective': set(),
        'shared': {'prior_mu', 'prior_sd'},
        'full': {'risky_prior_mu', 'risky_prior_sd',
                  'safe_prior_mu', 'safe_prior_sd'},
        'klw': {'prior_sd'},
    }
    assert expected_priors[prior_estimate].issubset(pars)


@pytest.mark.parametrize('cls_name,extras', [
    ('DDMMagnitudeComparisonModel', {'fit_v_scale': True, 'fit_prior': True}),
    ('DDMFlexibleNoiseComparisonModel', {'spline_order': 4, 'fit_prior': True}),
])
def test_ddm_magnitude_models_build(paradigm_magnitude, cls_name, extras):
    pytest.importorskip('hssm')
    import bauer.models as M
    Cls = getattr(M, cls_name)
    m = Cls(paradigm=paradigm_magnitude, fit_seperate_evidence_sd=True, **extras)
    try:
        m.build_estimation_model(data=paradigm_magnitude, hierarchical=True)
    except TypeError:
        m.build_estimation_model(paradigm=paradigm_magnitude, hierarchical=True)
    assert 'a' in m.free_parameters
    assert 't0' in m.free_parameters


@pytest.mark.parametrize('cls_name,extras', [
    ('DDMRiskModel', {'fit_v_scale': True}),
])
def test_ddm_risk_model_builds(paradigm_risk, cls_name, extras):
    pytest.importorskip('hssm')
    import bauer.models as M
    Cls = getattr(M, cls_name)
    m = Cls(paradigm=paradigm_risk, prior_estimate='full',
             fit_seperate_evidence_sd=True, **extras)
    m.build_estimation_model(data=paradigm_risk, hierarchical=True)
    assert {'a', 't0', 'risky_prior_mu', 'safe_prior_mu'}.issubset(m.free_parameters)


@pytest.mark.parametrize('cls_name', [
    'RaceDiffusionMagnitudeComparisonModel',
    'RaceDiffusionFlexibleNoiseComparisonModel',
    'RaceDiffusionRiskModel',
])
def test_rdm_models_build(paradigm_magnitude, paradigm_risk, cls_name):
    import bauer.models as M
    Cls = getattr(M, cls_name)
    # Race models do not accept `fit_v_scale` (the v_scale parameter belongs
    # to DDM only).
    if 'Risk' in cls_name:
        m = Cls(paradigm=paradigm_risk, prior_estimate='shared',
                 fit_seperate_evidence_sd=True)
        try:
            m.build_estimation_model(data=paradigm_risk, hierarchical=True)
        except TypeError:
            m.build_estimation_model(paradigm=paradigm_risk, hierarchical=True)
    else:
        kwargs = {'fit_seperate_evidence_sd': True}
        if 'Flexible' in cls_name:
            kwargs['spline_order'] = 4
        else:
            kwargs['fit_prior'] = True
        m = Cls(paradigm=paradigm_magnitude, **kwargs)
        try:
            m.build_estimation_model(data=paradigm_magnitude, hierarchical=True)
        except TypeError:
            m.build_estimation_model(paradigm=paradigm_magnitude, hierarchical=True)
    assert 'a' in m.free_parameters
    assert 't0' in m.free_parameters


def test_flat_observer_prior_magnitude(paradigm_magnitude):
    """Magnitude/Flex/PowerLaw + DDMFlex with flat_observer_prior=True should
    build without populating n*_prior_* keys, and should reject fit_prior=True."""
    import pytensor.tensor as pt
    from bauer.models import (MagnitudeComparisonModel,
                                FlexibleNoiseComparisonModel,
                                PowerLawNoiseComparisonModel)
    # Static comparison
    m = MagnitudeComparisonModel(paradigm=paradigm_magnitude,
                                  fit_prior=False, flat_observer_prior=True)
    m.build_estimation_model(data=paradigm_magnitude, hierarchical=True)
    assert m.flat_observer_prior is True
    assert 'prior_mu' not in m.free_parameters
    assert 'prior_sd' not in m.free_parameters

    # Mutual exclusivity
    with pytest.raises(ValueError, match='flat_observer_prior'):
        MagnitudeComparisonModel(paradigm=paradigm_magnitude,
                                  fit_prior=True, flat_observer_prior=True)

    # Flexible spline (spline_order=4 dodges patsy df>=4 requirement)
    mf = FlexibleNoiseComparisonModel(paradigm=paradigm_magnitude,
                                        spline_order=4, fit_prior=False,
                                        flat_observer_prior=True)
    mf.build_estimation_model(paradigm=paradigm_magnitude, hierarchical=True)
    assert mf.flat_observer_prior is True

    # Power-law
    mp = PowerLawNoiseComparisonModel(paradigm=paradigm_magnitude,
                                        fit_prior=False, flat_observer_prior=True)
    mp.build_estimation_model(data=paradigm_magnitude, hierarchical=True)
    assert mp.flat_observer_prior is True
    assert 'noise_exponent' in mp.free_parameters


def test_flat_observer_prior_ddm_flex(paradigm_magnitude):
    """DDMFlexibleNoiseComparisonModel composes flat_observer_prior with the DDM mixin."""
    pytest.importorskip('hssm')
    from bauer.models import DDMFlexibleNoiseComparisonModel
    m = DDMFlexibleNoiseComparisonModel(paradigm=paradigm_magnitude,
                                          spline_order=4, fit_prior=False,
                                          flat_observer_prior=True)
    m.build_estimation_model(paradigm=paradigm_magnitude, hierarchical=True)
    assert m.flat_observer_prior is True
    assert 'a' in m.free_parameters
    assert 't0' in m.free_parameters


def test_data_loaders():
    """Verify all bundled-data loaders work."""
    from bauer.utils.data import (load_garcia2022,
                                    load_dehollander2024_risk,
                                    load_dehollander2024_symbolic,
                                    load_dehollander_tms_risk)
    df1 = load_garcia2022(task='magnitude')
    assert len(df1) > 0
    df2 = load_garcia2022(task='risk')
    assert len(df2) > 0
    df3 = load_dehollander2024_risk()
    assert df3.index.get_level_values('subject').nunique() == 30
    df4 = load_dehollander2024_symbolic()
    assert df4.index.get_level_values('subject').nunique() == 58
    df5 = load_dehollander_tms_risk()
    assert df5.index.get_level_values('subject').nunique() == 35
    df6 = load_dehollander_tms_risk(tms_only=False)
    assert df6.index.get_level_values('subject').nunique() == 73
