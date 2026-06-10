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
                                 fit_separate_evidence_sd=True, fit_prior=True)
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
        paradigm=paradigm_magnitude, fit_separate_evidence_sd=True,
        spline_order=4, fit_prior=True,
    )
    m.build_estimation_model(paradigm=paradigm_magnitude, hierarchical=True)
    assert any(k.startswith('n1_evidence_sd_spline') for k in m.free_parameters)


@pytest.mark.parametrize('prior_estimate', ['objective', 'shared', 'full', 'klw'])
def test_risk_model_prior_estimates(paradigm_risk, prior_estimate):
    from bauer.models import RiskModel
    m = RiskModel(paradigm=paradigm_risk, prior_estimate=prior_estimate,
                  fit_separate_evidence_sd=True)
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
    m = Cls(paradigm=paradigm_magnitude, fit_separate_evidence_sd=True, **extras)
    try:
        m.build_estimation_model(data=paradigm_magnitude, hierarchical=True)
    except TypeError:
        m.build_estimation_model(paradigm=paradigm_magnitude, hierarchical=True)
    assert 'a' in m.free_parameters
    assert 't0' in m.free_parameters


def test_ddm_memory_as_sv_builds(paradigm_magnitude):
    """memory_as_sv=True maps held-n1 memory noise to across-trial drift
    variability `sv`, normalizing drift by perceptual noise only. Requires the
    shared_perceptual_noise front-end and the hssm sdv WFPT."""
    pytest.importorskip('hssm')
    from bauer.models import DDMMagnitudeComparisonModel
    m = DDMMagnitudeComparisonModel(
        paradigm=paradigm_magnitude, fit_separate_evidence_sd=True,
        memory_model='shared_perceptual_noise', memory_as_sv=True)
    m.build_estimation_model(data=paradigm_magnitude, hierarchical=True,
                             save_p_choice=True)
    assert m.memory_as_sv is True
    assert {'perceptual_noise_sd', 'memory_noise_sd', 'a', 't0'}.issubset(
        m.free_parameters)
    # drift + sv deterministics exist and evaluate finite, with sv >= 0.
    with m.estimation_model:
        drift = m.estimation_model['drift'].eval()
        sv = m.estimation_model['sv'].eval()
    assert np.all(np.isfinite(drift))
    assert np.all(np.isfinite(sv))
    assert np.all(sv >= 0)


def test_ddm_memory_as_sv_requires_shared_perceptual(paradigm_magnitude):
    """memory_as_sv with the wrong (independent) memory_model errors clearly."""
    pytest.importorskip('hssm')
    from bauer.models import DDMMagnitudeComparisonModel
    m = DDMMagnitudeComparisonModel(
        paradigm=paradigm_magnitude, fit_separate_evidence_sd=True,
        memory_model='independent', memory_as_sv=True)
    with pytest.raises(ValueError, match='shared_perceptual_noise'):
        m.build_estimation_model(data=paradigm_magnitude, hierarchical=True)


@pytest.mark.parametrize('cls_name,extras', [
    ('DDMRiskModel', {'fit_v_scale': True}),
])
def test_ddm_risk_model_builds(paradigm_risk, cls_name, extras):
    pytest.importorskip('hssm')
    import bauer.models as M
    Cls = getattr(M, cls_name)
    m = Cls(paradigm=paradigm_risk, prior_estimate='full',
            fit_separate_evidence_sd=True, **extras)
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
                fit_separate_evidence_sd=True)
        try:
            m.build_estimation_model(data=paradigm_risk, hierarchical=True)
        except TypeError:
            m.build_estimation_model(paradigm=paradigm_risk, hierarchical=True)
    else:
        kwargs = {'fit_separate_evidence_sd': True}
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


@pytest.mark.parametrize('cls_name,kind', [
    ('DDMMagnitudeComparisonModel', 'magnitude'),
    ('DDMRiskModel', 'risk'),
    ('RaceDiffusionMagnitudeComparisonModel', 'magnitude'),
    ('RaceDiffusionRiskModel', 'risk'),
])
def test_ddm_rdm_lapse_models_build(paradigm_magnitude, paradigm_risk,
                                    cls_name, kind):
    """The base DDM/RDM models carry the RT-aware contaminant mixin. With a
    *hierarchical* ``p_outlier`` they expose a free, logit-transformed outlier
    rate and build the contaminant-mixture likelihood without errors.

    0.3.0 removed the dedicated ``*Lapse*`` DDM/RDM classes: the contaminant is
    now baked into the base classes (fixed ``p_outlier=0.05`` by default). Set
    ``model.p_outlier='hierarchical'`` to estimate a per-subject rate."""
    if cls_name.startswith('DDM'):
        pytest.importorskip('hssm')
    import numpy as np
    import bauer.models as M
    Cls = getattr(M, cls_name)
    if kind == 'risk':
        m = Cls(paradigm=paradigm_risk, prior_estimate='objective',
                fit_separate_evidence_sd=True)
        data = paradigm_risk
    else:
        kwargs = {'fit_separate_evidence_sd': True, 'fit_prior': True}
        m = Cls(paradigm=paradigm_magnitude, **kwargs)
        data = paradigm_magnitude
    # Promote the fixed default p_outlier=0.05 to a free hierarchical rate.
    m.p_outlier = 'hierarchical'
    m.lapse_group = 'logit_normal'
    m.free_parameters = m.get_free_parameters()
    try:
        m.build_estimation_model(data=data, hierarchical=True)
    except TypeError:
        m.build_estimation_model(paradigm=data, hierarchical=True)
    assert 'p_outlier' in m.free_parameters
    assert m.free_parameters['p_outlier']['transform'] == 'logistic'
    assert 'a' in m.free_parameters and 't0' in m.free_parameters
    # Contaminant log-density = log(0.5) - log(20) ≈ -3.689 (HSSM convention).
    assert np.isclose(m._lapse_logp_const(), np.log(0.5) - np.log(20.0))
    # ll node uses the lapse-mixture CustomDist.
    assert 'll' in m.estimation_model.named_vars


@pytest.mark.parametrize('cls_name,kind', [
    ('DDMMagnitudeComparisonModel', 'magnitude'),
    ('RaceDiffusionMagnitudeComparisonModel', 'magnitude'),
])
def test_lapse_beta_group_prior_builds(paradigm_magnitude, cls_name, kind):
    """The ``lapse_group='beta'`` option replaces the logit-Normal group prior
    on ``p_outlier`` with a hierarchical Beta (group_mu + concentration kappa),
    which removes the logit funnel on clean data.

    In 0.3.0 ``lapse_group`` is a class/instance attribute (not a constructor
    kwarg) and the contaminant only becomes a free parameter when
    ``p_outlier='hierarchical'``."""
    if cls_name.startswith('DDM'):
        pytest.importorskip('hssm')
    import bauer.models as M
    Cls = getattr(M, cls_name)
    m = Cls(paradigm=paradigm_magnitude, fit_separate_evidence_sd=True,
            fit_prior=True)
    m.p_outlier = 'hierarchical'
    m.lapse_group = 'beta'
    m.free_parameters = m.get_free_parameters()
    assert m.free_parameters['p_outlier']['transform'] == 'beta'
    m.build_estimation_model(data=paradigm_magnitude, hierarchical=True)
    nv = m.estimation_model.named_vars
    # Hierarchical-Beta exposes group mean + concentration, not a logit SD.
    assert 'p_outlier_mu' in nv
    assert 'p_outlier_kappa' in nv
    assert 'p_outlier_untransformed' not in nv
    assert 'll' in nv


def test_lapse_model_beta_group_prior_static():
    """The static-choice LapseModel also accepts the Beta '1/x' group prior."""
    import numpy as np
    import pandas as pd
    from bauer.core import LapseModel
    from bauer.models.magnitude import MagnitudeComparisonModel

    class MagLapse(LapseModel, MagnitudeComparisonModel):
        lapse_group = 'beta'

    rng = np.random.default_rng(0)
    rows = []
    for s in range(1, 4):
        for t in range(30):
            n1, n2 = rng.uniform(5, 25), rng.uniform(5, 25)
            rows.append((s, t, n1, n2, n2 > n1))
    df = pd.DataFrame(rows, columns=['subject', 'trial', 'n1', 'n2', 'choice'])
    df = df.set_index(['subject', 'trial'])

    m = MagLapse(paradigm=df)
    assert m.free_parameters['p_lapse']['transform'] == 'beta'
    m.build_estimation_model(data=df, hierarchical=True)
    nv = m.estimation_model.named_vars
    assert 'p_lapse_mu' in nv and 'p_lapse_kappa' in nv
    assert 'p_lapse_untransformed' not in nv


def test_flat_observer_prior_magnitude(paradigm_magnitude):
    """Magnitude/Flex/PowerLaw + DDMFlex with flat_observer_prior=True should
    build without populating n*_prior_* keys, and should reject fit_prior=True."""
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


# ---------------------------------------------------------------------------
# Regression-model prior equivalence
# ---------------------------------------------------------------------------

def test_intercept_only_regression_matches_basic_priors(paradigm_magnitude):
    """Regression model with intercept-only formulas should produce the same
    Normal priors as the basic hierarchical model.

    Regression-tests a real bug (2026-05-23) where
    RegressionModel.build_hierarchical_nodes silently dropped the
    mu_intercept/sigma_intercept for softplus-transformed parameters,
    leaving them at Normal(0, sigma_regressors). For DDM t0 this set the
    prior mean to ~0.69 s — larger than most RTs — and produced r̂ ≈ 4
    on every regression-DDM fit until the bug was found.
    """
    from bauer.models import (DDMMagnitudeComparisonModel,
                              DDMMagnitudeComparisonRegressionModel)

    m_basic = DDMMagnitudeComparisonModel(
        paradigm=paradigm_magnitude,
        fit_separate_evidence_sd=True, fit_prior=True)
    m_basic.build_estimation_model(data=paradigm_magnitude, hierarchical=True)

    m_reg = DDMMagnitudeComparisonRegressionModel(
        paradigm=paradigm_magnitude,
        fit_separate_evidence_sd=True, fit_prior=True,
        regressors={'n1_evidence_sd': '1'})  # intercept-only formula
    m_reg.build_estimation_model(data=paradigm_magnitude, hierarchical=True)

    def basic_normal_params(model, name):
        for v in (f'{name}_mu_untransformed', f'{name}_mu'):
            if v in model.named_vars:
                rv = model.named_vars[v]
                return (float(rv.owner.inputs[-2].eval()),
                        float(rv.owner.inputs[-1].eval()))
        raise KeyError(name)

    def reg_intercept_params(model, name):
        rv = model.named_vars[f'{name}_mu']
        return (float(rv.owner.inputs[-2].eval()[0]),
                float(rv.owner.inputs[-1].eval()[0]))

    for param in ('n1_evidence_sd', 't0', 'a', 'prior_mu', 'prior_sd'):
        mu_b, sd_b = basic_normal_params(m_basic.estimation_model, param)
        mu_r, sd_r = reg_intercept_params(m_reg.estimation_model, param)
        assert abs(mu_b - mu_r) < 1e-6, (
            f'Intercept prior mu for {param!r} differs between basic '
            f'({mu_b:.6f}) and intercept-only regression ({mu_r:.6f}) — '
            f'the RegressionModel softplus-branch bug has regressed.'
        )
        assert abs(sd_b - sd_r) < 1e-6, (
            f'Intercept prior sigma for {param!r} differs between basic '
            f'({sd_b:.6f}) and intercept-only regression ({sd_r:.6f}) — '
            f'check the sigma_intercept default in '
            f'RegressionModel.build_hierarchical_nodes.'
        )


# ===========================================================================
# bauer 0.3.0 release coverage
# ===========================================================================

@pytest.fixture
def paradigm_magnitude_group(paradigm_magnitude):
    """`paradigm_magnitude` with a between-subjects ``group`` column.

    Subjects 1,2 -> 'control'; subject 3 -> 'dyscalculia'. The label is
    constant within subject (a true between-subjects contrast), so a random
    *slope* on ``C(group)`` is non-identified — exactly the structure the
    fixed/random split is designed to handle.
    """
    df = paradigm_magnitude.copy()
    subj = df.index.get_level_values('subject')
    df['group'] = np.where(subj == 3, 'dyscalculia', 'control')
    return df


def _sd_rv_distname(model, name):
    """Name of the PyMC distribution backing the group-SD RV ``{name}_sd``.

    The RV's op type is e.g. ``HalfNormalRV`` / ``HalfCauchyRV``; strip the
    trailing ``RV`` so callers can compare against ``'HalfNormal'`` etc."""
    rv = model.named_vars[f'{name}_sd']
    opname = type(rv.owner.op).__name__
    return opname[:-2] if opname.endswith('RV') else opname


# --- 1. group_sd_dist default + override -----------------------------------

def test_group_sd_dist_default_is_halfnormal(paradigm_magnitude):
    """Default ``group_sd_dist == 'halfnormal'`` (0.3.0 changed it from
    halfcauchy) and the built group-SD RV is a HalfNormal."""
    from bauer.models import MagnitudeComparisonRegressionModel
    m = MagnitudeComparisonRegressionModel(
        paradigm=paradigm_magnitude, fit_separate_evidence_sd=True,
        fit_prior=True, regressors={'n1_evidence_sd': '1'})
    assert m.group_sd_dist == 'halfnormal'
    m.build_estimation_model(data=paradigm_magnitude, hierarchical=True)
    assert _sd_rv_distname(m.estimation_model, 'n1_evidence_sd') == 'HalfNormal'


def test_group_sd_dist_halfcauchy_override(paradigm_magnitude):
    """``group_sd_dist='halfcauchy'`` (pre-0.3.0 behaviour) yields a HalfCauchy
    group-SD RV."""
    from bauer.models import MagnitudeComparisonRegressionModel
    m = MagnitudeComparisonRegressionModel(
        paradigm=paradigm_magnitude, fit_separate_evidence_sd=True,
        fit_prior=True, regressors={'n1_evidence_sd': '1'})
    m.group_sd_dist = 'halfcauchy'
    m.build_estimation_model(data=paradigm_magnitude, hierarchical=True)
    assert _sd_rv_distname(m.estimation_model, 'n1_evidence_sd') == 'HalfCauchy'


# --- 2. fixed_regressors / random_regressors API ---------------------------

def test_regressors_kwarg_is_deprecated(paradigm_magnitude):
    """The legacy ``regressors=`` keyword emits a DeprecationWarning."""
    from bauer.models import MagnitudeComparisonRegressionModel
    with pytest.warns(DeprecationWarning, match='regressors'):
        MagnitudeComparisonRegressionModel(
            paradigm=paradigm_magnitude, fit_separate_evidence_sd=True,
            fit_prior=True, regressors={'n1_evidence_sd': '1'})


def test_regressors_and_new_kwargs_mutually_exclusive(paradigm_magnitude):
    """Passing both ``regressors`` and the new fixed/random kwargs raises."""
    from bauer.models import MagnitudeComparisonRegressionModel
    with pytest.raises(ValueError, match='regressors'):
        MagnitudeComparisonRegressionModel(
            paradigm=paradigm_magnitude, fit_separate_evidence_sd=True,
            fit_prior=True,
            regressors={'n1_evidence_sd': '1'},
            fixed_regressors={'n1_evidence_sd': '1'})


def test_legacy_regressors_is_full_random_slope(paradigm_magnitude_group):
    """``regressors=`` maps bit-for-bit to a full random slope: the per-subject
    ``*_offset`` spans EVERY fixed-design column (the legacy graph), i.e. it
    includes the between-subjects group contrast."""
    from bauer.models import MagnitudeComparisonRegressionModel
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = MagnitudeComparisonRegressionModel(
            paradigm=paradigm_magnitude_group, fit_separate_evidence_sd=True,
            fit_prior=True, regressors={'n1_evidence_sd': 'C(group)'})
    m.build_estimation_model(data=paradigm_magnitude_group, hierarchical=True)
    model = m.estimation_model
    # Legacy path: offset dims == ('subject', '{name}_regressors'); the
    # regressors coord carries both the Intercept and the group column.
    offset_dims = model.named_vars_to_dims['n1_evidence_sd_offset']
    assert 'n1_evidence_sd_regressors' in offset_dims
    reg_coord = list(model.coords['n1_evidence_sd_regressors'])
    assert any('Intercept' in c for c in reg_coord)
    assert any('group' in c for c in reg_coord)
    # No subset '_re' coord in the legacy full-slope path.
    assert 'n1_evidence_sd_re' not in model.coords


def test_fixed_random_split_drops_group_random_effect(paradigm_magnitude_group):
    """``fixed_regressors={p:'C(group)'}`` + ``random_regressors={p:'1'}``:
    the population-mean group contrast lives in the fixed design but carries NO
    per-subject offset (its column is absent from the ``*_offset`` random
    structure), while the intercept DOES."""
    from bauer.models import MagnitudeComparisonRegressionModel
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = MagnitudeComparisonRegressionModel(
            paradigm=paradigm_magnitude_group, fit_separate_evidence_sd=True,
            fit_prior=True,
            fixed_regressors={'n1_evidence_sd': 'C(group)'},
            random_regressors={'n1_evidence_sd': '1'})
    m.build_estimation_model(data=paradigm_magnitude_group, hierarchical=True)
    model = m.estimation_model
    # Fixed design (the group-mean coefficient vector) keeps both columns.
    fixed_cols = list(model.coords['n1_evidence_sd_regressors'])
    assert any('Intercept' in c for c in fixed_cols)
    assert any('group' in c for c in fixed_cols)
    # The random structure (subset path) spans only the Intercept; the
    # between-subjects group column has NO per-subject offset.
    re_cols = list(model.coords['n1_evidence_sd_re'])
    assert any('Intercept' in c for c in re_cols)
    assert not any('group' in c for c in re_cols)
    # The offset RV is indexed by the subset coord, not the full design.
    offset_dims = model.named_vars_to_dims['n1_evidence_sd_offset']
    assert 'n1_evidence_sd_re' in offset_dims
    assert 'n1_evidence_sd_regressors' not in offset_dims


# --- 3. memory_as_sv composes with the contaminant -------------------------

def test_memory_as_sv_composes_with_contaminant(paradigm_magnitude):
    """``memory_as_sv=True`` (shared_perceptual_noise front-end) builds with the
    default fixed contaminant (p_outlier=0.05): the graph exposes an ``sv`` node
    AND the lapse-mixture ``ll`` node, and ``sv`` is finite & non-negative."""
    pytest.importorskip('hssm')
    from bauer.models import DDMMagnitudeComparisonModel
    m = DDMMagnitudeComparisonModel(
        paradigm=paradigm_magnitude, fit_separate_evidence_sd=True,
        memory_model='shared_perceptual_noise', memory_as_sv=True)
    # Default p_outlier is the FIXED 0.05 contaminant (not a free parameter).
    assert m.p_outlier == 0.05
    assert 'p_outlier' not in m.free_parameters
    m.build_estimation_model(data=paradigm_magnitude, hierarchical=True)
    nv = m.estimation_model.named_vars
    assert 'sv' in nv
    assert 'll' in nv          # contaminant-mixture likelihood node
    with m.estimation_model:
        sv = nv['sv'].eval()
    assert np.all(np.isfinite(sv))
    assert np.all(sv >= 0)


# --- 4. race fit_w_d / fit_w_s toggles -------------------------------------

def test_race_fit_w_toggles(paradigm_magnitude):
    """``fit_w_d=False`` removes ``w_d`` (fixes the discriminative gain to 1);
    ``fit_w_s=False`` removes ``w_s``; defaults keep both."""
    from bauer.models import RaceDiffusionMagnitudeComparisonModel

    def free_pars(**toggles):
        m = RaceDiffusionMagnitudeComparisonModel(
            paradigm=paradigm_magnitude, fit_separate_evidence_sd=True,
            fit_prior=True)
        for k, v in toggles.items():
            setattr(m, k, v)
        return set(m.get_free_parameters())

    both = free_pars()
    assert 'w_d' in both and 'w_s' in both

    no_wd = free_pars(fit_w_d=False)
    assert 'w_d' not in no_wd and 'w_s' in no_wd

    no_ws = free_pars(fit_w_s=False)
    assert 'w_s' not in no_ws and 'w_d' in no_ws


# --- 5. consistent_choice_noise --------------------------------------------

def test_consistent_choice_noise_builds_and_differs(paradigm_magnitude):
    """A static ``MagnitudeComparisonRegressionModel`` with
    ``consistent_choice_noise=True`` builds and respects the flag, producing a
    different choice log-likelihood than the default raw-evidence_sd
    normalization on the same data."""
    from bauer.models import MagnitudeComparisonRegressionModel

    def build(consistent):
        m = MagnitudeComparisonRegressionModel(
            paradigm=paradigm_magnitude, fit_separate_evidence_sd=True,
            fit_prior=True, regressors={'n1_evidence_sd': '1'})
        m.consistent_choice_noise = consistent
        m.build_estimation_model(data=paradigm_magnitude, hierarchical=True)
        return m

    m_default = build(False)
    m_consistent = build(True)
    assert m_consistent.consistent_choice_noise is True
    assert m_default.consistent_choice_noise is False

    # Evaluate each model's total logp at its initial point. The likelihoods
    # differ because the choice noise is normalized differently.
    def total_logp(m):
        model = m.estimation_model
        ip = model.initial_point()
        return float(model.compile_logp()(ip))

    lp_default = total_logp(m_default)
    lp_consistent = total_logp(m_consistent)
    assert np.isfinite(lp_default) and np.isfinite(lp_consistent)
    assert not np.isclose(lp_default, lp_consistent), (
        'consistent_choice_noise=True did not change the choice likelihood — '
        'the flag is being ignored.')
