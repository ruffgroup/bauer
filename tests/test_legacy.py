"""Smoke tests for the legacy safe-vs-risky models.

These models live in ``bauer.models.legacy`` and are also re-exported from
``bauer.models`` for backwards compatibility (commit 2302d9a, April 2026).
Tests do NOT run MCMC — they verify that:
- each class is importable via both paths,
- ``build_estimation_model()`` succeeds on a tiny multi-subject paradigm,
- ``get_free_parameters()`` returns the expected keys for the main options.
"""
import pytest
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


@pytest.fixture
def paradigm_safe_vs_risky_gain():
    """Tiny multi-subject safe-vs-risky paradigm with positive magnitudes."""
    rng = np.random.default_rng(0)
    n_trials, n_subj = 16, 3
    risky_first = np.tile([True, False] * (n_trials // 2), n_subj)
    risky_n = np.tile([10.0, 15.0, 20.0, 25.0] * (n_trials // 4), n_subj)
    safe_n = np.tile([12.0, 18.0] * (n_trials // 2), n_subj)
    p_risky = np.tile([0.35, 0.55] * (n_trials // 2), n_subj)
    n1 = np.where(risky_first, risky_n, safe_n)
    n2 = np.where(risky_first, safe_n, risky_n)
    p1 = np.where(risky_first, p_risky, 1.0)
    p2 = np.where(risky_first, 1.0, p_risky)
    return pd.DataFrame({
        'subject': np.repeat([1, 2, 3], n_trials),
        'run': 1,
        'trial_nr': np.tile(np.arange(n_trials), n_subj),
        'n1': n1, 'n2': n2, 'p1': p1, 'p2': p2,
        'choice': rng.random(n_trials * n_subj) > 0.5,
        'stim': np.tile(['A', 'B'] * (n_trials // 2), n_subj),
    }).set_index(['subject', 'run', 'trial_nr'])


@pytest.fixture
def paradigm_safe_vs_risky_loss(paradigm_safe_vs_risky_gain):
    df = paradigm_safe_vs_risky_gain.copy()
    df['n1'] = -df['n1']
    df['n2'] = -df['n2']
    return df


@pytest.fixture
def paradigm_joint_safe_vs_risky():
    """Tiny multi-subject paradigm with gain AND loss trials."""
    rng = np.random.default_rng(1)
    n_per_domain, n_subj = 16, 3
    rows = []
    trial_counter = 0
    for s in (1, 2, 3):
        for is_gain in (1, 0):
            for k in range(n_per_domain):
                risky_first = (k % 2 == 0)
                n_r = float([10, 15, 20, 25][k % 4]) * (1 if is_gain else -1)
                n_s = float([12, 18][k % 2]) * (1 if is_gain else -1)
                p_r = float([0.35, 0.55][k % 2])
                if risky_first:
                    n1, p1, n2, p2 = n_r, p_r, n_s, 1.0
                else:
                    n1, p1, n2, p2 = n_s, 1.0, n_r, p_r
                rows.append({
                    'subject': s, 'run': 1, 'trial_nr': trial_counter,
                    'n1': n1, 'n2': n2, 'p1': p1, 'p2': p2,
                    'is_gain': is_gain,
                    'choice': bool(rng.random() > 0.5),
                })
                trial_counter += 1
    return pd.DataFrame(rows).set_index(['subject', 'run', 'trial_nr'])


def test_legacy_models_importable_from_models_package():
    """The four legacy classes are re-exported from ``bauer.models``."""
    from bauer.models import (
        SafeVsRiskyModel, SafeVsRiskyRegressionModel,
        SafeVsRiskyMemoryModel, JointSafeVsRiskyModel,
    )
    from bauer.models.legacy import (
        SafeVsRiskyModel as LSafeVsRiskyModel,
        SafeVsRiskyRegressionModel as LSafeVsRiskyRegressionModel,
        SafeVsRiskyMemoryModel as LSafeVsRiskyMemoryModel,
        JointSafeVsRiskyModel as LJointSafeVsRiskyModel,
    )
    # same objects (re-exported, not redefined)
    assert SafeVsRiskyModel is LSafeVsRiskyModel
    assert SafeVsRiskyRegressionModel is LSafeVsRiskyRegressionModel
    assert SafeVsRiskyMemoryModel is LSafeVsRiskyMemoryModel
    assert JointSafeVsRiskyModel is LJointSafeVsRiskyModel


@pytest.mark.parametrize('domain', ['gain', 'loss'])
def test_safe_vs_risky_model_builds(paradigm_safe_vs_risky_gain,
                                     paradigm_safe_vs_risky_loss, domain):
    from bauer.models import SafeVsRiskyModel
    df = (paradigm_safe_vs_risky_gain if domain == 'gain'
          else paradigm_safe_vs_risky_loss)
    m = SafeVsRiskyModel(df, domain=domain)
    m.build_estimation_model(df, hierarchical=True)
    free = set(m.get_free_parameters())
    assert {'prior_mu_risky', 'prior_mu_safe',
            'prior_sd_risky', 'prior_sd_safe',
            'evidence_sd_n1', 'evidence_sd_n2'} == free


def test_safe_vs_risky_model_shared_prior(paradigm_safe_vs_risky_gain):
    from bauer.models import SafeVsRiskyModel
    m = SafeVsRiskyModel(paradigm_safe_vs_risky_gain, domain='gain',
                          separate_priors=False, separate_evidence_sd=False)
    m.build_estimation_model(paradigm_safe_vs_risky_gain, hierarchical=True)
    free = set(m.get_free_parameters())
    assert {'prior_mu', 'prior_sd', 'evidence_sd'} == free


def test_safe_vs_risky_model_fixed_priors(paradigm_safe_vs_risky_gain):
    from bauer.models import SafeVsRiskyModel
    m = SafeVsRiskyModel(paradigm_safe_vs_risky_gain, domain='gain',
                          fix_prior_mus=True, fix_prior_sds=True)
    m.build_estimation_model(paradigm_safe_vs_risky_gain, hierarchical=True)
    free = set(m.get_free_parameters())
    # only evidence noise should be free
    assert free == {'evidence_sd_n1', 'evidence_sd_n2'}


def test_safe_vs_risky_regression_builds(paradigm_safe_vs_risky_gain):
    from bauer.models import SafeVsRiskyRegressionModel
    m = SafeVsRiskyRegressionModel(
        paradigm_safe_vs_risky_gain, domain='gain',
        regressors={'evidence_sd_n1': 'stim'},
    )
    m.build_estimation_model(paradigm_safe_vs_risky_gain, hierarchical=True)


@pytest.mark.parametrize('memory_model,combine_noise,expected', [
    ('shared_perceptual_noise', 'add_sd',
     {'prior_mu_risky', 'prior_mu_safe', 'prior_sd_risky', 'prior_sd_safe',
      'encoding_noise_sd', 'memory_noise_sd'}),
    ('shared_perceptual_noise', 'variance',
     {'prior_mu_risky', 'prior_mu_safe', 'prior_sd_risky', 'prior_sd_safe',
      'encoding_noise_sd', 'memory_noise_sd'}),
    ('independent', 'add_sd',
     {'prior_mu_risky', 'prior_mu_safe', 'prior_sd_risky', 'prior_sd_safe',
      'evidence_sd_n1', 'evidence_sd_n2'}),
])
def test_safe_vs_risky_memory_builds(paradigm_safe_vs_risky_gain,
                                      memory_model, combine_noise, expected):
    from bauer.models import SafeVsRiskyMemoryModel
    m = SafeVsRiskyMemoryModel(
        paradigm_safe_vs_risky_gain, domain='gain',
        memory_model=memory_model, combine_noise=combine_noise,
    )
    m.build_estimation_model(paradigm_safe_vs_risky_gain, hierarchical=True)
    assert set(m.get_free_parameters()) == expected


@pytest.mark.parametrize('prior_scope', ['global', 'role', 'domain', 'domain_role'])
@pytest.mark.parametrize('evidence_scope',
                          ['global', 'position', 'domain', 'domain_position'])
def test_joint_safe_vs_risky_builds(paradigm_joint_safe_vs_risky,
                                     prior_scope, evidence_scope):
    from bauer.models import JointSafeVsRiskyModel
    m = JointSafeVsRiskyModel(
        paradigm_joint_safe_vs_risky,
        prior_scope=prior_scope, evidence_scope=evidence_scope,
    )
    m.build_estimation_model(paradigm_joint_safe_vs_risky, hierarchical=True)
    # sanity-check that the right top-level parameter names appear
    free = set(m.get_free_parameters())
    if prior_scope == 'domain_role':
        assert {'gain_prior_mu_risky', 'loss_prior_mu_safe'}.issubset(free)
    if evidence_scope == 'domain_position':
        assert {'gain_evidence_sd_n1', 'loss_evidence_sd_n2'}.issubset(free)


def test_joint_safe_vs_risky_fixed_priors(paradigm_joint_safe_vs_risky):
    from bauer.models import JointSafeVsRiskyModel
    m = JointSafeVsRiskyModel(
        paradigm_joint_safe_vs_risky,
        prior_scope='domain_role', evidence_scope='domain_position',
        fix_prior_mus=True, fix_prior_sds=True,
    )
    m.build_estimation_model(paradigm_joint_safe_vs_risky, hierarchical=True)
    free = set(m.get_free_parameters())
    # only evidence noise should be free
    assert free == {'gain_evidence_sd_n1', 'gain_evidence_sd_n2',
                    'loss_evidence_sd_n1', 'loss_evidence_sd_n2'}


def test_safe_vs_risky_memory_invalid_model_rejected(paradigm_safe_vs_risky_gain):
    """An unknown memory_model is caught at construction (BaseModel.__init__
    calls get_free_parameters)."""
    from bauer.models import SafeVsRiskyMemoryModel
    with pytest.raises(ValueError, match="Unknown memory_model"):
        SafeVsRiskyMemoryModel(paradigm_safe_vs_risky_gain, domain='gain',
                                memory_model='not_a_real_option')


def test_joint_safe_vs_risky_accepts_domain_column():
    """The joint model auto-derives `is_gain` from a `domain` column if present."""
    from bauer.models import JointSafeVsRiskyModel
    rng = np.random.default_rng(2)
    rows = []
    for i in range(20):
        is_gain = i % 2 == 0
        n_r = 15.0 * (1 if is_gain else -1)
        n_s = 12.0 * (1 if is_gain else -1)
        rows.append({'subject': 1, 'run': 1, 'trial_nr': i,
                     'n1': n_r, 'n2': n_s,
                     'p1': 0.55, 'p2': 1.0,
                     'domain': 'gain' if is_gain else 'loss',
                     'choice': bool(rng.random() > 0.5)})
    df = pd.DataFrame(rows).set_index(['subject', 'run', 'trial_nr'])
    m = JointSafeVsRiskyModel(df, prior_scope='global', evidence_scope='global')
    # `is_gain` should now live on the stored frame
    assert 'is_gain' in m.paradigm.columns
    assert set(m.paradigm['is_gain'].unique()) == {0, 1}
    m.build_estimation_model(m.paradigm, hierarchical=False)


def test_joint_safe_vs_risky_missing_domain_info_raises():
    """Without `is_gain` or `domain`, init should raise a clear error."""
    from bauer.models import JointSafeVsRiskyModel
    df = pd.DataFrame({
        'subject': [1, 1, 1, 1],
        'run': 1, 'trial_nr': range(4),
        'n1': [10., 12., 15., 20.], 'n2': [12., 10., 18., 14.],
        'p1': [0.55, 1.0, 0.55, 1.0], 'p2': [1.0, 0.55, 1.0, 0.55],
        'choice': [True, False, True, False],
    }).set_index(['subject', 'run', 'trial_nr'])
    with pytest.raises(ValueError, match="is_gain"):
        JointSafeVsRiskyModel(df)
