"""Build / smoke tests for continuous-response estimation models.

These models live in:
  * ``bauer.numerosity``       (LogEncoding / FlexibleEncoding /
                                 EfficientEncoding numerosity-estimation models)
  * ``bauer.efficient_coding`` (EfficientPerception / EfficientValuation /
                                 SequentialEfficientCoding / Categorical models)

All inherit from :class:`bauer.estimation.EstimationBaseModel` and build a
``pm.Potential('ll', ...)`` likelihood over a continuous response (rather than
the Bernoulli used in the comparison/risk models). No MCMC is run here —
tests verify that build_estimation_model() succeeds on a tiny multi-subject
paradigm and that the expected free parameters appear.
"""
import pytest
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Numerosity paradigm fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def paradigm_numerosity():
    """Tiny multi-subject, two-condition numerosity-estimation paradigm.

    Each subject performs the same stimulus set under a 'narrow' (10-25) and a
    'wide' (10-40) range condition, with one response per stimulus.
    """
    rng = np.random.default_rng(0)
    stimuli = np.array([10, 14, 18, 22, 26, 30, 34, 38], dtype=float)
    rows = []
    trial = 0
    for subject in (1, 2, 3):
        for cond, lo, hi in [('narrow', 10, 25), ('wide', 10, 40)]:
            for n in stimuli:
                if not (lo <= n <= hi):
                    continue
                rows.append({
                    'subject': subject, 'run': 1, 'trial_nr': trial,
                    'n': n,
                    'range': cond,
                    # Response = stimulus + small noise, clipped into range
                    'response': float(np.clip(n + rng.normal(0, 1), lo, hi)),
                })
                trial += 1
    return pd.DataFrame(rows).set_index(['subject', 'run', 'trial_nr'])


# ---------------------------------------------------------------------------
# Orientation→value paradigm fixture (efficient coding)
# ---------------------------------------------------------------------------

@pytest.fixture
def paradigm_orientation():
    """Tiny orientation→value-estimation paradigm for efficient-coding models."""
    rng = np.random.default_rng(1)
    orientations = np.linspace(0, 180, 9, endpoint=False)
    rows = []
    trial = 0
    for subject in (1, 2):
        for mapping in ('linear', 'cdf'):
            for ori in orientations:
                rows.append({
                    'subject': subject, 'run': 1, 'trial_nr': trial,
                    'orientation': float(ori),
                    'mapping': mapping,
                    # Plausible value response in CHF
                    'response': float(np.clip(2 + ori / 180 * 40 + rng.normal(0, 1),
                                              2.0, 42.0)),
                })
                trial += 1
    return pd.DataFrame(rows).set_index(['subject', 'run', 'trial_nr'])


# ===========================================================================
# Numerosity models
# ===========================================================================

def test_log_encoding_estimation_builds(paradigm_numerosity):
    from bauer.numerosity import LogEncodingEstimationModel
    m = LogEncodingEstimationModel(paradigm=paradigm_numerosity,
                                   grid_resolution=33)
    m.build_estimation_model(data=paradigm_numerosity, hierarchical=True)
    free = set(m.free_parameters)
    assert free == {'nu', 'sigma_motor'}
    # Likelihood must be wired in
    assert 'll' in m.estimation_model.named_vars


@pytest.mark.parametrize('condition_specific', [True, False])
def test_flexible_encoding_estimation_builds(paradigm_numerosity, condition_specific):
    from bauer.numerosity import FlexibleEncodingEstimationModel
    m = FlexibleEncodingEstimationModel(
        paradigm=paradigm_numerosity, grid_resolution=25, n_poly=4,
        condition_specific_encoding=condition_specific,
    )
    m.build_estimation_model(data=paradigm_numerosity, hierarchical=True)
    # Always has nu and sigma_motor
    assert {'nu', 'sigma_motor'}.issubset(m.free_parameters)
    # Has encoding-increment parameters
    enc_pars = [k for k in m.free_parameters if k.startswith('enc_')]
    expected = (m.n_poly - 1) * (len(m.conditions) if condition_specific else 1)
    assert len(enc_pars) == expected


def test_efficient_encoding_estimation_builds(paradigm_numerosity):
    from bauer.numerosity import EfficientEncodingEstimationModel
    m = EfficientEncodingEstimationModel(paradigm=paradigm_numerosity,
                                         grid_resolution=25)
    m.build_estimation_model(data=paradigm_numerosity, hierarchical=True)
    assert set(m.free_parameters) == {'nu', 'sigma_motor'}


# ===========================================================================
# Efficient-coding (orientation → value) models
# ===========================================================================

def test_efficient_perception_model_builds(paradigm_orientation):
    from bauer.efficient_coding import EfficientPerceptionModel
    m = EfficientPerceptionModel(paradigm=paradigm_orientation,
                                 grid_resolution=33)
    m.build_estimation_model(data=paradigm_orientation, hierarchical=True)
    assert set(m.free_parameters) == {'kappa_r'}


def test_efficient_valuation_model_builds(paradigm_orientation):
    from bauer.efficient_coding import EfficientValuationModel
    m = EfficientValuationModel(paradigm=paradigm_orientation,
                                grid_resolution=33)
    m.build_estimation_model(data=paradigm_orientation, hierarchical=True)
    assert set(m.free_parameters) == {'sigma_rep'}


def test_sequential_efficient_coding_builds(paradigm_orientation):
    from bauer.efficient_coding import SequentialEfficientCodingModel
    m = SequentialEfficientCodingModel(paradigm=paradigm_orientation,
                                       grid_resolution=33)
    m.build_estimation_model(data=paradigm_orientation, hierarchical=True)
    # Both perceptual and value noise are free
    assert {'kappa_r', 'sigma_rep'}.issubset(m.free_parameters)


# ===========================================================================
# Helper module functions
# ===========================================================================

def test_orientation_to_value_np_linear_monotone():
    """The 'linear' mapping is approximately linear and monotone in orientation."""
    from bauer.efficient_coding import orientation_to_value_np, V_MIN, V_MAX
    angles = np.linspace(0, 180, 50)
    vals = orientation_to_value_np(angles, mapping='linear')
    assert vals[0] == pytest.approx(V_MIN)
    assert vals[-1] == pytest.approx(V_MAX)
    # Monotonically non-decreasing
    assert (np.diff(vals) >= -1e-9).all()


def test_orientation_to_value_np_vectorized_mapping():
    """When ``mapping`` is an array, each element is dispatched independently."""
    from bauer.efficient_coding import orientation_to_value_np
    angles = np.array([0.0, 45.0, 45.0, 180.0])
    mappings = np.array(['linear', 'cdf', 'inverse_cdf', 'linear'])
    out = orientation_to_value_np(angles, mapping=mappings)
    # At 45° the three mappings diverge (cdf is compressed at the bottom,
    # inverse_cdf is compressed in the middle).
    assert out[1] != out[2]
    # Endpoints are identical across mappings
    assert out[0] == pytest.approx(2.0)
    assert out[3] == pytest.approx(42.0)


def test_long_term_orientation_prior_integrates_to_one():
    """The unnormalized prior should integrate to 1 over [0, 2π]."""
    from bauer.efficient_coding import long_term_orientation_prior_np
    phi = np.linspace(0, 2 * np.pi, 1001)
    p = long_term_orientation_prior_np(phi)
    np.testing.assert_allclose(np.trapezoid(p, phi), 1.0, atol=1e-3)


def test_bernstein_basis_partition_of_unity():
    """Bernstein polynomials of any degree sum to 1 pointwise."""
    from bauer.numerosity import bernstein_basis
    t = np.linspace(0, 1, 51)
    basis = bernstein_basis(t, degree=5)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-10)
