"""Unit tests for bauer.utils.math and bauer.utils.bayes.

Pure-numerical tests for the NumPy helpers that bauer's models rely on
(transforms, posterior combination, PPC summarizers). No pymc/pytensor
context required.
"""
from bauer.utils.bayes import (
    get_posterior_np, get_diff_dist_np, posterior_mean_sd_np,
    summarize_ppc, summarize_ppc_group,
)
from bauer.utils.math import (
    softplus_np, inverse_softplus_np,
    logistic_np, logit_np,
)
import pytest
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# math.py — transforms
# ---------------------------------------------------------------------------

def test_softplus_np_matches_reference():
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    expected = np.log1p(np.exp(x))
    got = softplus_np(x)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_softplus_np_positive_everywhere():
    x = np.linspace(-50, 50, 101)
    assert (softplus_np(x) > 0).all()


def test_softplus_inverse_roundtrip():
    # softplus is invertible on (0, +inf)
    y = np.array([0.01, 0.5, 1.0, 5.0, 20.0])
    np.testing.assert_allclose(softplus_np(inverse_softplus_np(y)), y, atol=1e-6)


def test_logistic_logit_roundtrip():
    p = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    np.testing.assert_allclose(logistic_np(logit_np(p)), p, atol=1e-10)
    x = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    np.testing.assert_allclose(logit_np(logistic_np(x)), x, atol=1e-10)


def test_logistic_np_bounds():
    x = np.array([-100.0, 0.0, 100.0])
    out = logistic_np(x)
    assert 0.0 <= out[0] < 1e-6
    assert abs(out[1] - 0.5) < 1e-10
    assert 1.0 - 1e-6 < out[2] <= 1.0


# ---------------------------------------------------------------------------
# bayes.py — closed-form Gaussian combinators
# ---------------------------------------------------------------------------

def test_get_posterior_np_mean_is_precision_weighted():
    mu1, sd1, mu2, sd2 = 0.0, 1.0, 4.0, 2.0
    post_mu, post_sd = get_posterior_np(mu1, sd1, mu2, sd2)
    # Precision-weighted: τ1=1, τ2=0.25, posterior mean = (τ1*μ1 + τ2*μ2)/(τ1+τ2)
    expected_mu = (1.0 * 0.0 + 0.25 * 4.0) / (1.0 + 0.25)
    expected_sd = np.sqrt(1.0 / (1.0 + 0.25))
    np.testing.assert_allclose(post_mu, expected_mu, atol=1e-10)
    np.testing.assert_allclose(post_sd, expected_sd, atol=1e-10)


def test_get_posterior_np_equal_priors_average():
    """With equal SDs, posterior mean is the midpoint."""
    post_mu, post_sd = get_posterior_np(2.0, 1.5, 8.0, 1.5)
    np.testing.assert_allclose(post_mu, 5.0)
    np.testing.assert_allclose(post_sd, 1.5 / np.sqrt(2))


def test_get_diff_dist_np_combines_variances():
    diff_mu, diff_sd = get_diff_dist_np(1.0, 1.0, 4.0, 2.0)
    np.testing.assert_allclose(diff_mu, 3.0)
    np.testing.assert_allclose(diff_sd, np.sqrt(1.0 + 4.0))


def test_posterior_mean_sd_np_flat_prior_limit():
    """σ_p → ∞ ⇒ weight → 1 ⇒ SD[μ_post] → σ_e (no shrinkage)."""
    out = posterior_mean_sd_np(prior_sd=1e6, evidence_sd=2.0)
    np.testing.assert_allclose(out, 2.0, rtol=1e-6)


def test_posterior_mean_sd_np_strong_shrinkage_limit():
    """σ_p → 0 ⇒ weight → 0 ⇒ SD[μ_post] → 0 (μ_post pinned to prior)."""
    out = posterior_mean_sd_np(prior_sd=1e-6, evidence_sd=2.0)
    assert out < 1e-10


def test_posterior_mean_sd_np_vectorizes():
    out = posterior_mean_sd_np(
        prior_sd=np.array([1e6, 1.0, 1e-6]),
        evidence_sd=np.array([2.0, 2.0, 2.0]),
    )
    assert out.shape == (3,)
    # Endpoints behave; middle is between 0 and σ_e
    assert 0 < out[1] < 2.0


# ---------------------------------------------------------------------------
# bayes.py — PPC summarisers
# ---------------------------------------------------------------------------

def test_summarize_ppc_returns_expected_columns():
    rng = np.random.default_rng(0)
    n_trials, n_samples = 20, 50
    ppc = pd.DataFrame(
        rng.random((n_trials, n_samples)) > 0.5,
        index=pd.MultiIndex.from_product(
            [[1, 2], range(10)], names=['subject', 'trial']),
    ).astype(float)
    out = summarize_ppc(ppc)
    assert list(out.columns) == ['p_predicted', 'hdi025', 'hdi975']
    # HDI brackets the posterior mean
    assert (out['hdi025'] <= out['p_predicted']).all()
    assert (out['p_predicted'] <= out['hdi975']).all()


def test_summarize_ppc_group_two_step_averaging():
    """Group-mean PPC: average within (subject, condition) first, then over subjects."""
    rng = np.random.default_rng(1)
    n_subj, n_cond, n_trials_per, n_samples = 3, 4, 6, 30
    rows = []
    for s in range(n_subj):
        for c in range(n_cond):
            for t in range(n_trials_per):
                rows.append({'subject': s, 'condition': c, 'trial_nr': t})
    df = pd.DataFrame(rows)
    # Posterior-sample columns must NOT be strings (the function selects non-str cols)
    sample_cols = list(zip(np.zeros(n_samples, dtype=int), range(n_samples)))
    for col in sample_cols:
        df[col] = rng.random(len(df))
    out = summarize_ppc_group(df, condition_cols=['condition'])
    assert out.index.name == 'condition'
    assert len(out) == n_cond
    assert {'p_predicted', 'hdi025', 'hdi975'} == set(out.columns)
    assert (out['hdi025'] <= out['p_predicted']).all()
    assert (out['p_predicted'] <= out['hdi975']).all()
