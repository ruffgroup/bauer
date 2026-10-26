"""Tests for ``bauer.models.EstimationTaskModel``.

* builds + finite gradients for every component combination;
* the closed-form path (log space + log-normal prior) agrees with the grid path;
* a uniform prior compresses, more so with more noise;
* alpha -> 0 of the power space reproduces the log space;
* simulate -> fit -> recover (single-subject MAP; hierarchical NUTS, ``slow``).
"""
import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings('ignore')

from bauer.models import (EstimationTaskModel, PowerSpace, LogSpace, LinearSpace,  # noqa: E402
                          UniformPrior, GaussianPrior, TruncatedGaussianPrior)


def make_paradigm(n_subjects=3, n_trials=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{'subject': s, 'trial': t, 'n': int(rng.integers(5, 25)),
             'response': int(rng.integers(5, 26))}
            for s in range(1, n_subjects + 1) for t in range(n_trials)]
    return pd.DataFrame(rows).set_index(['subject', 'trial'])


def grid_paradigm():
    return pd.DataFrame({'subject': 1, 'n': np.arange(5, 25), 'response': 10}).set_index('subject')


SPACES = {'log': LogSpace, 'linear': LinearSpace, 'free': PowerSpace}
PRIORS = {'uniform': lambda: UniformPrior(4.5, 24.5), 'gaussian': GaussianPrior,
          'truncated': lambda: TruncatedGaussianPrior(5, 24)}


@pytest.mark.parametrize('space,prior,motor', itertools.product(SPACES, PRIORS, (False, True)))
def test_builds_and_finite_gradient(space, prior, motor):
    par = make_paradigm()
    m = EstimationTaskModel(par, encoding_space=SPACES[space](), prior=PRIORS[prior](),
                            motor_noise=motor, n_measurement_grid=80, n_hypothesis_grid=100)
    m.build_estimation_model(par, hierarchical=True)
    ip = m.estimation_model.initial_point()
    assert np.isfinite(m.estimation_model.compile_logp()(ip))
    assert np.all(np.isfinite(m.estimation_model.compile_dlogp()(ip)))
    expected = {'nu', 'p_lapse'}
    if space == 'free':
        expected.add('alpha')
    if prior in ('gaussian', 'truncated'):
        expected |= {'prior_mu', 'prior_sd'}
    if motor:
        expected.add('sigma_r')
    assert set(m.free_parameters) == expected
    assert m.closed_form == (space == 'log' and prior == 'gaussian' and not motor)


@pytest.mark.parametrize('nu,mu,sd', [(0.3, 11.0, 0.4), (0.5, 14.0, 0.3), (0.2, 9.0, 0.8)])
def test_closed_form_matches_grid(nu, mu, sd):
    par = grid_paradigm()
    pars = dict(nu=nu, prior_mu=mu, prior_sd=sd, p_lapse=0.02)
    closed = EstimationTaskModel(par, encoding_space=LogSpace(), prior=GaussianPrior())
    assert closed.closed_form
    grid = EstimationTaskModel(par, encoding_space=PowerSpace(alpha=1e-9), prior=GaussianPrior(),
                               n_measurement_grid=400, n_hypothesis_grid=600,
                               hypothesis_range=(0.5, 80.0))
    assert not grid.closed_form
    pc = closed.predict_response_distribution(par, pars).values
    pg = grid.predict_response_distribution(par, pars).values
    np.testing.assert_allclose(pc, pg, atol=2e-3)


def test_power_space_alpha_to_zero_is_log():
    par = grid_paradigm()
    pars = dict(nu=0.4, p_lapse=0.0)
    p_log = EstimationTaskModel(par, encoding_space=LogSpace()).predict_response_distribution(
        par, pars).values
    p_pow = EstimationTaskModel(par, encoding_space=PowerSpace(1e-6)).predict_response_distribution(
        par, pars).values
    np.testing.assert_allclose(p_log, p_pow, atol=5e-4)  # |alpha| floored at 1e-4
    # free alpha at 0 is the same thing
    p_free = EstimationTaskModel(par, encoding_space=PowerSpace()).predict_response_distribution(
        par, dict(pars, alpha=0.0)).values
    np.testing.assert_allclose(p_log, p_free, atol=5e-4)


def test_uniform_prior_compresses():
    par = grid_paradigm()
    slopes = []
    for nu in (0.15, 0.3, 0.5):
        pred = EstimationTaskModel(par, lapse=False).predict(par, dict(nu=nu))
        slopes.append(np.polyfit(pred['n'], pred['predicted_mean'], 1)[0])
    assert 1.0 > slopes[0] > slopes[1] > slopes[2] > 0.2
    # the estimate stays inside the prior's support
    pred = EstimationTaskModel(par, lapse=False).predict(par, dict(nu=0.5))
    assert pred['predicted_mean'].between(4.5, 24.5).all()


def test_linear_gaussian_prior_gives_flat_sd():
    """Linear space + Gaussian prior: constant report s.d. away from the scale ends."""
    par = grid_paradigm()
    pred = EstimationTaskModel(par, encoding_space=LinearSpace(), prior=GaussianPrior(),
                               lapse=False).predict(par, dict(nu=0.35, prior_mu=14.0, prior_sd=0.4))
    mid = pred[(pred.n >= 9) & (pred.n <= 20)]['predicted_sd']
    assert mid.max() - mid.min() < 0.15


def test_response_distribution_and_simulation():
    par = make_paradigm()
    m = EstimationTaskModel(par, encoding_space=PowerSpace(), prior=GaussianPrior())
    pars = dict(nu=0.35, alpha=0.5, prior_mu=12.0, prior_sd=0.5, p_lapse=0.03)
    p = m.predict_response_distribution(par, pars)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-8)
    sim = m.simulate(par, pars, n_samples=2, rng=0)
    assert len(sim) == 2 * len(par)
    assert sim['simulated_response'].between(5, 25).all()


@pytest.mark.parametrize('space', ['log', 'linear'])
def test_single_subject_map_recovery(space):
    par = make_paradigm(n_subjects=1, n_trials=1500, seed=3).droplevel('subject')
    truth = dict(nu=0.35, p_lapse=0.04)
    gen = EstimationTaskModel(par, encoding_space=SPACES[space]())
    data = gen.simulate(par, truth, rng=4).droplevel('sample').rename(
        columns={'simulated_response': 'response'})
    fit = EstimationTaskModel(data, encoding_space=SPACES[space]())
    fit.build_estimation_model(data, hierarchical=False)
    est = fit.fit_map(progressbar=False)
    assert est['nu'] == pytest.approx(truth['nu'], rel=0.1)
    assert est['p_lapse'] == pytest.approx(truth['p_lapse'], abs=0.02)


@pytest.mark.slow
def test_hierarchical_nuts_recovery_free_alpha():
    par = make_paradigm(n_subjects=6, n_trials=400, seed=5)
    rng = np.random.default_rng(6)
    truth = pd.DataFrame({'nu': rng.uniform(0.2, 0.5, 6), 'alpha': rng.uniform(-0.2, 1.0, 6),
                          'p_lapse': rng.uniform(0.01, 0.05, 6)},
                         index=pd.Index(np.arange(1, 7), name='subject'))
    m = EstimationTaskModel(par, encoding_space=PowerSpace(), n_measurement_grid=120,
                            n_hypothesis_grid=150)
    table, idata = m.parameter_recovery(par, truth, rng=7, draws=300, tune=300, chains=2,
                                        cores=1, target_accept=0.9, progressbar=False,
                                        random_seed=1)
    assert np.corrcoef(table['true']['nu'], table['recovered']['nu'])[0, 1] > 0.8
    assert np.corrcoef(table['true']['alpha'], table['recovered']['alpha'])[0, 1] > 0.6


def test_subject_covariate_on_noise():
    par = make_paradigm()
    z = {1: -1.0, 2: 0.0, 3: 1.0}
    par['z'] = [z[s] for s in par.index.get_level_values('subject')]
    m = EstimationTaskModel(par, covariates={'nu': 'z'})
    m.build_estimation_model(par, hierarchical=True)
    assert 'nu_z' in m.estimation_model.named_vars
    ip = m.estimation_model.initial_point()
    assert np.all(np.isfinite(m.estimation_model.compile_dlogp()(ip)))


@pytest.mark.parametrize('space', ['log', 'linear'])
def test_truncated_prior_limits(space):
    """Narrow and mid-range = the untruncated Gaussian (both spaces). Very wide =
    uniform in f_alpha space, which is uniform in n only for the linear space
    (in log space it tends to a 1/n prior), so that limit is checked for linear."""
    par = grid_paradigm()
    sp = SPACES[space]
    base = dict(nu=0.3, p_lapse=0.0)
    wide = dict(base, prior_mu=14.0, prior_sd=50.0)
    for discrete, uni in (((True, UniformPrior(5, 24, discrete=True)),
                           (False, UniformPrior(4.5, 24.5))) if space == 'linear' else ()):
        lo, hi = (5, 24) if discrete else (4.5, 24.5)
        p_tr = EstimationTaskModel(par, encoding_space=sp(), prior=TruncatedGaussianPrior(
            lo, hi, discrete=discrete), lapse=False).predict_response_distribution(par, wide).values
        p_un = EstimationTaskModel(par, encoding_space=sp(), prior=uni,
                                   lapse=False).predict_response_distribution(par, base).values
        np.testing.assert_allclose(p_tr, p_un, atol=2e-3)
    narrow = dict(base, prior_mu=14.0, prior_sd=0.08)
    p_tr = EstimationTaskModel(par, encoding_space=sp(), prior=TruncatedGaussianPrior(
        4.5, 24.5, discrete=False), lapse=False).predict_response_distribution(par, narrow).values
    p_g = EstimationTaskModel(par, encoding_space=sp(), prior=GaussianPrior(),
                              lapse=False).predict_response_distribution(par, narrow).values
    np.testing.assert_allclose(p_tr, p_g, atol=2e-3)
