"""Tests for ``bauer.magnitude_estimation.MagnitudeEstimationModel``.

Build tests for every option, analytic sanity checks of the observer (the
response distribution is a proper distribution, censoring puts mass on the
scale ends, a uniform prior compresses), and simulate -> fit -> recover tests
on synthetic data (single-subject MAP; hierarchical NUTS marked ``slow``).
"""
import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings('ignore')

from bauer.magnitude_estimation import MagnitudeEstimationModel  # noqa: E402


def make_paradigm(n_subjects=3, n_trials=120, seed=0, response=True):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_subjects + 1):
        for t in range(n_trials):
            n, u = rng.integers(5, 25, size=2)
            rows.append({'subject': s, 'trial': t, 'n': int(n), 'n_uncued': int(u),
                         'response': int(rng.integers(5, 26)) if response else 5})
    return pd.DataFrame(rows).set_index(['subject', 'trial'])


@pytest.fixture
def paradigm():
    return make_paradigm()


OPTIONS = [
    dict(encoding=e, prior=p, readout=r)
    for e, p, r in itertools.product(('weber', 'constant', 'flexible'),
                                     ('true', 'fitted', 'empirical'),
                                     ('mean', 'median'))
] + [dict(lapse=False), dict(leakage='mixture'), dict(leakage='average')]


@pytest.mark.parametrize('kwargs', OPTIONS)
def test_builds_and_finite_gradient(paradigm, kwargs):
    m = MagnitudeEstimationModel(paradigm, n_measurement_grid=60,
                                 n_hypothesis_grid=80, **kwargs)
    m.build_estimation_model(paradigm, hierarchical=True)
    model = m.estimation_model
    assert 'response' in model.named_vars
    ip = model.initial_point()
    assert np.isfinite(model.compile_logp()(ip))
    assert np.all(np.isfinite(model.compile_dlogp()(ip)))
    expected = {'nu', 'sigma_r'}
    if kwargs.get('encoding') == 'flexible':
        expected.add('nu_exponent')
    if kwargs.get('prior') == 'fitted':
        expected |= {'prior_mu', 'prior_sd'}
    if kwargs.get('lapse', True):
        expected.add('p_lapse')
    if kwargs.get('leakage'):
        expected.add('leak')
    assert set(m.free_parameters) == expected


def _pars(**kw):
    base = dict(nu=0.35, sigma_r=1.5, p_lapse=0.03)
    base.update(kw)
    return base


def test_response_distribution_is_normalised(paradigm):
    m = MagnitudeEstimationModel(paradigm, leakage='average')
    p = m.predict_response_distribution(paradigm, _pars(leak=0.1))
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-8)
    assert (p.values >= 0).all()


def test_censoring_puts_mass_on_scale_ends():
    """With large motor noise the end options carry the censored tails."""
    par = pd.DataFrame({'subject': 1, 'n': [5, 24], 'response': [5, 25]}).set_index('subject')
    m = MagnitudeEstimationModel(par, lapse=False)
    p = m.predict_response_distribution(par, dict(nu=0.1, sigma_r=4.0)).values
    assert p[0, 0] > 0.3          # n = 5: much of the mass at the bottom option
    assert p[1, -1] > p[1, -2]    # n = 24: top option collects the upper tail


def test_uniform_prior_compresses_and_noise_scales_compression():
    par = pd.DataFrame({'subject': 1, 'n': np.arange(5, 25), 'response': 10}).set_index('subject')
    slopes = []
    for nu in (0.15, 0.3, 0.5):
        m = MagnitudeEstimationModel(par, lapse=False)
        pred = m.predict(par, dict(nu=nu, sigma_r=1.0))
        slopes.append(np.polyfit(pred['n'], pred['predicted_mean'], 1)[0])
    assert slopes[0] > slopes[1] > slopes[2]
    assert slopes[0] < 1.0


def test_median_and_mean_readouts_differ_under_weber():
    par = pd.DataFrame({'subject': 1, 'n': np.arange(5, 25), 'response': 10}).set_index('subject')
    p_mean = MagnitudeEstimationModel(par, readout='mean', lapse=False).predict(
        par, dict(nu=0.4, sigma_r=1.0))['predicted_mean']
    p_med = MagnitudeEstimationModel(par, readout='median', lapse=False).predict(
        par, dict(nu=0.4, sigma_r=1.0))['predicted_mean']
    assert np.abs(p_mean - p_med).max() > 0.05


def test_mixture_leak_shifts_report_towards_uncued():
    par = pd.DataFrame({'subject': 1, 'n': [10] * 2, 'n_uncued': [6, 22],
                        'response': 10}).set_index('subject')
    m = MagnitudeEstimationModel(par, leakage='mixture')
    pred = m.predict(par, _pars(leak=0.2))['predicted_mean'].values
    assert pred[1] > pred[0] + 1.0


def test_simulate_shapes_and_support(paradigm):
    m = MagnitudeEstimationModel(paradigm)
    sim = m.simulate(paradigm, _pars(), n_samples=3, rng=0)
    assert len(sim) == 3 * len(paradigm)
    assert sim['simulated_response'].between(5, 25).all()
    assert 'response' not in sim.columns


@pytest.mark.parametrize('encoding,truth', [
    ('weber', dict(nu=0.35, sigma_r=1.5, p_lapse=0.04)),
    ('constant', dict(nu=5.0, sigma_r=1.5, p_lapse=0.04)),
])
def test_single_subject_map_recovery(encoding, truth):
    par = make_paradigm(n_subjects=1, n_trials=1500, seed=3).droplevel('subject')
    gen = MagnitudeEstimationModel(par, encoding=encoding)
    sim = gen.simulate(par, truth, n_samples=1, rng=4)
    data = sim.droplevel('sample').rename(columns={'simulated_response': 'response'})
    fit = MagnitudeEstimationModel(data, encoding=encoding)
    fit.build_estimation_model(data, hierarchical=False)
    est = fit.fit_map(progressbar=False)
    assert est['nu'] == pytest.approx(truth['nu'], rel=0.15)
    assert est['sigma_r'] == pytest.approx(truth['sigma_r'], rel=0.3)
    assert est['p_lapse'] == pytest.approx(truth['p_lapse'], abs=0.025)


@pytest.mark.slow
def test_hierarchical_nuts_recovery():
    par = make_paradigm(n_subjects=5, n_trials=400, seed=5)
    rng = np.random.default_rng(6)
    truth = pd.DataFrame({'nu': rng.uniform(0.2, 0.5, 5),
                          'sigma_r': rng.uniform(1.0, 2.0, 5),
                          'p_lapse': rng.uniform(0.01, 0.05, 5)},
                         index=pd.Index(np.arange(1, 6), name='subject'))
    gen = MagnitudeEstimationModel(par)
    sim = gen.simulate(par, truth, n_samples=1, rng=7)
    data = sim.droplevel('sample').rename(columns={'simulated_response': 'response'})
    fit = MagnitudeEstimationModel(data, n_measurement_grid=100, n_hypothesis_grid=150)
    fit.build_estimation_model(data, hierarchical=True)
    idata = fit.sample(draws=300, tune=300, chains=2, cores=1, target_accept=0.9,
                       progressbar=False, random_seed=1)
    est = idata.posterior['nu'].mean(('chain', 'draw')).values
    assert np.corrcoef(est, truth['nu'].values)[0, 1] > 0.8
    np.testing.assert_allclose(est, truth['nu'].values, atol=0.08)
