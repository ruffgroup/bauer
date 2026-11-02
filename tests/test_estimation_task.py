"""Tests for ``bauer.models.EstimationTaskModel``.

* builds + finite gradients for every component combination;
* the closed-form path (log space + log-normal prior) agrees with the grid path;
* a uniform prior compresses, more so with more noise;
* alpha -> 0 of the power space reproduces the log space;
* simulate -> fit -> recover (single-subject MAP; hierarchical NUTS, ``slow``);
* the uncued-magnitude terms (leak, prior_leak, response_pull, swap) and the pooled lapse
  reduce exactly to the plain model.
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
    # kept small so it runs in ~1.5 min (60 x 80 grid, 2 x 50 numpyro draws); 400 trials,
    # a 120 x 150 grid and 2 x 300 pymc draws took about an hour on CI for the same answer
    par = make_paradigm(n_subjects=6, n_trials=250, seed=5)
    rng = np.random.default_rng(6)
    truth = pd.DataFrame({'nu': rng.uniform(0.2, 0.5, 6), 'alpha': rng.uniform(-0.2, 1.0, 6),
                          'p_lapse': rng.uniform(0.01, 0.05, 6)},
                         index=pd.Index(np.arange(1, 7), name='subject'))
    m = EstimationTaskModel(par, encoding_space=PowerSpace(), n_measurement_grid=60,
                            n_hypothesis_grid=80)
    table, idata = m.parameter_recovery(par, truth, rng=7, draws=50, tune=50, chains=2,
                                        backend='numpyro', target_accept=0.8,
                                        progressbar=False, random_seed=1)
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


# ---------------------------------------------------------------------------
# Censored (unfinished) responses
# ---------------------------------------------------------------------------

def _pointwise(m):
    model = m.estimation_model
    ip = model.initial_point()
    # the Deterministic is a graph of RVs: replace them by their value variables,
    # otherwise compile_fn evaluates p at random draws
    p_val = model.replace_rvs_by_values([model['p_response']])[0]
    fn = model.compile_fn([p_val, model.logp(vars=[model['response']], sum=False)[0]],
                          inputs=model.value_vars, on_unused_input='ignore')
    return fn(ip)


def test_censor_zero_equals_categorical():
    par = make_paradigm(n_subjects=2, n_trials=60)
    a = EstimationTaskModel(par, prior=TruncatedGaussianPrior(5, 24), motor_noise=True)
    a.build_estimation_model(par, hierarchical=True)
    par_c = par.assign(censor=0)
    b = EstimationTaskModel(par_c, prior=TruncatedGaussianPrior(5, 24), motor_noise=True)
    b.build_estimation_model(par_c, hierarchical=True)
    ip = a.estimation_model.initial_point()
    np.testing.assert_allclose(a.estimation_model.compile_logp()(ip),
                               b.estimation_model.compile_logp()(ip), rtol=1e-10)
    np.testing.assert_allclose(a.estimation_model.compile_dlogp()(ip),
                               b.estimation_model.compile_dlogp()(ip), rtol=1e-8)


def test_censored_trial_probability_is_tail_sum():
    par = make_paradigm(n_subjects=1, n_trials=30, seed=2).droplevel('subject')
    par['censor'] = 0
    par.iloc[0, par.columns.get_loc('censor')] = 1
    par.iloc[1, par.columns.get_loc('censor')] = -1
    m = EstimationTaskModel(par, motor_noise=True)
    m.build_estimation_model(par, hierarchical=False, save_p_choice=True)
    p, ll = _pointwise(m)
    r = (par['response'].values - 5).astype(int)
    np.testing.assert_allclose(np.exp(ll[0]), p[0, r[0]:].sum(), rtol=1e-8)
    np.testing.assert_allclose(np.exp(ll[1]), p[1, :r[1] + 1].sum(), rtol=1e-8)
    np.testing.assert_allclose(np.exp(ll[2:]), p[np.arange(2, len(r)), r[2:]], rtol=1e-8)


def test_censoring_recovers_parameters_from_unfinished_movements():
    """Simulate reports, then cut a third of them short (a slider moving from a random
    start towards the report stops early). Fitting with censoring recovers nu; treating
    the cut reports as observed does not."""
    par = make_paradigm(n_subjects=1, n_trials=1500, seed=8).droplevel('subject')
    truth = dict(nu=0.3, sigma_r=1.5, p_lapse=0.02)
    gen = EstimationTaskModel(par, motor_noise=True)
    data = gen.simulate(par, truth, rng=9).droplevel('sample').rename(
        columns={'simulated_response': 'response'})
    rng = np.random.default_rng(10)
    start = rng.integers(5, 26, len(data))
    dist = data['response'].values - start
    cut = (rng.random(len(data)) < .33) & (np.abs(dist) >= 2)
    frac = rng.uniform(.2, .8, len(data))
    final = np.round(start + frac * dist).astype(int)
    data['censor'] = 0
    data.loc[cut, 'censor'] = np.sign(dist[cut]).astype(int)
    data.loc[cut, 'response'] = final[cut]
    est = {}
    for name, d in (('censored', data), ('ignored', data.drop(columns='censor'))):
        fit = EstimationTaskModel(d, motor_noise=True)
        fit.build_estimation_model(d, hierarchical=False)
        est[name] = fit.fit_map(progressbar=False)
    assert est['censored']['nu'] == pytest.approx(truth['nu'], rel=0.12)
    assert abs(est['ignored']['nu'] - truth['nu']) > abs(est['censored']['nu'] - truth['nu'])


@pytest.mark.parametrize('motor', [True, False])
def test_measurement_grid_accuracy(motor):
    """Default grids against a fine, wide reference at strongly compressive, noisy
    parameters (where mass near the grid ends matters most)."""
    par = grid_paradigm()
    pars = dict(nu=0.5, alpha=-1.2, prior_mu=17.0, prior_sd=0.5, p_lapse=0.0)
    if motor:
        pars['sigma_r'] = 1.8
    kw = dict(encoding_space=PowerSpace(), prior=TruncatedGaussianPrior(5, 24), lapse=False,
              motor_noise=motor)
    cur = EstimationTaskModel(par, **kw)
    ref = EstimationTaskModel(par, n_measurement_grid=4000, **kw)
    ref.meas_t = np.linspace(-3, 4, 4000)
    pc = cur.predict_response_distribution(par, pars).values
    pr = ref.predict_response_distribution(par, pars).values
    assert np.abs(pc - pr).max() < 1e-3


def test_condition_offsets():
    """Two-level condition offsets: builds, finite gradient, and with zero offsets the
    response likelihood equals the model without them."""
    par = make_paradigm(n_subjects=3, n_trials=80)
    par['side'] = np.where(np.arange(len(par)) % 2, 'right', 'left')
    kw = dict(encoding_space=PowerSpace(), prior=TruncatedGaussianPrior(5, 24), motor_noise=True)
    a = EstimationTaskModel(par, **kw)
    a.build_estimation_model(par, hierarchical=True)
    b = EstimationTaskModel(par, condition='side',
                            condition_parameters=('alpha', 'nu', 'prior_mu', 'prior_sd', 'sigma_r'),
                            **kw)
    b.build_estimation_model(par, hierarchical=True)
    assert 'nu_cond_mu' in b.estimation_model.named_vars
    ipb = b.estimation_model.initial_point()
    assert np.all(np.isfinite(b.estimation_model.compile_dlogp()(ipb)))
    ipa = a.estimation_model.initial_point()

    def resp_logp(m, ip):
        mod = m.estimation_model
        return mod.compile_fn(mod.logp(vars=[mod['response']]), inputs=mod.value_vars,
                              on_unused_input='ignore')(ip)
    la, lb = resp_logp(a, ipa), resp_logp(b, ipb)
    np.testing.assert_allclose(la, lb, rtol=1e-10)


def _resp_logp_at(m, values):
    """Response log-likelihood with the given parameter values (by RV name)."""
    mod = m.estimation_model
    ip = mod.initial_point()
    for k, v in values.items():
        ip[k] = np.asarray(v, dtype=float)
    return mod.compile_fn(mod.logp(vars=[mod['response']]), inputs=mod.value_vars,
                          on_unused_input='ignore')(ip)


def test_trial_covariates_reduce_to_plain_model_and_bin_finely():
    par = make_paradigm(n_subjects=2, n_trials=150, seed=4)
    rng = np.random.default_rng(5)
    par['z'] = rng.normal(size=len(par))
    par['u'] = rng.normal(size=len(par))
    kw = dict(encoding_space=PowerSpace(), prior=TruncatedGaussianPrior(5, 24), motor_noise=True)
    base = EstimationTaskModel(par, **kw)
    base.build_estimation_model(par, hierarchical=False if False else True)
    both = EstimationTaskModel(par, measurement_shift='z', noise_modulator='u', **kw)
    both.build_estimation_model(par, hierarchical=True)
    # zero effects: identical likelihood
    zero = {'shift_beta_mu': 0.0, 'shift_beta_offset': np.zeros(2),
            'noise_gamma_mu': 0.0, 'noise_gamma_offset': np.zeros(2)}
    np.testing.assert_allclose(_resp_logp_at(base, {}), _resp_logp_at(both, zero), rtol=1e-10)
    assert np.all(np.isfinite(both.estimation_model.compile_dlogp()(
        both.estimation_model.initial_point())))
    # 7 bins vs (nearly) per-trial evaluation, at sizeable effects
    eff = {'shift_beta_mu': 0.4, 'shift_beta_offset': np.zeros(2),
           'noise_gamma_mu': 0.3, 'noise_gamma_offset': np.zeros(2)}
    fine = EstimationTaskModel(par, measurement_shift='z', noise_modulator='u',
                               n_covariate_bins=50, **kw)
    fine.build_estimation_model(par, hierarchical=True)
    l7, l50 = _resp_logp_at(both, eff), _resp_logp_at(fine, eff)
    assert abs(l7 - l50) / len(par) < 0.01            # < .01 nats per trial


def test_prior_gaussian_in_items():
    """space='n': a very wide prior equals the discrete uniform prior, and the prior (hence
    predictions at fixed noise) does not depend on alpha through the prior."""
    par = grid_paradigm()
    pars = dict(nu=0.3, alpha=0.3, prior_mu=14.0, prior_sd=500.0, p_lapse=0.0)
    m = EstimationTaskModel(par, encoding_space=PowerSpace(),
                            prior=TruncatedGaussianPrior(5, 24, space='n'), lapse=False)
    u = EstimationTaskModel(par, encoding_space=PowerSpace(),
                            prior=UniformPrior(5, 24, discrete=True), lapse=False)
    np.testing.assert_allclose(m.predict_response_distribution(par, pars).values,
                               u.predict_response_distribution(par, pars).values, atol=1e-4)
    # prior mass independent of alpha: evaluate the log-prior term directly
    import pytensor.tensor as pt
    for a in (-1.0, 0.0, 1.0):
        inp = {'nu': pt.as_tensor([.3]), 'alpha': pt.as_tensor([a]),
               'prior_mu': pt.as_tensor([12.0]), 'prior_sd': pt.as_tensor([4.0])}
        m2 = EstimationTaskModel(par, encoding_space=PowerSpace(),
                                 prior=TruncatedGaussianPrior(5, 24, space='n'), lapse=False)
        mg, nh, _, _ = [x.eval() for x in m2.estimate_on_grid(inp, 1)]
        # with very large noise the estimate is the prior mean, whatever alpha
        inp['nu'] = pt.as_tensor([50.0])
        mg, nh, _, _ = [x.eval() for x in m2.estimate_on_grid(inp, 1)]
        k = np.arange(5, 25)
        w = np.exp(-.5 * ((k - 12.0) / 4.0) ** 2)
        np.testing.assert_allclose(nh[0, len(nh[0]) // 2], (w * k).sum() / w.sum(), atol=1e-2)


def _uncued_paradigm(seed=6):
    par = make_paradigm(n_subjects=2, n_trials=100, seed=seed)
    par['n_u'] = np.random.default_rng(seed + 1).integers(5, 25, len(par))
    return par


UNCUED_KW = dict(encoding_space=PowerSpace(), prior=TruncatedGaussianPrior(5, 24, space='n'),
                 motor_noise=True)


@pytest.mark.parametrize('term,zero', [
    ('leak', {'leak_w_mu': 0.0, 'leak_w_offset': np.zeros(2)}),
    ('prior_leak', {'prior_leak_k_mu': 0.0, 'prior_leak_k_offset': np.zeros(2)}),
    ('response_pull', {'pull_k_mu': 0.0, 'pull_k_offset': np.zeros(2)}),
    ('swap', {'swap_a_mu': -60.0, 'swap_a_offset': np.zeros(2),
              'swap_b_mu': 0.0, 'swap_b_offset': np.zeros(2)}),
])
def test_uncued_terms_reduce_to_plain_model(term, zero):
    """leak / prior_leak / response_pull / swap at zero effect = the model without them."""
    par = _uncued_paradigm()
    base = EstimationTaskModel(par, **UNCUED_KW)
    base.build_estimation_model(par, hierarchical=True)
    m = EstimationTaskModel(par, **{term: 'n_u'}, **UNCUED_KW)
    m.build_estimation_model(par, hierarchical=True)
    assert np.all(np.isfinite(m.estimation_model.compile_dlogp()(
        m.estimation_model.initial_point())))
    np.testing.assert_allclose(_resp_logp_at(base, {}), _resp_logp_at(m, zero), rtol=1e-8)


def test_swap_probability_one_exchanges_roles():
    par = _uncued_paradigm(seed=8)
    m = EstimationTaskModel(par, swap='n_u', **UNCUED_KW)
    m.build_estimation_model(par, hierarchical=True)
    swapped = par.copy()
    swapped['n'], swapped['n_u'] = par['n_u'].values, par['n'].values
    base = EstimationTaskModel(swapped, **UNCUED_KW)
    base.build_estimation_model(swapped, hierarchical=True)
    one = {'swap_a_mu': 60.0, 'swap_a_offset': np.zeros(2),
           'swap_b_mu': 0.0, 'swap_b_offset': np.zeros(2)}
    np.testing.assert_allclose(_resp_logp_at(base, {}), _resp_logp_at(m, one), rtol=1e-8)


def test_pooled_lapse_equals_hierarchical_at_equal_rates():
    par = make_paradigm(n_subjects=3, n_trials=80, seed=9)
    kw = dict(encoding_space=PowerSpace(), prior=TruncatedGaussianPrior(5, 24), motor_noise=True)
    hier = EstimationTaskModel(par, **kw)
    hier.build_estimation_model(par, hierarchical=True)
    pooled = EstimationTaskModel(par, lapse='pooled', **kw)
    pooled.build_estimation_model(par, hierarchical=True)
    mod = pooled.estimation_model
    assert 'p_lapse_kappa' not in mod.named_vars and 'p_lapse_log_kappa' not in mod.named_vars
    assert np.all(np.isfinite(mod.compile_dlogp()(mod.initial_point())))
    x = np.log(.03 / .97)
    np.testing.assert_allclose(_resp_logp_at(hier, {'p_lapse_logodds__': np.full(3, x)}),
                               _resp_logp_at(pooled, {'p_lapse_mu_logodds__': x}), rtol=1e-10)
