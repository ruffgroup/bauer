"""Within-subject condition × session designs: lapse-regression prior, rank
check, long conditionwise draws, the reproduction model and the summary/plot
helpers."""
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from bauer.models import (PsychophysicalLapseRegressionModel, ReproductionModel,  # noqa: E402
                          ReproductionRegressionModel, MagnitudeComparisonLapseRegressionModel)
from bauer.utils import (summarize_posterior, contrast_draws, ppc_summary, ppc_observed,  # noqa: E402
                         ppc_coverage, plot_forest, plot_sorted_forest, plot_ppc_panels,
                         plot_subject_ppc)

CONDS = ['baseline', 'placebo', '5 mg', '10 mg']


def _design(n_sub=4, baseline_first=True):
    rows = []
    for s in range(1, n_sub + 1):
        order = ['baseline'] + list(np.roll(CONDS[1:], s)) if baseline_first else list(np.roll(CONDS, s))
        for sess, c in enumerate(order, start=1):
            rows.append((s, sess, c))
    d = pd.DataFrame(rows, columns=['subject', 'session', 'condition'])
    d['session_c'] = d['session'] - 2.5
    return d


def _psychophysics(n_sub=4, n=40, seed=0):
    rng = np.random.default_rng(seed)
    d = _design(n_sub).merge(pd.DataFrame({'x2': np.linspace(-3, 3, n)}), how='cross')
    d['x1'] = 0.0
    d['choice'] = rng.random(len(d)) < 1 / (1 + np.exp(-d['x2']))
    d['trial'] = d.groupby('subject').cumcount()
    return d.set_index(['subject', 'trial'])


F = "C(condition, Treatment('placebo')) + session_c"


@pytest.mark.parametrize('cls', [PsychophysicalLapseRegressionModel,
                                 MagnitudeComparisonLapseRegressionModel])
def test_lapse_regression_lapse_is_bounded(cls):
    df = _psychophysics()
    if cls is MagnitudeComparisonLapseRegressionModel:
        df = df.assign(n1=10.0, n2=np.exp(df['x2'] / 3) * 10)
        regs = {'n1_evidence_sd': F}
    else:
        regs = {'nu': F}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = cls(df.reset_index(), regressors=regs)
    assert m.free_parameters['p_lapse']['transform'] == 'logistic'
    m.build_estimation_model(df, hierarchical=True)
    # Push the untransformed lapse far out: with the transform the lapse stays
    # in (0, 1) and the log-probability is finite; the old bug gave lapse > 1.
    ip = m.estimation_model.initial_point()
    ip['p_lapse_mu'] = ip['p_lapse_mu'] + 4.0
    logp = m.estimation_model.compile_logp()(ip)
    assert np.isfinite(logp)


def test_rank_deficient_design_raises():
    df = _psychophysics()
    m = PsychophysicalLapseRegressionModel(
        df.reset_index(), fixed_regressors={'nu': "C(condition) + C(session)"})
    with pytest.raises(ValueError, match='rank-deficient'):
        m.build_estimation_model(df, hierarchical=True)


def test_conditionwise_draws_long_format():
    df = _psychophysics()
    m = PsychophysicalLapseRegressionModel(df.reset_index(), fixed_regressors={'nu': F},
                                           random_regressors={'nu': F})
    m.build_estimation_model(df, hierarchical=True)
    idata = m.sample(draws=20, tune=20, chains=1, progressbar=False, random_seed=1)
    conds = pd.DataFrame({'condition': CONDS, 'session_c': 0.0})
    g = m.conditionwise_draws(idata, conds, parameters=['nu'])
    assert set(g.columns) >= {'parameter', 'chain', 'draw', 'condition', 'session_c', 'value'}
    assert len(g) == 20 * len(CONDS)
    s = m.conditionwise_draws(idata, conds, parameters=['nu'], group=False)
    assert len(s) == 20 * len(CONDS) * 4
    summ = summarize_posterior(g, ['parameter', 'condition'])
    assert {'mean', 'lo', 'hi', 'p_positive'} <= set(summ.columns)
    con = contrast_draws(g.drop(columns='session_c'), 'condition', 'placebo')
    assert set(con['condition']) == {'baseline', '5 mg', '10 mg'}
    fig = plot_forest(summ, ['nu'], order=CONDS)
    fig2 = plot_sorted_forest(summarize_posterior(s, ['parameter', 'subject', 'condition']),
                              'nu', order=CONDS, sort_by='placebo', group_summary=summ)
    assert fig is not None and fig2 is not None


def _reproduction(n_sub=6, nu=0.25, bias=0.08, seed=0, reps=30):
    rng = np.random.default_rng(seed)
    x = np.repeat(np.geomspace(0.5, 2.0, 7), reps)
    rows = []
    for s in range(1, n_sub + 1):
        log_x = np.log(x)
        mu_p, sd_p = log_x.mean(), log_x.std()
        m = log_x + bias + nu * rng.standard_normal(len(x))
        w = sd_p ** 2 / (sd_p ** 2 + nu ** 2)
        rows.append(pd.DataFrame({'subject': s, 'stimulus': x,
                                  'response': np.exp(w * m + (1 - w) * mu_p)}))
    d = pd.concat(rows)
    d['trial'] = d.groupby('subject').cumcount()
    return d.set_index(['subject', 'trial'])


def test_reproduction_map_recovers_nu_and_bias():
    df = _reproduction()
    m = ReproductionModel(df, fit_lapse=False)
    m.build_estimation_model(df, hierarchical=True)
    pars = m.fit_map(progressbar=False)
    assert abs(pars['nu'].mean() - 0.25) < 0.04
    assert abs(pars['bias'].mean() - 0.08) < 0.03


def test_reproduction_regression_ppc_and_plots():
    df = _reproduction(n_sub=4, reps=8).reset_index()
    df['condition'] = np.where(df['trial'] % 2, 'placebo', '10 mg')
    df = df.set_index(['subject', 'trial'])
    m = ReproductionRegressionModel(df.reset_index(), fixed_regressors={'nu': 'C(condition)'},
                                    response_range=(0.25, 4.0))
    m.build_estimation_model(df, hierarchical=True)
    idata = m.sample(draws=20, tune=20, chains=1, progressbar=False, random_seed=2)
    ppc = m.ppc(df, idata, n_posterior_samples=10, random_seed=3)
    assert list(ppc.columns) == ['simulated_response']
    sim = ppc.join(df[['condition', 'stimulus']]).reset_index()
    pred = ppc_summary(sim, ['condition', 'stimulus'], 'simulated_response')
    obs = ppc_observed(df.reset_index(), ['condition', 'stimulus'], 'response')
    assert 0 <= ppc_coverage(pred, obs, ['condition', 'stimulus']) <= 1
    plot_ppc_panels(pred, obs, 'stimulus', logx=True, xticks=[0.5, 1, 2])
    plot_subject_ppc(ppc_summary(sim, ['stimulus'], 'simulated_response', subject_level=True),
                     ppc_observed(df.reset_index(), ['stimulus'], 'response', subject_level=True),
                     'stimulus', col_wrap=4)


def test_reproduction_fits_prior_width_and_noise():
    df = _reproduction(n_sub=6, nu=0.25, reps=40)
    m = ReproductionModel(df, fit_lapse=False, fit_prior_sd=True)
    assert 'prior_sd' in m.free_parameters
    m.build_estimation_model(df, hierarchical=True)
    pars = m.fit_map(progressbar=False)
    true_sd = np.log(np.geomspace(0.5, 2.0, 7)).std()
    assert abs(pars['nu'].mean() - 0.25) < 0.05
    assert abs(pars['prior_sd'].mean() - true_sd) < 0.1
