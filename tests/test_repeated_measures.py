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
                         plot_subject_ppc, convergence_report)

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
    plot_sorted_forest(summarize_posterior(s[s.condition == 'placebo'], ['parameter', 'subject']),
                       'nu', by=None)
    assert fig is not None and fig2 is not None


def _reproduction(n_sub=6, nu=0.25, prior_shift=0.15, seed=0, reps=30):
    rng = np.random.default_rng(seed)
    x = np.repeat(np.geomspace(0.5, 2.0, 7), reps)
    rows = []
    for s in range(1, n_sub + 1):
        log_x = np.log(x)
        mu_p, sd_p = log_x.mean() + prior_shift, log_x.std()
        m = log_x + nu * rng.standard_normal(len(x))
        w = sd_p ** 2 / (sd_p ** 2 + nu ** 2)
        rows.append(pd.DataFrame({'subject': s, 'stimulus': x,
                                  'response': np.exp(w * m + (1 - w) * mu_p)}))
    d = pd.concat(rows)
    d['trial'] = d.groupby('subject').cumcount()
    return d.set_index(['subject', 'trial'])


def test_reproduction_map_recovers_nu_and_prior_mu():
    df = _reproduction()
    m = ReproductionModel(df, fit_lapse=False)
    m.build_estimation_model(df, hierarchical=True)
    pars = m.fit_map(progressbar=False)
    presented = np.log(np.geomspace(0.5, 2.0, 7)).mean()
    assert abs(pars['nu'].mean() - 0.25) < 0.04
    assert abs(pars['prior_mu'].mean() - (presented + 0.15)) < 0.06


def test_reproduction_bias_and_prior_mu_exclusive():
    with pytest.raises(ValueError, match='intercept'):
        ReproductionModel(_reproduction(n_sub=1, reps=2), fit_bias=True, fit_prior_mu=True)


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


# ─── Misuse fails loudly ─────────────────────────────────────────────────────

def _fake_draws(n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for par in ['nu', 'bias']:
        for cond, mu in zip(CONDS, [1.0, 1.0, 1.2, 1.4]):
            rows.append(pd.DataFrame({'parameter': par, 'chain': 0, 'draw': np.arange(n),
                                      'condition': cond, 'session_c': 0.0,
                                      'value': mu + 0.1 * rng.standard_normal(n)}))
    return pd.concat(rows, ignore_index=True)


def test_contrast_draws_checks():
    d = _fake_draws()
    out = contrast_draws(d, 'condition', 'placebo')     # constant session_c is fine
    assert list(pd.unique(out['condition'])) == ['baseline', '5 mg', '10 mg']
    with pytest.raises(ValueError, match='not a level'):
        contrast_draws(d, 'condition', 'Placebo')
    varying = d.assign(session_c=np.where(d['condition'] == 'placebo', 0.0, 1.0))
    with pytest.raises(ValueError, match='vary between rows'):
        contrast_draws(varying, 'condition', 'placebo')
    with pytest.raises(ValueError, match='Several rows'):
        contrast_draws(pd.concat([d, d]), 'condition', 'placebo')


def test_summarize_posterior_checks():
    d = _fake_draws()
    with pytest.raises(ValueError, match='NaN values'):
        summarize_posterior(d.assign(value=np.where(d.index == 0, np.nan, d['value'])),
                            ['parameter', 'condition'])
    with pytest.warns(UserWarning, match='pooling'):
        summarize_posterior(d, ['parameter'])
    with pytest.warns(UserWarning, match='Only 5 draws'):
        summarize_posterior(d[d['draw'] < 5], ['parameter', 'condition'])


def test_ppc_coverage_checks():
    pred = pd.DataFrame({'condition': ['a', 'b'], 'x': [1, 1], 'pred': [.5, .5],
                         'lo': [.4, .4], 'hi': [.6, .6]})
    obs = pd.DataFrame({'condition': ['a', 'c'], 'x': [1, 1], 'obs': [.5, .5]})
    with pytest.raises(ValueError, match='only one of'):
        ppc_coverage(pred, obs, ['condition', 'x'])
    with pytest.raises(ValueError, match='several rows'):
        ppc_coverage(pred, obs.assign(condition='a'), ['x'])


def test_plot_input_checks():
    s = summarize_posterior(_fake_draws(), ['parameter', 'condition'])
    with pytest.raises(ValueError, match='not in order'):
        plot_forest(s, ['nu'], order=['Placebo', 'baseline', '5 mg', '10 mg'])
    with pytest.raises(ValueError, match="no rows for parameter 'Nu'"):
        plot_forest(s, ['Nu'])
    subj = pd.concat([s.assign(subject=i) for i in range(3)])
    with pytest.raises(ValueError, match='several rows per'):
        plot_forest(subj, ['nu'])
    with pytest.raises(ValueError, match='sort_by'):
        plot_sorted_forest(subj, 'nu', sort_by='Placebo')


def test_random_slope_on_session_warns_and_conditionwise_checks():
    df = _psychophysics()
    with pytest.warns(UserWarning, match='not identified within any subject'):
        m = PsychophysicalLapseRegressionModel(df.reset_index(), fixed_regressors={'nu': F},
                                               random_regressors={'nu': F})
        m.build_estimation_model(df, hierarchical=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        m = PsychophysicalLapseRegressionModel(
            df.reset_index(), fixed_regressors={'nu': F},
            random_regressors={'nu': "C(condition, Treatment('placebo'))"})
        m.build_estimation_model(df, hierarchical=True)
    idata = m.sample(draws=10, tune=10, chains=1, progressbar=False, random_seed=1)
    conds = pd.DataFrame({'condition': CONDS, 'session_c': 0.0})
    with pytest.raises(ValueError, match='Unknown parameter'):
        m.conditionwise_draws(idata, conds, parameters=['Nu'])
    with pytest.raises(ValueError, match='duplicate rows'):
        m.conditionwise_draws(idata, pd.concat([conds, conds]), parameters='nu')


def test_convergence_report_skips_subject_level_sd_and_mu():
    # Subject-level parameters whose names end in '_sd' / '_mu' (n1_evidence_sd,
    # prior_mu) must not be swept into the default group-level check.
    import arviz as az
    rng = np.random.default_rng(0)
    post = {'n1_evidence_sd': rng.normal(size=(2, 100, 5)),
            'prior_mu': rng.normal(size=(2, 100, 5)),
            'n1_evidence_sd_mu': rng.normal(size=(2, 100)),
            'n1_evidence_sd_sd': rng.normal(size=(2, 100))}
    idata = az.from_dict(posterior=post,
                         dims={'n1_evidence_sd': ['subject'], 'prior_mu': ['subject']})
    table = convergence_report(idata, verbose=False)
    assert sorted(table.index) == ['n1_evidence_sd_mu', 'n1_evidence_sd_sd']
