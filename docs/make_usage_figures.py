"""Generate the example figures of docs/usage.rst ("Summarising and plotting posteriors").

Simulates a small within-subject dose experiment, fits it with
``PsychophysicalRegressionModel`` and saves one PNG per plotting helper to
``docs/_static/usage/``. The analysis code mirrors the snippets in usage.rst, so a
change there should be made here too. Takes a few minutes on a laptop CPU::

    python docs/make_usage_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from bauer.models import PsychophysicalRegressionModel
from bauer.utils import (contrast_draws, convergence_report, plot_forest, plot_ppc_overlay,
                         plot_ppc_panels, plot_sorted_forest, plot_subject_ppc, ppc_coverage,
                         ppc_observed, ppc_summary, set_house_style, summarize_posterior)

OUT_DIR = Path(__file__).parent / '_static' / 'usage'
SEED = 20261118

CONDITIONS = ['placebo', '5 mg', '10 mg']
PALETTE = {'placebo': '0.4', '5 mg': '#3B5BA5', '10 mg': '#C0392B'}
COND = dict(order=CONDITIONS, palette=PALETTE, level_labels={'placebo': 'Placebo'})

# Generative group means per condition
TRUE_NU = {'placebo': 0.8, '5 mg': 1.0, '10 mg': 1.3}
TRUE_BIAS = {'placebo': 0.0, '5 mg': 0.15, '10 mg': 0.4}
SESSION_EFFECT_NU = -0.08       # practice: noise drops from session 1 to session 2
DELTAS = [-4., -2.5, -1.25, -0.5, 0.5, 1.25, 2.5, 4.]


def simulate_dataset(n_subjects=24, n_repeats=8, seed=SEED):
    """Every subject does every condition in each of two sessions."""
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(1, n_subjects + 1):
        nu_scale = np.exp(rng.normal(0, 0.2))
        bias_offset = rng.normal(0, 0.15)
        dose_gain = np.exp(rng.normal(0, 0.3))      # how strongly this subject responds
        for session_c in (-0.5, 0.5):
            for condition in CONDITIONS:
                gain_nu = dose_gain * (TRUE_NU[condition] - TRUE_NU['placebo'])
                nu = nu_scale * (TRUE_NU['placebo'] + gain_nu + SESSION_EFFECT_NU * session_c)
                bias = bias_offset + dose_gain * TRUE_BIAS[condition]
                for delta in DELTAS:
                    p = norm.cdf((delta + bias) / (np.sqrt(2) * nu))
                    for _ in range(n_repeats):
                        rows.append(dict(subject=subject, condition=condition,
                                         session_c=session_c, delta=delta,
                                         x1=-delta / 2, x2=delta / 2,
                                         choice=bool(rng.random() < p)))
    df = pd.DataFrame(rows)
    df['trial'] = df.groupby('subject').cumcount()
    return df.set_index(['subject', 'trial'])


def save(fig, name):
    fig.savefig(OUT_DIR / f'{name}.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Wrote {OUT_DIR / name}.png')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_house_style()
    df = simulate_dataset()

    F = "C(condition, Treatment('placebo')) + session_c"
    model = PsychophysicalRegressionModel(paradigm=df.reset_index(),
                                          fixed_regressors={'nu': F, 'bias': F},
                                          random_regressors={'nu': F, 'bias': F})
    model.build_estimation_model(data=df, hierarchical=True)
    idata = model.sample(draws=1000, tune=1000, chains=4, target_accept=0.9,
                         backend='numpyro', random_seed=SEED)
    diag = convergence_report(idata)
    if not diag.attrs['ok']:
        raise RuntimeError('The example fit did not converge; not writing figures.')

    # Group level
    conds = pd.DataFrame({'condition': CONDITIONS, 'session_c': 0.0})
    draws = model.conditionwise_draws(idata, conds, parameters=['nu', 'bias'])
    group = summarize_posterior(draws, ['parameter', 'condition'])
    save(plot_forest(group, ['nu', 'bias'], xlabels={'nu': 'Noise ν', 'bias': 'Bias'}, **COND),
         'forest_group')

    effects = summarize_posterior(contrast_draws(draws, 'condition', 'placebo'),
                                  ['parameter', 'condition'])
    save(plot_forest(effects, ['nu', 'bias'], ref=0,
                     xlabels={'nu': 'Δ noise ν (vs. placebo)', 'bias': 'Δ bias (vs. placebo)'},
                     **COND),
         'forest_contrast')

    # Subject level
    subj = summarize_posterior(
        model.conditionwise_draws(idata, conds, parameters=['nu'], group=False),
        ['parameter', 'subject', 'condition'])
    save(plot_sorted_forest(subj, 'nu', 'Noise ν', sort_by='placebo', group_summary=group, **COND),
         'forest_subjects')

    # Posterior predictive checks
    ppc = model.ppc(df, idata, n_posterior_samples=200, random_seed=SEED)
    ppc = ppc.join(df[['condition', 'delta']]).reset_index()
    ppc['y'] = ppc['simulated_choice'].astype(float)
    data = df.reset_index().assign(y=lambda d: d['choice'].astype(float))
    axes = dict(xlabel='Stimulus difference (x2 − x1)', ylabel='P(chose option 2)',
                ref=0.5, xref=0)

    pred = ppc_summary(ppc, ['condition', 'delta'], 'y')
    obs = ppc_observed(data, ['condition', 'delta'], 'y')
    print(f"Group coverage: {ppc_coverage(pred, obs, ['condition', 'delta']):.0%}")
    save(plot_ppc_panels(pred, obs, 'delta', **COND, **axes), 'ppc_panels')
    save(plot_ppc_overlay(pred, obs, 'delta', **COND, **axes), 'ppc_overlay')

    pred_s = ppc_summary(ppc, ['delta'], 'y', subject_level=True)
    obs_s = ppc_observed(data, ['delta'], 'y', subject_level=True)
    print(f"Subject coverage: {ppc_coverage(pred_s, obs_s, ['subject', 'delta']):.0%}")
    save(plot_subject_ppc(pred_s, obs_s, 'delta', **axes), 'ppc_subjects')


if __name__ == '__main__':
    main()
