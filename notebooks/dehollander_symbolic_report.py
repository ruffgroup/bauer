"""Dehollander 2024 symbolic risky-choice — per-fit report.

Same structure as dehollander_dotcloud_report.py but with the 58-subj
symbolic dataset (continuous n).
"""
# %%
import warnings; warnings.filterwarnings('ignore')
import os, os.path as op
import numpy as np, pandas as pd, arviz as az
import seaborn as sns, matplotlib.pyplot as plt

import sys; sys.path.insert(0, op.join(op.dirname(__file__), 'lib'))
from plotting import (
    diagnostics_summary, group_param_forest, per_subject_dots, implied_noise,
)

from bauer.utils.data import load_dehollander2024_symbolic
from bauer.models import (
    RiskModel, DDMRiskModel, RaceDiffusionRiskModel,
    FlexibleNoiseRiskModel, DDMFlexibleNoiseRiskModel,
    RaceDiffusionFlexibleNoiseRiskModel,
)

sns.set_theme(context='notebook', style='whitegrid', palette='deep')

RES = '/shares/zne.uzh/gdehol/bauer_results/dehollander_symbolic/58subj'
OUT = '/Users/gdehol/git/bauer/notebooks/figures/dehollander_symbolic_report'
os.makedirs(OUT, exist_ok=True)

df = load_dehollander2024_symbolic()
df['chose_risky'] = ((df['p2'] == 0.55) & df['choice']) | ((df['p1'] == 0.55) & ~df['choice'])
df['risky_first'] = (df['p1'] == 0.55)
df['n_safe'] = np.where(df['risky_first'], df['n2'], df['n1'])
df['log_ratio'] = np.log(np.where(df['risky_first'], df['n1'] / df['n2'], df['n2'] / df['n1']))

FITS = {
    'choice':       ('choice_full.nc',                   RiskModel,                                  dict(prior_estimate='full', fit_seperate_evidence_sd=True)),
    'choice-flex':  ('choice_flex_full.nc',              FlexibleNoiseRiskModel,                     dict(prior_estimate='full', fit_seperate_evidence_sd=True, spline_order=5)),
    'DDM':          ('ddm_freescale_full.nc',            DDMRiskModel,                               dict(prior_estimate='full', fit_seperate_evidence_sd=True, fit_v_scale=True)),
    'DDM-flex':     ('ddm_flex_freescale_full.nc',       DDMFlexibleNoiseRiskModel,                  dict(prior_estimate='full', fit_seperate_evidence_sd=True, spline_order=5, fit_v_scale=True)),
    'RDM':          ('rdm_full.nc',                      RaceDiffusionRiskModel,                     dict(prior_estimate='full', fit_seperate_evidence_sd=True, advantage=True)),
    'RDM-flex':     ('rdm_flex_full.nc',                 RaceDiffusionFlexibleNoiseRiskModel,        dict(prior_estimate='full', fit_seperate_evidence_sd=True, spline_order=5, advantage=True)),
}

idatas, models = {}, {}
for name, (nc, Cls, kw) in FITS.items():
    full = op.join(RES, nc)
    if not op.exists(full): print(f'  [skip] {name}'); continue
    idatas[name] = az.from_netcdf(full)
    models[name] = Cls(paradigm=df, **kw)
    try: models[name].build_estimation_model(data=df, hierarchical=True)
    except TypeError: models[name].build_estimation_model(paradigm=df, hierarchical=True)

diag = pd.DataFrame({k: diagnostics_summary(v) for k, v in idatas.items()}).T
print(diag.to_string())

for name, idata in idatas.items():
    fig, ax = plt.subplots(figsize=(8, 5))
    group_param_forest(idata, ax=ax)
    ax.set_title(f'Dehollander symbolic — {name}: group-level parameters')
    plt.tight_layout()
    fig.savefig(op.join(OUT, f'group_{name}.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)

COG_PARAMS = ['risky_prior_mu', 'safe_prior_mu', 'risky_prior_sd', 'safe_prior_sd',
              'n1_evidence_sd', 'n2_evidence_sd']
for name, idata in idatas.items():
    g = per_subject_dots(idata, COG_PARAMS)
    if g is None: continue
    g.fig.suptitle(f'Dehollander symbolic — {name}: per-subject posteriors (95% CI)', y=1.02)
    g.savefig(op.join(OUT, f'subjects_{name}.png'), dpi=140, bbox_inches='tight')
    plt.close(g.fig)

for name in ['choice-flex', 'DDM-flex', 'RDM-flex']:
    if name not in idatas: continue
    fig, ax = plt.subplots(figsize=(8, 5))
    for var, color in [('n1_evidence_sd', 'C0'), ('n2_evidence_sd', 'C1')]:
        s = implied_noise(models[name], idatas[name], variable=var)
        ax.fill_between(s['n'], s['lo'], s['hi'], alpha=0.2, color=color)
        ax.plot(s['n'], s['mean'], color=color, lw=2,
                 label={'n1_evidence_sd': 'σ₁(n)', 'n2_evidence_sd': 'σ₂(n)'}[var])
    ax.set_xlabel('n'); ax.set_ylabel('σ_k(n)')
    ax.set_title(f'Dehollander symbolic — {name}: implied encoding noise (94% HDI)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(op.join(OUT, f'sigma_{name}.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)

print(f'figures saved under {OUT}')
