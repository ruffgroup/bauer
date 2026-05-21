"""TMS risky-choice — per-fit report (35 subjects, sessions 2/3).

Loads production fits from /shares/zne.uzh/gdehol/bauer_results/tms/35subj/
and produces:

  1. Diagnostics
  2. Group-level forest + per-subject dots (FacetGrid by parameter)
  3. PPCs (paper Fig 1B/1C style with stim_condition as additional facet)
  4. Implied σ_k(n) for flex models (with shaded CI bands)
  5. **TMS regression**: implied σ_k(n) PER stimulation_condition
     — the actual TMS-on-noise effect
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

from bauer.utils.data import load_dehollander_tms_risk
from bauer.models import (
    RiskModel, DDMRiskModel, RaceDiffusionRiskModel,
    FlexibleNoiseRiskModel, DDMFlexibleNoiseRiskModel,
    RaceDiffusionFlexibleNoiseRiskModel,
    RiskRegressionModel, FlexibleNoiseRiskRegressionModel,
    DDMFlexibleNoiseRiskRegressionModel,
    RaceDiffusionFlexibleNoiseRiskRegressionModel,
)

sns.set_theme(context='notebook', style='whitegrid', palette='deep')

RES = '/shares/zne.uzh/gdehol/bauer_results/tms/35subj'
OUT = '/Users/gdehol/git/bauer/notebooks/figures/tms_report'
os.makedirs(OUT, exist_ok=True)

# %% Data
df = load_dehollander_tms_risk()
df['chose_risky'] = ((df['p2'] == 0.55) & df['choice']) | ((df['p1'] == 0.55) & ~df['choice'])
df['risky_first'] = (df['p1'] == 0.55)
df['n_safe'] = np.where(df['risky_first'], df['n2'], df['n1'])
df['log_ratio'] = np.log(np.where(df['risky_first'], df['n1'] / df['n2'], df['n2'] / df['n1']))
print(f'{len(df)} trials, {df.index.get_level_values("subject").nunique()} subjects, '
      f'stim conditions: {sorted(df["stimulation_condition"].unique())}')

# Patsy needs stimulation_condition as a column for regression fits
df_reg = df.reset_index().set_index('subject')

# %% Fits
NOISE_REG = {'n1_evidence_sd': 'stimulation_condition',
             'n2_evidence_sd': 'stimulation_condition'}

FITS = {
    # baseline (no regression)
    'choice':       ('choice_full.nc',                      RiskModel,                                  df,      dict(prior_estimate='full', fit_seperate_evidence_sd=True)),
    'choice-flex':  ('choice_flex_full.nc',                 FlexibleNoiseRiskModel,                     df,      dict(prior_estimate='full', fit_seperate_evidence_sd=True, spline_order=5)),
    'DDM':          ('ddm_freescale_full.nc',               DDMRiskModel,                               df,      dict(prior_estimate='full', fit_seperate_evidence_sd=True, fit_v_scale=True)),
    'DDM-flex':     ('ddm_flex_freescale_full.nc',          DDMFlexibleNoiseRiskModel,                  df,      dict(prior_estimate='full', fit_seperate_evidence_sd=True, spline_order=5, fit_v_scale=True)),
    'RDM':          ('rdm_full.nc',                         RaceDiffusionRiskModel,                     df,      dict(prior_estimate='full', fit_seperate_evidence_sd=True, advantage=True)),
    'RDM-flex':     ('rdm_flex_full.nc',                    RaceDiffusionFlexibleNoiseRiskModel,        df,      dict(prior_estimate='full', fit_seperate_evidence_sd=True, spline_order=5, advantage=True)),
    # regression on stimulation_condition (noise-only)
    'choice-reg':       ('choice_reg_full.nc',                  RiskRegressionModel,                                df_reg, dict(regressors=NOISE_REG, prior_estimate='full', fit_seperate_evidence_sd=True)),
    'choice-flex-reg':  ('choice_flex_reg_full.nc',             FlexibleNoiseRiskRegressionModel,                   df_reg, dict(regressors=NOISE_REG, prior_estimate='full', fit_seperate_evidence_sd=True, spline_order=5)),
    'DDM-flex-reg':     ('ddm_flex_reg_freescale_full.nc',      DDMFlexibleNoiseRiskRegressionModel,                df_reg, dict(regressors=NOISE_REG, prior_estimate='full', fit_seperate_evidence_sd=True, spline_order=5, fit_v_scale=True)),
    'RDM-flex-reg':     ('rdm_flex_reg_full.nc',                RaceDiffusionFlexibleNoiseRiskRegressionModel,      df_reg, dict(regressors=NOISE_REG, prior_estimate='full', fit_seperate_evidence_sd=True, spline_order=5, advantage=True)),
}

idatas, models = {}, {}
for name, (nc, Cls, df_use, kw) in FITS.items():
    full = op.join(RES, nc)
    if not op.exists(full): print(f'  [skip] {name}'); continue
    idatas[name] = az.from_netcdf(full)
    models[name] = Cls(paradigm=df_use, **kw)
    try: models[name].build_estimation_model(data=df_use, hierarchical=True)
    except TypeError: models[name].build_estimation_model(paradigm=df_use, hierarchical=True)

# %% Diagnostics
diag = pd.DataFrame({k: diagnostics_summary(v) for k, v in idatas.items()}).T
print(diag.to_string())

# %% Group + per-subject (cog params only; regression coefs handled separately)
COG_PARAMS = ['risky_prior_mu', 'safe_prior_mu', 'risky_prior_sd', 'safe_prior_sd',
              'n1_evidence_sd', 'n2_evidence_sd']

for name, idata in idatas.items():
    fig, ax = plt.subplots(figsize=(8, 5))
    group_param_forest(idata, ax=ax)
    ax.set_title(f'TMS — {name}: group-level parameters')
    plt.tight_layout()
    fig.savefig(op.join(OUT, f'group_{name}.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)

    g = per_subject_dots(idata, COG_PARAMS)
    if g is not None:
        g.fig.suptitle(f'TMS — {name}: per-subject posteriors (95% CI)', y=1.02)
        g.savefig(op.join(OUT, f'subjects_{name}.png'), dpi=140, bbox_inches='tight')
        plt.close(g.fig)

# %% Implied σ_k(n) for flex models (single condition pooled)
for name in ['choice-flex', 'DDM-flex', 'RDM-flex']:
    if name not in idatas: continue
    fig, ax = plt.subplots(figsize=(8, 5))
    for var, color in [('n1_evidence_sd', 'C0'), ('n2_evidence_sd', 'C1')]:
        s = implied_noise(models[name], idatas[name], variable=var)
        ax.fill_between(s['n'], s['lo'], s['hi'], alpha=0.2, color=color)
        ax.plot(s['n'], s['mean'], color=color, lw=2,
                 label={'n1_evidence_sd': 'σ₁(n)', 'n2_evidence_sd': 'σ₂(n)'}[var])
    ax.set_xlabel('n'); ax.set_ylabel('σ_k(n)')
    ax.set_title(f'TMS — {name}: implied encoding noise (pooled across conditions, 94% HDI)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(op.join(OUT, f'sigma_{name}.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)

# %% TMS regression: σ_k(n) PER stimulation_condition
# For each *_reg fit, evaluate the spline at each condition's design row.
# This is the headline plot for the TMS analysis.
#
# TODO: implement once a regression fit lands. Steps:
#   1. Build the patsy design matrix for each unique stimulation_condition value
#      using model.regression_dms[<param>] (or rebuild via patsy.build_design_matrices)
#   2. Combine with spline coefficient posterior to get σ_k(n | condition)
#   3. Plot two curves with shaded HDIs, one per condition.

print(f'figures saved under {OUT}')
