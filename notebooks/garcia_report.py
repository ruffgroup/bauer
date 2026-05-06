"""Garcia 2022 magnitude-comparison — per-fit report.

Loads the production fits from /shares/zne.uzh/gdehol/bauer_results/garcia/64subj/
and produces:

  1. Diagnostics table (max r̂, divergences, min ESS) per fit
  2. Group-level forest plot (one figure)
  3. Per-subject parameter dots — sns.FacetGrid by parameter
  4. PPCs: psychometric (P(choose 2)) + chronometric (RT) by stake (n1) and
     |log(n2/n1)|; data points, PPC = line + shaded HDI band
  5. Flex models: implied σ_k(n) curves with 94% HDI bands
  6. Spline-order sweep: LOO comparison across spline_order ∈ {3,5,7,9,11,13}

This is the source-of-truth script; convert to ipynb at the end via
`jupytext --to ipynb garcia_report.py`.
"""
# %%
import warnings; warnings.filterwarnings('ignore')
import os, os.path as op
import numpy as np
import pandas as pd
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, op.join(op.dirname(__file__), 'lib'))
from plotting import (
    diagnostics_summary, group_param_forest, per_subject_dots, implied_noise,
)

from bauer.utils.data import load_garcia2022
from bauer.models import (
    MagnitudeComparisonModel, DDMMagnitudeComparisonModel,
    RaceDiffusionMagnitudeComparisonModel,
    FlexibleNoiseComparisonModel, DDMFlexibleNoiseComparisonModel,
    RaceDiffusionFlexibleNoiseComparisonModel,
)

sns.set_theme(context='notebook', style='whitegrid', palette='deep')

RES = '/shares/zne.uzh/gdehol/bauer_results/garcia/64subj'
SWEEP = '/shares/zne.uzh/gdehol/bauer_results/garcia_spline_sweep'
OUT = '/Users/gdehol/git/bauer/notebooks/figures/garcia_report'
os.makedirs(OUT, exist_ok=True)

# %% [markdown]
# # Garcia 2022 — magnitude comparison report
#
# Models fit:
# - **Choice** (RiskModel-style cognitive front-end, Bernoulli choice)
# - **DDM** (HSSM Wiener WFPT)
# - **RDM** (race-diffusion, advantage decomposition)
# - Plus **flex** variants with B-spline noise σ_k(n)
#
# All hierarchical with 64 subjects, fit on cluster L4 GPU via numpyro.

# %%
df = load_garcia2022(task='magnitude')
print(f'{len(df)} trials, {df.index.get_level_values("subject").nunique()} subjects')

# %% Load all fits
FITS = {
    'choice':       (op.join(RES, 'choice.nc'),                   MagnitudeComparisonModel,                   dict(fit_seperate_evidence_sd=True, fit_prior=True)),
    'DDM':          (op.join(RES, 'ddm_freescale.nc'),            DDMMagnitudeComparisonModel,                dict(fit_seperate_evidence_sd=True, fit_prior=True, fit_v_scale=True)),
    'RDM':          (op.join(RES, 'rdm.nc'),                      RaceDiffusionMagnitudeComparisonModel,      dict(fit_seperate_evidence_sd=True, fit_prior=True, advantage=True)),
    'choice-flex':  (op.join(RES, 'choice_flex.nc'),              FlexibleNoiseComparisonModel,               dict(fit_seperate_evidence_sd=True, fit_prior=True, spline_order=5)),
    'DDM-flex':     (op.join(RES, 'ddm_flex_fixedscale.nc'),      DDMFlexibleNoiseComparisonModel,            dict(fit_seperate_evidence_sd=True, fit_prior=True, spline_order=5, fit_v_scale=False)),
    'RDM-flex':     (op.join(RES, 'rdm_flex.nc'),                 RaceDiffusionFlexibleNoiseComparisonModel,  dict(fit_seperate_evidence_sd=True, fit_prior=True, spline_order=5, advantage=True)),
}

idatas, models = {}, {}
for name, (nc, Cls, kw) in FITS.items():
    if not op.exists(nc):
        print(f'  [skip] {name}: {nc} missing'); continue
    idatas[name] = az.from_netcdf(nc)
    models[name] = Cls(paradigm=df, **kw)
    try:
        models[name].build_estimation_model(data=df, hierarchical=True)
    except TypeError:
        models[name].build_estimation_model(paradigm=df, hierarchical=True)

# %% Diagnostics table
diag = pd.DataFrame({k: diagnostics_summary(v) for k, v in idatas.items()}).T
diag = diag[['max_rhat', 'min_ess_bulk', 'min_ess_tail', 'divergences', 'n_samples']]
print(diag.to_string())

# %% Group-level forest (one figure per fit)
for name, idata in idatas.items():
    fig, ax = plt.subplots(figsize=(8, 5))
    group_param_forest(idata, ax=ax)
    ax.set_title(f'Garcia — {name}: group-level parameters')
    plt.tight_layout()
    fig.savefig(op.join(OUT, f'group_{name}.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)

# %% Per-subject dots (FacetGrid per parameter)
COG_PARAMS = ['prior_mu', 'prior_sd', 'n1_evidence_sd', 'n2_evidence_sd']
for name, idata in idatas.items():
    g = per_subject_dots(idata, COG_PARAMS)
    if g is None: continue
    g.fig.suptitle(f'Garcia — {name}: per-subject posteriors (95% CI)', y=1.02)
    g.savefig(op.join(OUT, f'subjects_{name}.png'), dpi=140, bbox_inches='tight')
    plt.close(g.fig)

# %% Implied σ_k(n) for flex models
for name in ['choice-flex', 'DDM-flex', 'RDM-flex']:
    if name not in idatas: continue
    fig, ax = plt.subplots(figsize=(8, 5))
    for var, color, ls in [('n1_evidence_sd', 'C0', '-'), ('n2_evidence_sd', 'C1', '-')]:
        s = implied_noise(models[name], idatas[name], variable=var)
        ax.fill_between(s['n'], s['lo'], s['hi'], alpha=0.2, color=color)
        ax.plot(s['n'], s['mean'], color=color, lw=2, ls=ls,
                 label={'n1_evidence_sd': 'σ₁(n)', 'n2_evidence_sd': 'σ₂(n)'}[var])
    ax.set_xlabel('n'); ax.set_ylabel('σ_k(n)')
    ax.set_title(f'Garcia — {name}: implied encoding noise (94% HDI)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(op.join(OUT, f'sigma_{name}.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)

# %% Spline-order sweep on RDM-flex (Garcia)
sweep_results = []
for so in [3, 5, 7, 9, 11, 13]:
    nc = op.join(SWEEP, f'so{so}', '64subj', 'rdm_flex.nc') if so != 5 \
         else op.join(RES, 'rdm_flex.nc')
    if not op.exists(nc):
        print(f'[skip] spline_order={so}: {nc} missing'); continue
    idata = az.from_netcdf(nc)
    diag_sweep = diagnostics_summary(idata)
    sweep_results.append({'spline_order': so, **diag_sweep})

sweep_df = pd.DataFrame(sweep_results)
print(sweep_df.to_string())
sweep_df.to_csv(op.join(OUT, 'spline_sweep_diagnostics.csv'), index=False)

# %% PPC plots — TODO: psychometric + chronometric per dataset
#    Will reuse the existing all-models PPC pattern from
#    /tmp/garcia_8subj_all_models_ppc.py, ported to seaborn.

print(f'figures saved under {OUT}')
