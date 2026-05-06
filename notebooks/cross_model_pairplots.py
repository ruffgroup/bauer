"""Cross-model parameter pairplots for any dataset.

For each cognitive parameter, plots an N×N grid where N = number of models
that have that parameter:
  - Diagonal: per-subject posterior mean ± 95% CI (one row per subject,
    sorted by mean — same view as `notebooks/figures/early_garcia/`).
  - Off-diagonal: scatter of model[col] vs model[row], errorbars in both
    axes, y=x dashed reference. Cells with a tight band on the diagonal =
    models agree on that parameter.

Skips fits that don't exist on disk OR that fail diagnostics
(max r̂ > 1.05 or min ESS < 50/chain). The skip filter keeps the plots
honest — we don't want non-converged fits dragging the axis ranges.

Usage:
    python notebooks/cross_model_pairplots.py garcia
    python notebooks/cross_model_pairplots.py dehollander_dotcloud
    python notebooks/cross_model_pairplots.py tms
"""
import warnings; warnings.filterwarnings('ignore')
import os, os.path as op, sys, argparse
import numpy as np, pandas as pd, arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, op.join(op.dirname(__file__), 'lib'))
from plotting import diagnostics_summary

sns.set_theme(context='notebook', style='whitegrid', palette='deep')

ROOT = '/Users/gdehol/git/bauer/notebooks/results_cluster'
OUT_BASE = '/Users/gdehol/git/bauer/notebooks/figures'

# Per-dataset fit registry: maps fit label → (subdir, filename, model_kind)
# model_kind in {'static', 'ddm', 'rdm', 'static-flex', 'ddm-flex', 'rdm-flex'}
DATASETS = {
    'garcia': {
        'subdir': 'garcia/64subj',
        'fits': {
            'choice':       ('choice.nc',                   'static'),
            'choice-flex':  ('choice_flex.nc',              'static-flex'),
            'DDM':          ('ddm_freescale.nc',            'ddm'),
            'DDM-flex':     ('ddm_flex_fixedscale.nc',      'ddm-flex'),
            'RDM':          ('rdm.nc',                      'rdm'),
            'RDM-flex':     ('rdm_flex.nc',                 'rdm-flex'),
        },
        'cog_params': ['prior_mu', 'prior_sd', 'n1_evidence_sd', 'n2_evidence_sd'],
    },
    'dehollander_dotcloud': {
        'subdir': 'dehollander_dotcloud/30subj',
        'fits': {
            'choice':       ('choice_full.nc',                  'static'),
            'choice-flex':  ('choice_flex_full.nc',             'static-flex'),
            'DDM':          ('ddm_freescale_full.nc',           'ddm'),
            'DDM-flex':     ('ddm_flex_freescale_full.nc',      'ddm-flex'),
            'RDM':          ('rdm_full.nc',                     'rdm'),
            'RDM-flex':     ('rdm_flex_full.nc',                'rdm-flex'),
        },
        'cog_params': ['risky_prior_mu', 'safe_prior_mu',
                        'risky_prior_sd', 'safe_prior_sd',
                        'n1_evidence_sd', 'n2_evidence_sd'],
    },
    'dehollander_symbolic': {
        'subdir': 'dehollander_symbolic/58subj',
        'fits': {
            'choice':       ('choice_full.nc',                  'static'),
            'choice-flex':  ('choice_flex_full.nc',             'static-flex'),
            'DDM':          ('ddm_freescale_full.nc',           'ddm'),
            'DDM-flex':     ('ddm_flex_freescale_full.nc',      'ddm-flex'),
            'RDM':          ('rdm_full.nc',                     'rdm'),
            'RDM-flex':     ('rdm_flex_full.nc',                'rdm-flex'),
        },
        'cog_params': ['risky_prior_mu', 'safe_prior_mu',
                        'risky_prior_sd', 'safe_prior_sd',
                        'n1_evidence_sd', 'n2_evidence_sd'],
    },
    'tms': {
        'subdir': 'tms/35subj',
        'fits': {
            'choice':       ('choice_full.nc',                  'static'),
            'choice-flex':  ('choice_flex_full.nc',             'static-flex'),
            'DDM':          ('ddm_freescale_full.nc',           'ddm'),
            'DDM-flex':     ('ddm_flex_freescale_full.nc',      'ddm-flex'),
            'RDM':          ('rdm_full.nc',                     'rdm'),
            'RDM-flex':     ('rdm_flex_full.nc',                'rdm-flex'),
            'choice-reg':       ('choice_reg_full.nc',                  'static'),
            'choice-flex-reg':  ('choice_flex_reg_full.nc',             'static-flex'),
            'DDM-flex-reg':     ('ddm_flex_reg_freescale_full.nc',      'ddm-flex'),
            'RDM-flex-reg':     ('rdm_flex_reg_full.nc',                'rdm-flex'),
        },
        'cog_params': ['risky_prior_mu', 'safe_prior_mu',
                        'risky_prior_sd', 'safe_prior_sd',
                        'n1_evidence_sd', 'n2_evidence_sd'],
    },
}


def per_subject_summary(idata, param, hdi_prob=0.95):
    """Per-subject mean + HDI for one parameter, sorted by mean ascending."""
    if param not in idata.posterior.data_vars:
        return None
    da = idata.posterior[param]
    if 'subject' not in da.dims:
        return None
    flat = da.stack(sample=('chain', 'draw'))
    df = pd.DataFrame({
        'subject': idata.posterior.coords['subject'].values,
        'mean': flat.mean('sample').values,
        'lo': flat.quantile((1 - hdi_prob) / 2, dim='sample').values,
        'hi': flat.quantile(1 - (1 - hdi_prob) / 2, dim='sample').values,
    }).sort_values('mean').reset_index(drop=True)
    df['rank'] = np.arange(len(df))
    return df


def load_fits(dataset_name, root=ROOT, max_rhat=1.05, min_ess=50):
    """Load + filter fits for a dataset. Returns dict[label] -> idata."""
    cfg = DATASETS[dataset_name]
    subdir = cfg['subdir']
    out = {}
    skipped = []
    for label, (fname, kind) in cfg['fits'].items():
        fp = op.join(root, subdir, fname)
        if not op.exists(fp):
            continue
        try:
            idata = az.from_netcdf(fp)
        except Exception as e:
            skipped.append(f'{label}: load error ({e})')
            continue
        diag = diagnostics_summary(idata)
        if diag['max_rhat'] > max_rhat or diag['min_ess_bulk'] < min_ess:
            skipped.append(f'{label}: max_rhat={diag["max_rhat"]:.2f}, '
                            f'min_ess={diag["min_ess_bulk"]:.0f} → SKIPPED')
            continue
        out[label] = idata
    return out, skipped, cfg['cog_params']


def pair_plot(dataset_name, param, fits, out_path):
    """One figure per parameter; rows × cols = models × models."""
    summaries = {label: per_subject_summary(idata, param)
                 for label, idata in fits.items()}
    summaries = {k: v for k, v in summaries.items() if v is not None}
    models = list(summaries.keys())
    N = len(models)
    if N < 2:
        return False

    fig, axes = plt.subplots(N, N, figsize=(2.4 * N, 2.4 * N))
    all_lo = min(s['lo'].min() for s in summaries.values())
    all_hi = max(s['hi'].max() for s in summaries.values())
    pad = 0.05 * (all_hi - all_lo + 1e-12)
    lo, hi = all_lo - pad, all_hi + pad
    for i, m_y in enumerate(models):
        for j, m_x in enumerate(models):
            ax = axes[i, j] if N > 1 else axes
            if i == j:
                s = summaries[m_y]
                ax.errorbar(s['mean'], s['rank'],
                            xerr=np.array([s['mean'] - s['lo'], s['hi'] - s['mean']]),
                            fmt='o', ms=4, capsize=2, lw=0.7, color='C0')
                ax.set_yticks([]); ax.set_xlim(lo, hi)
                ax.text(0.04, 0.92, m_y, transform=ax.transAxes,
                        fontweight='bold', fontsize=9, va='top')
            else:
                sx, sy = summaries[m_x], summaries[m_y]
                merged = sx.merge(sy, on='subject', suffixes=('_x', '_y'))
                ax.errorbar(merged['mean_x'], merged['mean_y'],
                            xerr=np.array([merged['mean_x'] - merged['lo_x'],
                                           merged['hi_x'] - merged['mean_x']]),
                            yerr=np.array([merged['mean_y'] - merged['lo_y'],
                                           merged['hi_y'] - merged['mean_y']]),
                            fmt='o', ms=4, capsize=2, lw=0.7, alpha=0.75, color='C0')
                ax.plot([lo, hi], [lo, hi], 'k--', lw=0.6, alpha=0.4)
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.grid(alpha=0.25)
            if i == N - 1: ax.set_xlabel(m_x, fontsize=9)
            else: ax.tick_params(labelbottom=False)
            if j == 0: ax.set_ylabel(m_y, fontsize=9)
            else: ax.tick_params(labelleft=False)
    fig.suptitle(f'{dataset_name} — {param}: per-subject (95% CI)', y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=list(DATASETS.keys()))
    args = ap.parse_args()

    fits, skipped, cog_params = load_fits(args.dataset)
    print(f'=== {args.dataset} ===')
    print(f'fits passing filters: {list(fits.keys())}')
    if skipped:
        print('skipped:')
        for s in skipped: print(f'  {s}')

    out_dir = op.join(OUT_BASE, f'{args.dataset}_pairplots')
    os.makedirs(out_dir, exist_ok=True)
    rendered = []
    for p in cog_params:
        out = op.join(out_dir, f'pair_{p}.png')
        if pair_plot(args.dataset, p, fits, out):
            rendered.append(p)
    print(f'\nrendered {len(rendered)} parameter pair plots → {out_dir}/')
    for p in rendered:
        print(f'  pair_{p}.png')


if __name__ == '__main__':
    main()
