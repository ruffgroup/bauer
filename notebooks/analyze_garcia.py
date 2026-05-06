"""Garcia 2022 magnitude-comparison analysis script.

Loads pre-fit ``choice.nc``, ``ddm_freescale.nc``, ``ddm_flex_fixedscale.nc``
from a results dir (organised by N as ``<N>subj/...``), then produces:
- fig_psy.png: psychometric, x=log(n2/n1), hue=n1 (stake)
- fig_chr.png: chronometric, same layout
- fig_sigma_overlay.png: implied σ_k(n) curves (group-level), choice vs DDM
- params.txt

Usage:
    python notebooks/analyze_garcia.py --idata-dir <path> --out-dir <path>
"""
import argparse
import os
import os.path as op
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from bauer.utils.data import load_garcia2022
from bauer.models import (
    MagnitudeComparisonModel, DDMMagnitudeComparisonModel,
    FlexibleNoiseComparisonModel, DDMFlexibleNoiseComparisonModel,
)


def _prepare(fit_subs):
    df = load_garcia2022(task='magnitude')
    df = df.loc[df.index.get_level_values('subject').isin(fit_subs)].copy()
    df['log_ratio'] = np.log(df['n2'] / df['n1'])
    return df


def fig_psy_chr(out_path, df, models, kind='choice'):
    """One panel per model. x=log(n2/n1), hue=n1, y=P(chose 2) or RT.
    dots = data, shaded + line = PPC."""
    n1_levels = sorted(df['n1'].unique())
    cmap = cm.viridis(np.linspace(0.05, 0.95, len(n1_levels)))
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4.5),
                              sharey=True)
    if len(models) == 1: axes = [axes]
    obs_col = 'choice' if kind == 'choice' else 'rt'

    for ax, (label, summary) in zip(axes, models):
        for n1, color in zip(n1_levels, cmap):
            o = df[df['n1'] == n1].groupby('n2')[obs_col].mean()
            ax.plot(np.log(o.index / n1), o.values, 'o', color=color, ms=5,
                     mec='black', mew=0.4, label=f'n1={n1}')
            if summary is not None:
                try:
                    s = summary.xs(n1, level='n1').sort_index()
                    xs = np.log(s.index.astype(float) / n1)
                    ax.fill_between(xs, s['lo'], s['hi'],
                                     color=color, alpha=0.15)
                    ax.plot(xs, s['mean'], color=color, lw=1.4)
                except KeyError:
                    pass
        ax.set_xlabel('log(n2 / n1)')
        ax.set_title(label)
        ax.grid(alpha=0.3)
        if kind == 'choice':
            ax.axhline(0.5, color='gray', alpha=0.4, lw=0.7, ls=':')
            ax.set_ylim(-0.02, 1.02)
        ax.axvline(0, color='gray', alpha=0.4, lw=0.7, ls=':')
    axes[0].set_ylabel('P(choose n2)' if kind == 'choice' else 'mean RT (s)')
    axes[-1].legend(title='n1 (stake)', fontsize=8, loc='best')
    n_subj = df.index.get_level_values('subject').nunique()
    fig.suptitle(f'{kind.upper()} — observed (dots) vs PPC (shaded + line, 80% HDI), '
                  f'N={n_subj}', y=1.02, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'  saved -> {out_path}', flush=True)
    plt.close(fig)


def ppc_summary(ppc_df, df_, value_col):
    """Per-(n1,n2) PPC summary (mean + 80% HDI across draws). Works for any
    bauer model — all PPCs now share the same long format with ``ppc_sample``
    in the index and ``simulated_choice`` / ``simulated_rt`` columns.
    """
    pd_meta = df_.reset_index()[['subject', 'format', 'run', 'trial_nr', 'n1', 'n2']]
    flat = ppc_df.reset_index().merge(pd_meta, on=['subject', 'format', 'run', 'trial_nr'])
    flat[value_col] = flat[value_col].astype(float)
    per = flat.groupby(['n1', 'n2', 'ppc_sample'])[value_col].mean().rename('v').reset_index()
    out = per.groupby(['n1', 'n2'])['v'].agg(['mean',
                                              lambda s: np.quantile(s, 0.10),
                                              lambda s: np.quantile(s, 0.90)])
    out.columns = ['mean', 'lo', 'hi']
    return out


def fig_sigma_overlay(out_path, df, m_fc, idata_fc, m_fd, idata_fd):
    """Group-level implied σ_k(n) curves: choice (blue) vs DDM (orange).
    σ₁ solid, σ₂ dashed."""
    n_min = float(min(df['n1'].min(), df['n2'].min()))
    n_max = float(max(df['n1'].max(), df['n2'].max()))
    n_grid = np.linspace(n_min, n_max, 60)

    def implied_group(model, idata, variable):
        dm = np.asarray(model.make_dm(x=n_grid, variable=variable))
        poly = model.polynomial_order
        poly_v = poly[0] if variable == 'n1_evidence_sd' else poly[1]
        keys = [f'{variable}_spline{i}_mu' for i in range(1, poly_v + 1)]
        coefs = np.stack([idata.posterior[k].values for k in keys], axis=-1)
        z = np.einsum('cdp,np->cdn', coefs, dm)
        return np.log1p(np.exp(z))

    sig1_fc = implied_group(m_fc, idata_fc, 'n1_evidence_sd')
    sig2_fc = implied_group(m_fc, idata_fc, 'n2_evidence_sd')
    sig1_fd = implied_group(m_fd, idata_fd, 'n1_evidence_sd')
    sig2_fd = implied_group(m_fd, idata_fd, 'n2_evidence_sd')

    def _band(arr):
        flat = arr.reshape(-1, arr.shape[-1])
        return flat.mean(0), np.quantile(flat, 0.03, 0), np.quantile(flat, 0.97, 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    n1_lo, n1_hi = df['n1'].min(), df['n1'].max()
    ax.axvspan(n1_lo, n1_hi, color='gray', alpha=0.06, lw=0,
                label=f'n1 support [{n1_lo}, {n1_hi}]')
    for sig, color, ls, label in [
        (sig1_fc, 'C0', '-',  'σ₁ choice'),
        (sig2_fc, 'C0', '--', 'σ₂ choice'),
        (sig1_fd, 'C1', '-',  'σ₁ DDM'),
        (sig2_fd, 'C1', '--', 'σ₂ DDM'),
    ]:
        m, lo, hi = _band(sig)
        ax.fill_between(n_grid, lo, hi, color=color, alpha=0.18)
        ax.plot(n_grid, m, color=color, lw=2.0, ls=ls, label=label)
    ax.set_xlabel('n')
    ax.set_ylabel('σ_k(n)')
    n_subj = df.index.get_level_values('subject').nunique()
    ax.set_title(f'Implied encoding noise σ_k(n) — choice vs DDM, N={n_subj}\n'
                  '94% HDI bands; σ₁ solid, σ₂ dashed')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'  saved -> {out_path}', flush=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idata-dir', required=True,
                     help='Directory with choice.nc / ddm_freescale.nc / ddm_flex_fixedscale.nc')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    nc_choice = op.join(args.idata_dir, 'choice.nc')
    nc_ddm = op.join(args.idata_dir, 'ddm_freescale.nc')
    nc_fddm = op.join(args.idata_dir, 'ddm_flex_fixedscale.nc')
    if not op.exists(nc_choice):
        print(f'ERROR: {nc_choice} missing'); return

    print(f'Loading {nc_choice} ...', flush=True)
    idata_c = az.from_netcdf(nc_choice)
    fit_subs = idata_c.posterior.coords['subject'].values
    df = _prepare(fit_subs)
    print(f'  {len(df)} trials, {len(fit_subs)} subjects', flush=True)

    m_choice = MagnitudeComparisonModel(paradigm=df, fit_seperate_evidence_sd=True,
                                         fit_prior=True)
    m_choice.build_estimation_model(data=df, hierarchical=True)
    print('  choice PPC...', flush=True)
    ppc_c = m_choice.ppc(df, idata_c, progressbar=False)
    sum_psy_c = ppc_summary(ppc_c, df, 'simulated_choice')

    sum_psy_d = sum_chr_d = idata_d = None
    sum_psy_fd = sum_chr_fd = idata_fd = None
    m_fd = None
    if op.exists(nc_ddm):
        idata_d = az.from_netcdf(nc_ddm)
        m_ddm = DDMMagnitudeComparisonModel(paradigm=df, fit_seperate_evidence_sd=True,
                                             fit_v_scale=True, fit_prior=True)
        m_ddm.build_estimation_model(data=df, hierarchical=True)
        print('  DDM PPC...', flush=True)
        ppc_d = m_ddm.ppc(df, idata_d, n_posterior_samples=120, inner_samples=1,
                           random_seed=0, progressbar=False)
        sum_psy_d = ppc_summary(ppc_d, df, 'simulated_choice')
        sum_chr_d = ppc_summary(ppc_d, df, 'simulated_rt')
    if op.exists(nc_fddm):
        idata_fd = az.from_netcdf(nc_fddm)
        m_fd = DDMFlexibleNoiseComparisonModel(
            paradigm=df, fit_seperate_evidence_sd=True, polynomial_order=5,
            fit_prior=True, fit_v_scale=False)
        m_fd.build_estimation_model(paradigm=df, hierarchical=True)
        print('  flex-DDM PPC...', flush=True)
        ppc_fd = m_fd.ppc(df, idata_fd, n_posterior_samples=120, inner_samples=1,
                           random_seed=0, progressbar=False)
        sum_psy_fd = ppc_summary(ppc_fd, df, 'simulated_choice')
        sum_chr_fd = ppc_summary(ppc_fd, df, 'simulated_rt')

    psy_models = [('choice', sum_psy_c)]
    chr_models = []
    if sum_psy_d is not None: psy_models.append(('DDM', sum_psy_d))
    if sum_psy_fd is not None: psy_models.append(('flex-DDM', sum_psy_fd))
    if sum_chr_d is not None: chr_models.append(('DDM', sum_chr_d))
    if sum_chr_fd is not None: chr_models.append(('flex-DDM', sum_chr_fd))

    fig_psy_chr(op.join(args.out_dir, 'fig_psy.png'), df, psy_models, kind='choice')
    if chr_models:
        fig_psy_chr(op.join(args.out_dir, 'fig_chr.png'), df, chr_models, kind='rt')

    # Implied σ(n) — only if flex-DDM is available
    nc_fchoice = op.join(args.idata_dir, 'choice_flex.nc')
    if m_fd is not None and op.exists(nc_fchoice):
        idata_fc = az.from_netcdf(nc_fchoice)
        m_fc = FlexibleNoiseComparisonModel(paradigm=df, fit_seperate_evidence_sd=True,
                                              polynomial_order=5, fit_prior=True)
        m_fc.build_estimation_model(paradigm=df, hierarchical=True)
        fig_sigma_overlay(op.join(args.out_dir, 'fig_sigma_overlay.png'),
                           df, m_fc, idata_fc, m_fd, idata_fd)

    out_params = op.join(args.out_dir, 'params.txt')
    with open(out_params, 'w') as f:
        for name, idata in (('choice', idata_c), ('ddm', idata_d), ('flex-ddm', idata_fd)):
            if idata is None: continue
            wanted = [v for v in idata.posterior.data_vars if v.endswith('_mu')]
            wanted = [v for v in wanted if 'untransformed' not in v]
            f.write(f'=== {name} ===\n')
            try:
                summ = az.summary(idata, var_names=wanted)
                f.write(summ[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']].round(3).to_string() + '\n\n')
            except Exception as e:
                f.write(f'(summary error: {e})\n\n')
    print(f'  saved -> {out_params}', flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
