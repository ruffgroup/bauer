"""De Hollander (2024) risky-choice analysis script.

Loads pre-fit ``choice_full.nc`` and ``ddm_freescale_full.nc`` for either the
dotcloud or symbolic task, then produces:
- fig1b.png: P(chose risky) vs n_safe, hue=risky_first (paper Fig 1B style)
- fig1c.png: psychometric curves vs log(n_risky/n_safe), faceted by stake bin
- chronometric_*.png: same layouts but with mean RT on y-axis
- params_*.txt: parameter summaries (mean / sd / r_hat) for each model

Usage:
    python notebooks/analyze_dehollander.py --task {dotcloud,symbolic} \
        --idata-dir <path> --out-dir <path> [--n-bins 8]

Plot conventions (per user):
- dots + errorbars = observed data
- shaded area + line = posterior-predictive (80% HDI)
- Risky first = orange (C1); Safe first = blue (C0)
"""
import argparse
import os
import os.path as op
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

from bauer.utils.data import (load_dehollander2024_risk,
                                load_dehollander2024_symbolic)
from bauer.models import RiskModel, DDMRiskModel


ORDER_COLORS = {True: 'C1', False: 'C0'}
ORDER_LABELS = {True: 'Risky first', False: 'Safe first'}


def _stake_bins(df, task):
    """Three stake bins matching paper convention."""
    if task == 'dotcloud':
        # Paper levels: 5, 7, 10, 14, 20, 28
        df['stake_bin'] = pd.cut(df['n_safe'], bins=[0, 8, 16, 100],
                                  labels=['small', 'medium', 'large'])
    else:
        # Continuous values; use tertiles
        edges = df['n_safe'].quantile([0, 1/3, 2/3, 1.0]).values
        edges[0] -= 1e-6; edges[-1] += 1e-6
        df['stake_bin'] = pd.cut(df['n_safe'], bins=edges,
                                  labels=['small', 'medium', 'large'])
    return df


def _prepare_data(task, fit_subs):
    if task == 'dotcloud':
        df = load_dehollander2024_risk()
    else:
        df = load_dehollander2024_symbolic()
    df = df.loc[df.index.get_level_values('subject').isin(fit_subs)].copy()
    df['n_risky'] = np.where(df['p1'] == 0.55, df['n1'], df['n2'])
    df['n_safe']  = np.where(df['p1'] == 0.55, df['n2'], df['n1'])
    df['log_ratio'] = np.log(df['n_risky'] / df['n_safe'])
    df['chose_risky'] = ((df['p2'] == 0.55) & df['choice']) | \
                         ((df['p1'] == 0.55) & ~df['choice'])
    df['risky_first'] = (df['p1'] == 0.55)
    return _stake_bins(df, task)


def ppc_to_long(ppc_df, df_):
    """Per-trial-per-draw PPC → long frame with ``simulated_chose_risky`` (and
    ``simulated_rt`` if present). Works for both static-choice and DDM/race
    PPCs since ``BaseModel.ppc``, ``DDMMixin.ppc`` and ``RaceMixin.ppc`` now all
    return the same shape: ``(trial_keys, ppc_sample)`` × ``simulated_choice``.
    """
    flat = ppc_df.reset_index().merge(
        df_.reset_index()[['subject', 'session', 'run', 'trial_nr',
                            'n_safe', 'risky_first', 'log_ratio', 'stake_bin']],
        on=['subject', 'session', 'run', 'trial_nr'], how='inner',
    )
    flat['simulated_chose_risky'] = np.where(
        flat['risky_first'].astype(bool),
        ~flat['simulated_choice'].astype(bool),
        flat['simulated_choice'].astype(bool),
    )
    return flat.rename(columns={'ppc_sample': 'draw'})


def fig_1b(out_path, df, models, kind='choice'):
    """x = n_safe, hue = risky_first, y = P(chose risky) or mean RT."""
    n_safe_levels = sorted(df['n_safe'].dropna().unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.0),
                              sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, (label, ppc_kind, ppc_long) in zip(axes, models):
        # PPC band+line first
        if ppc_long is not None:
            value_col = 'simulated_chose_risky' if kind == 'choice' else 'simulated_rt'
            per = ppc_long.groupby(['risky_first', 'n_safe', 'draw'])[
                value_col].mean().reset_index()
            summary = per.groupby(['risky_first', 'n_safe'])[value_col].agg(
                mean='mean',
                lo=lambda s: s.quantile(0.10),
                hi=lambda s: s.quantile(0.90),
            )
            for rf in (True, False):
                if rf in summary.index.get_level_values('risky_first'):
                    s = summary.loc[rf].sort_index()
                    ax.fill_between(s.index, s['lo'], s['hi'],
                                     color=ORDER_COLORS[rf], alpha=0.22)
                    ax.plot(s.index, s['mean'], color=ORDER_COLORS[rf], lw=1.6)
        # Observed dots
        for rf in (True, False):
            sub = df[df['risky_first'] == rf]
            obs_col = 'chose_risky' if kind == 'choice' else 'rt'
            grp = sub.groupby('n_safe')[obs_col].mean()
            sem = sub.groupby('n_safe')[obs_col].sem()
            ax.errorbar(grp.index, grp.values, yerr=sem.values, fmt='o',
                          color=ORDER_COLORS[rf], ms=7, capsize=3,
                          mec='black', mew=0.5, lw=0,
                          label=ORDER_LABELS[rf])
        if kind == 'choice':
            ax.axhline(0.5, color='gray', alpha=0.4, lw=0.7, ls=':')
            ax.set_ylim(0.15, 0.85)
        ax.set_xlabel('n_safe (stake size, CHF)')
        ax.set_xscale('log')
        ax.set_title(label)
        ax.grid(alpha=0.3)
        if len(n_safe_levels) <= 10:
            ax.set_xticks(n_safe_levels)
            ax.set_xticklabels(n_safe_levels)
    axes[0].set_ylabel('P(chose risky)' if kind == 'choice' else 'mean RT (s)')
    axes[-1].legend(fontsize=9, loc='best', frameon=True)
    n_subj = df.index.get_level_values('subject').nunique()
    fig.suptitle(f'P({"chose risky" if kind == "choice" else "RT"}) by stake × order — N={n_subj}\n'
                  f'dots = data, shaded + line = PPC (80% HDI)',
                  y=1.04, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'  saved -> {out_path}', flush=True)
    plt.close(fig)


def fig_1c(out_path, df, models, kind='choice', n_bins=8):
    """Psychometric/chronometric curves: x=log_ratio (binned), hue=risky_first,
    faceted by stake bin (small/medium/large)."""
    fig, axes = plt.subplots(len(models), 3, figsize=(11, 3.5 * len(models)),
                              sharey=True, sharex=True, squeeze=False)
    stake_bins = ['small', 'medium', 'large']

    bin_centers = {}
    bin_edges = {}
    for sb in stake_bins:
        sub = df[df['stake_bin'] == sb]
        if len(sub) < 5:
            bin_centers[sb] = np.array([])
            bin_edges[sb] = np.array([])
            continue
        edges = np.unique(np.quantile(sub['log_ratio'], np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            edges = np.linspace(sub['log_ratio'].min(), sub['log_ratio'].max(), n_bins + 1)
        bin_centers[sb] = 0.5 * (edges[:-1] + edges[1:])
        bin_edges[sb] = edges

    for r, (label, ppc_kind, ppc_long) in enumerate(models):
        for c, sb in enumerate(stake_bins):
            ax = axes[r, c]
            df_b = df[df['stake_bin'] == sb].copy()
            centers = bin_centers[sb]
            edges = bin_edges[sb]
            if len(centers) == 0:
                ax.set_visible(False); continue
            df_b['lr_bin'] = np.clip(np.searchsorted(edges, df_b['log_ratio']) - 1,
                                       0, len(centers) - 1)

            if ppc_long is not None:
                ppc_b = ppc_long[ppc_long['stake_bin'] == sb].copy()
                ppc_b['lr_bin'] = np.clip(np.searchsorted(edges, ppc_b['log_ratio']) - 1,
                                            0, len(centers) - 1)
                value_col = 'simulated_chose_risky' if kind == 'choice' else 'simulated_rt'
                per = ppc_b.groupby(['risky_first', 'lr_bin', 'draw'])[
                    value_col].mean().reset_index()
                summary = per.groupby(['risky_first', 'lr_bin'])[value_col].agg(
                    mean='mean',
                    lo=lambda s: s.quantile(0.10),
                    hi=lambda s: s.quantile(0.90),
                )
                for rf in (True, False):
                    if rf in summary.index.get_level_values('risky_first'):
                        s = summary.loc[rf].sort_index()
                        xs = centers[s.index.astype(int)]
                        ax.fill_between(xs, s['lo'], s['hi'],
                                         color=ORDER_COLORS[rf], alpha=0.22)
                        ax.plot(xs, s['mean'], color=ORDER_COLORS[rf], lw=1.5)

            obs_col = 'chose_risky' if kind == 'choice' else 'rt'
            for rf in (True, False):
                sub = df_b[df_b['risky_first'] == rf]
                grp = sub.groupby('lr_bin')[obs_col].agg(['mean', 'sem'])
                xs = centers[grp.index.astype(int)]
                ax.errorbar(xs, grp['mean'], yerr=grp['sem'].fillna(0),
                              fmt='o', color=ORDER_COLORS[rf], ms=5, capsize=2,
                              mec='black', mew=0.4, lw=0,
                              label=ORDER_LABELS[rf] if (r == 0 and c == 0) else None)
            if r == 0:
                ax.set_title(f'{sb} stakes')
            if kind == 'choice':
                ax.axhline(0.5, color='gray', alpha=0.4, lw=0.7, ls=':')
                ax.set_ylim(0.0, 1.0)
            ax.axvline(0, color='gray', alpha=0.4, lw=0.7, ls=':')
            ax.grid(alpha=0.3)
            if r == len(models) - 1:
                ax.set_xlabel('log(n_risky / n_safe)')
            if c == 0:
                ax.set_ylabel(f'{label}\n' +
                                ('P(chose risky)' if kind == 'choice' else 'mean RT (s)'))
    axes[0, 0].legend(fontsize=9, loc='upper left', frameon=True)
    n_subj = df.index.get_level_values('subject').nunique()
    fig.suptitle(f'Psychometric curves ({n_bins} bins) by stake × order — N={n_subj}\n'
                  f'dots = data, shaded + line = PPC (80% HDI)',
                  y=1.01, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'  saved -> {out_path}', flush=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['dotcloud', 'symbolic'], default='dotcloud')
    ap.add_argument('--idata-dir', required=True,
                     help='Directory with choice_full.nc and ddm_freescale_full.nc')
    ap.add_argument('--out-dir', required=True,
                     help='Where to save figures')
    ap.add_argument('--n-bins', type=int, default=8)
    ap.add_argument('--prior-estimate', default='full')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    nc_choice = op.join(args.idata_dir, f'choice_{args.prior_estimate}.nc')
    nc_ddm = op.join(args.idata_dir, f'ddm_freescale_{args.prior_estimate}.nc')

    if not op.exists(nc_choice):
        print(f'ERROR: {nc_choice} not found'); return

    print(f'Loading {nc_choice} ...', flush=True)
    idata_choice = az.from_netcdf(nc_choice)
    fit_subs = idata_choice.posterior.coords['subject'].values
    df = _prepare_data(args.task, fit_subs)
    print(f'  {len(df)} trials, {len(fit_subs)} subjects', flush=True)

    m_choice = RiskModel(paradigm=df, prior_estimate=args.prior_estimate,
                          fit_separate_evidence_sd=True)
    m_choice.build_estimation_model(data=df, hierarchical=True)
    print('  Choice PPC...', flush=True)
    ppc_c = m_choice.ppc(df, idata_choice, progressbar=False)
    long_c = ppc_to_long(ppc_c, df)

    long_d = None
    idata_ddm = None
    if op.exists(nc_ddm):
        print(f'Loading {nc_ddm} ...', flush=True)
        idata_ddm = az.from_netcdf(nc_ddm)
        m_ddm = DDMRiskModel(paradigm=df, prior_estimate=args.prior_estimate,
                              fit_separate_evidence_sd=True, fit_v_scale=True)
        m_ddm.build_estimation_model(data=df, hierarchical=True)
        print('  DDM PPC...', flush=True)
        ppc_d = m_ddm.ppc(df, idata_ddm, n_posterior_samples=120,
                           inner_samples=1, random_seed=0, progressbar=False)
        long_d = ppc_to_long(ppc_d, df)

    models_choice = [('choice', 'static', long_c)]
    if long_d is not None:
        models_choice.append(('DDM', 'ddm', long_d))

    print('Generating Fig 1B (choice)...', flush=True)
    fig_1b(op.join(args.out_dir, 'fig1b.png'), df, models_choice, kind='choice')
    print('Generating Fig 1C (psychometric)...', flush=True)
    fig_1c(op.join(args.out_dir, 'fig1c.png'), df, models_choice, kind='choice',
            n_bins=args.n_bins)

    if long_d is not None:
        print('Generating Fig 1B chronometric (DDM only)...', flush=True)
        fig_1b(op.join(args.out_dir, 'fig1b_chronometric.png'), df,
                [('DDM', 'ddm', long_d)], kind='rt')
        print('Generating Fig 1C chronometric (DDM only)...', flush=True)
        fig_1c(op.join(args.out_dir, 'fig1c_chronometric.png'), df,
                [('DDM', 'ddm', long_d)], kind='rt', n_bins=args.n_bins)

    # Parameter summaries
    out_params = op.join(args.out_dir, 'params.txt')
    with open(out_params, 'w') as f:
        f.write(f'=== Choice fit ({args.prior_estimate}) ===\n')
        var_names = ['n1_evidence_sd_mu', 'n2_evidence_sd_mu',
                      'risky_prior_mu_mu', 'safe_prior_mu_mu',
                      'risky_prior_sd_mu', 'safe_prior_sd_mu',
                      'prior_mu_mu', 'prior_sd_mu']
        var_names = [v for v in var_names if v in idata_choice.posterior]
        summ = az.summary(idata_choice, var_names=var_names)
        f.write(summ[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']].round(3).to_string() + '\n')

        if idata_ddm is not None:
            f.write(f'\n=== DDM fit ({args.prior_estimate}, v_scale free) ===\n')
            var_names = ['a_mu', 't0_mu', 'v_scale_mu',
                          'n1_evidence_sd_mu', 'n2_evidence_sd_mu',
                          'risky_prior_mu_mu', 'safe_prior_mu_mu',
                          'risky_prior_sd_mu', 'safe_prior_sd_mu',
                          'prior_mu_mu', 'prior_sd_mu']
            var_names = [v for v in var_names if v in idata_ddm.posterior]
            summ = az.summary(idata_ddm, var_names=var_names)
            f.write(summ[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']].round(3).to_string() + '\n')
    print(f'  saved -> {out_params}', flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
