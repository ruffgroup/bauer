"""De Hollander TMS-risk analysis script.

Loads pre-fit ``choice_full.nc``, ``ddm_freescale_full.nc``, and
``choice_reg_full.nc`` (regression on stimulation_condition), then produces:
- fig_psy_by_condition.png: psychometric curves split by TMS condition × risky_first
- fig_chr_by_condition.png: chronometric same layout
- fig_regression_coefs.png: TMS condition effect on n1/n2 evidence_sd
- params_*.txt: parameter summaries

Usage:
    python notebooks/analyze_tms.py --idata-dir <path> --out-dir <path>
"""
import argparse
import os
import os.path as op
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

from bauer.utils.data import load_dehollander_tms_risk
from bauer.models import RiskModel, DDMRiskModel, RiskRegressionModel


ORDER_COLORS = {True: 'C1', False: 'C0'}
ORDER_LABELS = {True: 'Risky first', False: 'Safe first'}
COND_COLORS = {'baseline': 'C2', 'vertex': 'C7', 'ips': 'C3'}


def _prepare(fit_subs):
    df = load_dehollander_tms_risk()
    df = df.loc[df.index.get_level_values('subject').isin(fit_subs)].copy()
    df['n_risky'] = np.where(df['p1'] == 0.55, df['n1'], df['n2'])
    df['n_safe']  = np.where(df['p1'] == 0.55, df['n2'], df['n1'])
    df['log_ratio'] = np.log(df['n_risky'] / df['n_safe'])
    df['chose_risky'] = ((df['p2'] == 0.55) & df['choice']) | \
                         ((df['p1'] == 0.55) & ~df['choice'])
    df['risky_first'] = (df['p1'] == 0.55)
    return df


def ppc_to_long(ppc_df, df_, extra_keep=()):
    """Per-trial-per-draw PPC → long with ``simulated_chose_risky`` (and
    ``simulated_rt`` if present). All bauer PPCs share the same long format
    after the BaseModel/DDM/Race PPC unification.
    """
    cols = ['subject', 'session', 'stimulation_condition', 'run', 'trial_nr',
            'n_safe', 'risky_first', 'log_ratio'] + list(extra_keep)
    flat = ppc_df.reset_index().merge(
        df_.reset_index()[cols],
        on=['subject', 'session', 'stimulation_condition', 'run', 'trial_nr'],
        how='inner',
    )
    flat['simulated_chose_risky'] = np.where(
        flat['risky_first'].astype(bool),
        ~flat['simulated_choice'].astype(bool),
        flat['simulated_choice'].astype(bool),
    )
    return flat.rename(columns={'ppc_sample': 'draw'})


def fig_psy_by_condition(out_path, df, models, kind='choice', n_bins=8):
    """Psychometric curves split by stimulation_condition (rows) × order (hue)."""
    conditions = ['baseline', 'vertex', 'ips']
    fig, axes = plt.subplots(len(conditions), len(models),
                              figsize=(4.5 * len(models), 3.0 * len(conditions)),
                              sharex=True, sharey=True, squeeze=False)
    edges = np.unique(np.quantile(df['log_ratio'], np.linspace(0, 1, n_bins + 1)))
    centers = 0.5 * (edges[:-1] + edges[1:])

    df = df.copy()
    df['lr_bin'] = np.clip(np.searchsorted(edges, df['log_ratio']) - 1, 0, len(centers) - 1)

    obs_col = 'chose_risky' if kind == 'choice' else 'rt'

    for c, (label, _, ppc_long) in enumerate(models):
        for r, cond in enumerate(conditions):
            ax = axes[r, c]
            df_cb = df[df['stimulation_condition'] == cond]
            if ppc_long is not None:
                ppc_b = ppc_long[ppc_long['stimulation_condition'] == cond].copy()
                ppc_b['lr_bin'] = np.clip(
                    np.searchsorted(edges, ppc_b['log_ratio']) - 1,
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
            for rf in (True, False):
                sub = df_cb[df_cb['risky_first'] == rf]
                grp = sub.groupby('lr_bin')[obs_col].agg(['mean', 'sem'])
                xs = centers[grp.index.astype(int)]
                ax.errorbar(xs, grp['mean'], yerr=grp['sem'].fillna(0),
                              fmt='o', color=ORDER_COLORS[rf], ms=5, capsize=2,
                              mec='black', mew=0.4, lw=0,
                              label=ORDER_LABELS[rf] if (r == 0 and c == 0) else None)
            if r == 0:
                ax.set_title(label)
            if c == 0:
                ax.set_ylabel(f'{cond}\n' +
                                ('P(chose risky)' if kind == 'choice' else 'mean RT (s)'))
            if kind == 'choice':
                ax.axhline(0.5, color='gray', alpha=0.4, lw=0.7, ls=':')
                ax.set_ylim(0.0, 1.0)
            ax.axvline(0, color='gray', alpha=0.4, lw=0.7, ls=':')
            ax.grid(alpha=0.3)
            if r == len(conditions) - 1:
                ax.set_xlabel('log(n_risky / n_safe)')
    axes[0, 0].legend(fontsize=8, loc='upper left', frameon=True)
    n_subj = df.index.get_level_values('subject').nunique()
    fig.suptitle(f'TMS — {kind} curves split by stimulation × order, N={n_subj}\n'
                  f'dots = data, shaded + line = PPC (80% HDI)', y=1.005, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'  saved -> {out_path}', flush=True)
    plt.close(fig)


def fig_regression_coefs(out_path, idata_reg):
    """Plot the regression coefficients for n1/n2_evidence_sd ~ stimulation_condition."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, var in zip(axes, ['n1_evidence_sd', 'n2_evidence_sd']):
        if f'{var}_mu' not in idata_reg.posterior:
            ax.text(0.5, 0.5, f'{var}_mu not in posterior', transform=ax.transAxes,
                     ha='center'); continue
        coefs = idata_reg.posterior[f'{var}_mu']
        # coefs has dims (chain, draw, regressor)
        if 'regressor' not in coefs.dims and f'{var}_mu_regressors' in coefs.dims:
            coefs = coefs.rename({f'{var}_mu_regressors': 'regressor'})
        if 'regressor' in coefs.dims:
            reg_names = coefs.coords['regressor'].values if 'regressor' in coefs.coords else \
                        np.arange(coefs.sizes['regressor'])
        else:
            reg_names = ['(scalar)']
            coefs = coefs.expand_dims({'regressor': reg_names})
        flat = coefs.values.reshape(-1, coefs.sizes['regressor'])
        means = flat.mean(0)
        lo = np.quantile(flat, 0.03, 0); hi = np.quantile(flat, 0.97, 0)
        x = np.arange(len(reg_names))
        ax.errorbar(x, means, yerr=[means - lo, hi - means], fmt='o', ms=8, capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(reg_names, rotation=30, ha='right')
        ax.set_title(var)
        ax.axhline(0, color='gray', alpha=0.4, lw=0.7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('coefficient (group-level posterior)')
    fig.suptitle('TMS effect on encoding noise (regression coefficients, 94% HDI)',
                  y=1.02, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'  saved -> {out_path}', flush=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idata-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--prior-estimate', default='full')
    ap.add_argument('--n-bins', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    candidates = sorted(os.listdir(args.idata_dir))
    n_subj_dirs = [d for d in candidates if d.endswith('subj')]
    if not n_subj_dirs:
        print(f'No <N>subj directories under {args.idata_dir}'); return
    sub_dir = op.join(args.idata_dir, sorted(n_subj_dirs, key=lambda x: int(x.replace('subj', '')))[-1])
    print(f'Reading from {sub_dir}', flush=True)

    nc_choice = op.join(sub_dir, f'choice_{args.prior_estimate}.nc')
    nc_ddm = op.join(sub_dir, f'ddm_freescale_{args.prior_estimate}.nc')
    nc_reg = op.join(sub_dir, f'choice_reg_{args.prior_estimate}.nc')

    if not op.exists(nc_choice):
        print(f'ERROR: {nc_choice} missing'); return
    idata_choice = az.from_netcdf(nc_choice)
    fit_subs = idata_choice.posterior.coords['subject'].values
    df = _prepare(fit_subs)
    print(f'  {len(df)} trials, {len(fit_subs)} subjects', flush=True)

    m_choice = RiskModel(paradigm=df, prior_estimate=args.prior_estimate,
                          fit_separate_evidence_sd=True)
    m_choice.build_estimation_model(data=df, hierarchical=True)
    print('  Choice PPC...', flush=True)
    ppc_c = m_choice.ppc(df, idata_choice, progressbar=False)
    long_c = ppc_to_long(ppc_c, df, extra_keep=())

    long_d = None
    idata_ddm = None
    if op.exists(nc_ddm):
        idata_ddm = az.from_netcdf(nc_ddm)
        m_ddm = DDMRiskModel(paradigm=df, prior_estimate=args.prior_estimate,
                              fit_separate_evidence_sd=True, fit_v_scale=True)
        m_ddm.build_estimation_model(data=df, hierarchical=True)
        print('  DDM PPC...', flush=True)
        ppc_d = m_ddm.ppc(df, idata_ddm, n_posterior_samples=120,
                           inner_samples=1, random_seed=0, progressbar=False)
        long_d = ppc_to_long(ppc_d, df)

    models = [('choice', 'static', long_c)]
    if long_d is not None:
        models.append(('DDM', 'ddm', long_d))

    fig_psy_by_condition(op.join(args.out_dir, 'fig_psy_by_condition.png'),
                          df, models, kind='choice', n_bins=args.n_bins)
    if long_d is not None:
        fig_psy_by_condition(op.join(args.out_dir, 'fig_chr_by_condition.png'),
                              df, [('DDM', 'ddm', long_d)], kind='rt',
                              n_bins=args.n_bins)

    if op.exists(nc_reg):
        idata_reg = az.from_netcdf(nc_reg)
        fig_regression_coefs(op.join(args.out_dir, 'fig_regression_coefs.png'),
                              idata_reg)
    else:
        print(f'  (skipping regression plot — {nc_reg} not found)', flush=True)

    out_params = op.join(args.out_dir, 'params.txt')
    with open(out_params, 'w') as f:
        for name, idata in (('choice', idata_choice),
                              ('ddm', idata_ddm),
                              ('choice + reg(stim_cond)', az.from_netcdf(nc_reg) if op.exists(nc_reg) else None)):
            if idata is None: continue
            f.write(f'=== {name} ===\n')
            wanted = [v for v in idata.posterior.data_vars if v.endswith('_mu')]
            wanted = [v for v in wanted if 'untransformed' not in v]
            try:
                summ = az.summary(idata, var_names=wanted)
                f.write(summ[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']].round(3).to_string() + '\n\n')
            except Exception as e:
                f.write(f'(summary error: {e})\n\n')
    print(f'  saved -> {out_params}', flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
