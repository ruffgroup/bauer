"""Shared plotting helpers for bauer per-dataset notebooks.

All functions consume an arviz InferenceData and (where needed) the
corresponding paradigm DataFrame. Conventions:

- Per-subject figure: x = subject sorted by mean, y = mean ± 95% CI.
  Built with seaborn FacetGrid (col=parameter) + errorbar overlay.
- Group-level figure: forest-style ridge plot of `*_mu` posteriors.
- PPC figure: data = scatter / point, PPC = line + shaded HDI band.
- Implied noise figure (flex models): σ_k(n) curves with 94% HDI bands.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diagnostics_summary(idata) -> dict:
    """Quick fit health check. Returns max r̂, divergence count, min ESS."""
    summary = az.summary(idata, kind='diagnostics')
    n_div = int(idata.sample_stats['diverging'].sum()) \
            if 'sample_stats' in idata else None
    n_total = int(idata.posterior.sizes['chain'] * idata.posterior.sizes['draw'])
    return {
        'max_rhat': float(summary['r_hat'].max()),
        'min_ess_bulk': float(summary['ess_bulk'].min()),
        'min_ess_tail': float(summary['ess_tail'].min()),
        'divergences': n_div,
        'n_samples': n_total,
        'n_chains': int(idata.posterior.sizes['chain']),
    }


# ---------------------------------------------------------------------------
# Group-level forest plot
# ---------------------------------------------------------------------------

def group_param_forest(idata, var_names=None, ax=None, hdi_prob=0.94):
    """Forest plot of group-level (`*_mu`) parameters.

    var_names defaults to all `*_mu` posterior vars.
    """
    if var_names is None:
        var_names = [v for v in idata.posterior.data_vars
                     if v.endswith('_mu') and not v.startswith('_')
                     and 'subject' not in idata.posterior[v].dims]
    return az.plot_forest(idata, var_names=var_names, hdi_prob=hdi_prob,
                           combined=True, ax=ax)


# ---------------------------------------------------------------------------
# Per-subject parameter dots
# ---------------------------------------------------------------------------

def per_subject_summary_long(idata, params, hdi_prob=0.95) -> pd.DataFrame:
    """Build a long-format DataFrame of per-subject posterior mean + HDI.

    Columns: parameter, subject, mean, lo, hi, rank (within parameter).
    Subjects are ranked (per parameter) by mean value, ascending.
    """
    rows = []
    for p in params:
        if p not in idata.posterior.data_vars:
            continue
        da = idata.posterior[p]
        if 'subject' not in da.dims:
            continue
        flat = da.stack(sample=('chain', 'draw'))
        means = flat.mean('sample').values
        lo = flat.quantile((1 - hdi_prob) / 2, dim='sample').values
        hi = flat.quantile(1 - (1 - hdi_prob) / 2, dim='sample').values
        subjects = idata.posterior.coords['subject'].values
        df = pd.DataFrame({'parameter': p, 'subject': subjects,
                            'mean': means, 'lo': lo, 'hi': hi})
        df = df.sort_values('mean').reset_index(drop=True)
        df['rank'] = np.arange(len(df))
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def per_subject_dots(idata, params, hdi_prob=0.95, height=3.0, aspect=1.4):
    """Per-subject dot plot. x = rank-sorted subject, y = posterior mean ± HDI.

    Returns a sns.FacetGrid keyed by `parameter`. Errorbars use 95% credible
    intervals computed from posterior quantiles.
    """
    long = per_subject_summary_long(idata, params, hdi_prob=hdi_prob)
    if long.empty:
        return None
    g = sns.FacetGrid(long, col='parameter', col_wrap=3, sharey=False,
                       height=height, aspect=aspect)
    def _dots(data, **kw):
        ax = plt.gca()
        ax.errorbar(data['rank'], data['mean'],
                    yerr=[data['mean'] - data['lo'], data['hi'] - data['mean']],
                    fmt='o', ms=4, lw=0.7, capsize=2, color='C0', alpha=0.85)
        ax.set_xlabel('subject (sorted)')
    g.map_dataframe(_dots)
    g.set_titles(col_template='{col_name}')
    g.set_axis_labels('subject (sorted by posterior mean)', 'posterior mean ± 95% CI')
    return g


# ---------------------------------------------------------------------------
# PPC plotting (data points + line + shaded band)
# ---------------------------------------------------------------------------

def ppc_lineplot(data, x, y, hue, ax=None, palette='deep'):
    """Plot data scatter + PPC mean line + 80% HDI shaded band.

    Expects a long-format DataFrame with columns:
      x, y, hue, role  (role in {'data', 'ppc'})

    Or two separate DataFrames passed via `data={'data': df_obs, 'ppc': df_ppc}`.
    """
    raise NotImplementedError(
        'Implement in per-notebook context based on dataset-specific binning.')


# ---------------------------------------------------------------------------
# Implied noise function σ_k(n) for flex models
# ---------------------------------------------------------------------------

def implied_noise(model, idata, variable='n1_evidence_sd', n_grid=None,
                   hdi_prob=0.94):
    """Compute group-level σ_k(n) curves on a magnitude grid.

    For FlexibleNoise* models: applies the spline design matrix to the
    posterior of group-level coefficients (`{variable}_spline{i}_mu`),
    then softplus-transforms to get σ values.

    The grid is **restricted to the variable's own support** (the column the
    spline was anchored to) to avoid extrapolation artefacts. n1_evidence_sd
    is anchored to paradigm['n1'], n2_evidence_sd to paradigm['n2']; if these
    have different ranges (e.g. Garcia with fixed n1 set vs wider n2), the
    σ_1 and σ_2 curves cover different x-ranges. This is correct.

    Returns a DataFrame with columns: n, mean, lo, hi.
    """
    if n_grid is None:
        # Restrict to the column actually used to build the basis. Avoid
        # extrapolating σ_k(n) outside its training range.
        col = 'n1' if variable in ('n1_evidence_sd', 'memory_noise_sd') else 'n2'
        nmin, nmax = model.paradigm[col].min(), model.paradigm[col].max()
        n_grid = np.linspace(nmin, nmax, 60)
    dm = np.asarray(model.make_dm(x=n_grid, variable=variable))
    so = model.spline_order
    so_v = so[0] if variable == 'n1_evidence_sd' else so[1]
    keys = [f'{variable}_spline{i}_mu' for i in range(1, so_v + 1)]
    coefs = np.stack([idata.posterior[k].values for k in keys], axis=-1)
    z = np.einsum('cdp,np->cdn', coefs, dm)
    sigma = np.log1p(np.exp(z))  # softplus
    flat = sigma.reshape(-1, sigma.shape[-1])
    return pd.DataFrame({
        'n': n_grid,
        'mean': flat.mean(0),
        'lo': np.quantile(flat, (1 - hdi_prob) / 2, axis=0),
        'hi': np.quantile(flat, 1 - (1 - hdi_prob) / 2, axis=0),
    })
