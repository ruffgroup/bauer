import warnings
from .bayes import summarize_ppc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az
from .math import softplus_np, logistic_np


def cluster_offers(d, n=6, key='log(risky/safe)'):
    return pd.qcut(d[key], n, duplicates='drop').apply(lambda x: x.mid)


def get_hdi(d):
    return pd.Series(az.hdi(d.values, hdi_prob=0.95), index=['hdi_low', 'hdi_high'])


def plot_ppc(df, ppc, exp_type='magnitude', plot_type=1, var_name='p', level='subject', col_wrap=5, n_clusters=13):

    if exp_type == 'magnitude':
        x = 'log(n2/n1)'

        if 'log(n2/n1)' not in ppc.index.names:
            if 'frac' in ppc.index.names:
                ppc['log(n2/n1)'] = np.log(ppc.index.get_level_values('frac'))
            else:
                ppc['log(n2/n1)'] = np.log(ppc.index.get_level_values('n2')) - np.log(ppc.index.get_level_values('n1'))

            ppc.set_index('log(n2/n1)', append=True)

    assert (var_name in ['p', 'll_bernoulli'])

    ppc = ppc.xs(var_name, 0, 'variable').copy()

    df = df.copy()

    # Make sure that we group data from (Roughly) same fractions
    if not (df.groupby(['subject', x]).size().groupby('subject').size() < (n_clusters + 1)).all():
        if level == 'subject':
            df[x] = df.groupby(['subject'],
                               group_keys=False).apply(lambda d: cluster_offers(d, n_clusters, x))
        else:
            df[x] = cluster_offers(df, n_clusters, x)

    if plot_type == 1:
        groupby = [x, 'n1']
    elif plot_type == 2:
        groupby = ['n1']
    else:
        raise NotImplementedError

    if level == 'group':
        ppc = ppc.groupby(['subject'] + groupby).mean()

    if level == 'subject':
        groupby = ['subject'] + groupby

    ppc_summary = summarize_ppc(ppc, groupby=groupby)
    p = df.groupby(groupby).mean()[['choice']]
    ppc_summary = ppc_summary.join(p, how='outer').reset_index()

    if plot_type in [1]:
        fac = sns.FacetGrid(ppc_summary,
                            hue='n1',
                            col='subject' if level == 'subject' else None,
                            col_wrap=col_wrap if level == 'subject' else None)
        fac.map_dataframe(plot_prediction, x=x)
        fac.map(plt.scatter, x, 'choice')

    if plot_type in [2]:
        fac = sns.FacetGrid(ppc_summary,
                            col='subject' if level == 'subject' else None,
                            col_wrap=col_wrap if level == 'subject' else None)
        fac.map_dataframe(plot_prediction, x='n1')
        fac.map(plt.scatter, 'n1', 'choice')

    fac.add_legend()

    return fac


def plot_prediction(data, x, color, y='p_predicted', alpha=.25, **kwargs):
    data = data[~data['hdi025'].isnull()]

    plt.fill_between(data[x], data['hdi025'],
                     data['hdi975'], color=color, alpha=alpha)
    return plt.plot(data[x], data[y], color=color)


def plot_subjectwise_parameters(idata, parameter, transform=None, sort_subjects=True,
                                plot_group_mean=True, hdi_prob=0.94, color='steelblue',
                                ax=None, label=None, **kwargs):
    """Plot subject-level posterior estimates as a sorted point-plot with HDI error bars.

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior samples from a fitted bauer model.
    parameter : str
        Name of the subject-level parameter (e.g. ``'n1_evidence_sd'``).
    transform : str or None
        Optional transform applied to samples before plotting.
        One of ``'softplus'``, ``'logistic'``, or ``None``.
    sort_subjects : bool
        If True (default) subjects are sorted by their posterior mean on the x-axis.
        If False, subjects appear in their original order.
    plot_group_mean : bool
        If True (default) and a ``{parameter}_mu`` variable exists in ``idata``,
        draw a dashed horizontal line at the group-mean posterior mean.
    hdi_prob : float
        Posterior mass for the HDI interval shown as error bars (default 0.94).
    color : str
        Colour for the points and error bars.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.  If None, the current axes are used.
    label : str or None
        Legend label for the series.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    samples = idata.posterior[parameter].values  # shape (chain, draw, subject)
    samples_flat = samples.reshape(-1, samples.shape[-1])  # (samples, subject)

    if transform is not None:
        if transform == 'softplus':
            samples_flat = softplus_np(samples_flat)
        elif transform == 'logistic':
            samples_flat = logistic_np(samples_flat)
        else:
            raise ValueError(f'{transform} is not a valid transformation')

    means = samples_flat.mean(0)
    lo_prob = (1 - hdi_prob) / 2 * 100
    hi_prob = 100 - lo_prob
    lows = np.percentile(samples_flat, lo_prob, axis=0)
    highs = np.percentile(samples_flat, hi_prob, axis=0)

    sort_idx = np.argsort(means) if sort_subjects else np.arange(len(means))
    x_vals = np.arange(len(means))

    ax.errorbar(
        x_vals,
        means[sort_idx],
        yerr=[means[sort_idx] - lows[sort_idx], highs[sort_idx] - means[sort_idx]],
        fmt='o', ms=5, elinewidth=0.9, capsize=2.5, alpha=0.75,
        color=color, ecolor=color, label=label, **kwargs,
    )

    ax.set_xlabel('Subject (sorted by posterior mean)' if sort_subjects else 'Subject')
    ax.set_ylabel(parameter)
    sns.despine(ax=ax)

    if plot_group_mean:
        mu_key = parameter + '_mu'
        if mu_key in idata.posterior:
            group_mean_val = idata.posterior[mu_key].values.mean()
            ax.axhline(group_mean_val, c=color, ls='--', lw=1.5, alpha=0.7,
                       label=f'Group mean ({mu_key})')

    return ax


def get_subject_posterior_df(idata, parameters, hdi_prob=0.94):
    """Extract a tidy subject-level posterior summary DataFrame from an InferenceData.

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior from a fitted bauer model.
    parameters : list of str
        Parameter names to extract (e.g. ``['n1_evidence_sd', 'n2_evidence_sd']``).
    hdi_prob : float
        Posterior mass for the HDI interval (default 0.94).

    Returns
    -------
    pd.DataFrame
        Columns: ``parameter``, ``subject``, ``mean``, ``lo``, ``hi``.
        Suitable for use with ``sns.FacetGrid`` and ``plot_subjectwise_pointplot``.

    Examples
    --------
    >>> df = get_subject_posterior_df(idata_mag,
    ...                               ['n1_evidence_sd', 'n2_evidence_sd'])
    >>> g = sns.FacetGrid(df, col='parameter', sharey=True)
    >>> g.map_dataframe(plot_subjectwise_pointplot, 'mean', 'lo', 'hi')
    """
    lo_p = (1 - hdi_prob) / 2 * 100
    hi_p = 100 - lo_p
    rows = []
    for param in parameters:
        if param not in idata.posterior:
            continue
        samp = idata.posterior[param].values
        samp_flat = samp.reshape(-1, samp.shape[-1])  # (samples, subjects)
        for si in range(samp_flat.shape[-1]):
            rows.append({
                'parameter': param,
                'subject': si,
                'mean': samp_flat[:, si].mean(),
                'lo': np.percentile(samp_flat[:, si], lo_p),
                'hi': np.percentile(samp_flat[:, si], hi_p),
            })
    return pd.DataFrame(rows)


def plot_subjectwise_pointplot(data, mean_col='mean', lo_col='lo', hi_col='hi',
                               sort_subjects=True, ax=None, **kwargs):
    """FacetGrid-compatible sorted point-plot with HDI error bars.

    Designed for use with ``sns.FacetGrid.map_dataframe``.  Each row in *data*
    represents one subject; *mean_col*, *lo_col*, *hi_col* are the column names
    for the posterior mean and HDI bounds.

    Parameters
    ----------
    data : pd.DataFrame
        One row per subject with at minimum columns *mean_col*, *lo_col*, *hi_col*.
    mean_col, lo_col, hi_col : str
        Column names for the posterior mean, HDI lower bound, HDI upper bound.
    sort_subjects : bool
        If True (default), subjects are sorted by their posterior mean.
    **kwargs
        Forwarded to ``ax.errorbar``; ``color`` is set by seaborn hue mapping.

    Examples
    --------
    >>> df = get_subject_posterior_df(idata_mag,
    ...                               ['n1_evidence_sd', 'n2_evidence_sd'])
    >>> g = sns.FacetGrid(df, col='parameter', sharey=True)
    >>> g.map_dataframe(plot_subjectwise_pointplot, 'mean', 'lo', 'hi')
    >>> g.set_axis_labels('Subject (sorted)', 'Noise parameter')
    """
    if ax is None:
        ax = plt.gca()
    color = kwargs.pop('color', 'steelblue')
    label = kwargs.pop('label', None)

    means = data[mean_col].values
    los = data[lo_col].values
    his = data[hi_col].values

    sort_idx = np.argsort(means) if sort_subjects else np.arange(len(means))
    x_vals = np.arange(len(means))

    ax.errorbar(
        x_vals,
        means[sort_idx],
        yerr=[means[sort_idx] - los[sort_idx], his[sort_idx] - means[sort_idx]],
        fmt='o', ms=5, elinewidth=0.9, capsize=2.5, alpha=0.75,
        color=color, ecolor=color, label=label, **kwargs,
    )
    if sort_subjects:
        ax.set_xlabel('Subject (sorted by posterior mean)')
    else:
        ax.set_xlabel('Subject')
    sns.despine(ax=ax)
    return ax


# ─── House-style posterior and PPC plots ─────────────────────────────────────
#
# One vocabulary for all of them:
#   by            column whose levels are rows / panels / lines (default 'condition')
#   order         the levels, in the order to draw them (default: order of appearance)
#   palette       dict level -> colour, or one colour for everything
#   level_labels  dict level -> display name
#   xlabel(s)     axis labels
# Inputs are the tidy summaries from bauer.utils.summaries
# (summarize_posterior -> mean / lo / hi; ppc_summary + ppc_observed -> pred / lo /
# hi and obs). Dot = posterior mean, line = 95% HDI, open circle = true value
# (simulations), dashed line = reference (0 for contrasts, group mean for
# subjects). Every function returns the matplotlib Figure and raises a
# ValueError that says what is wrong instead of drawing empty panels.

HOUSE_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'lines.markersize': 4,
    'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
}


def set_house_style():
    """Publication style: Helvetica 7–8 pt, despined, outward ticks, editable PDF text."""
    import matplotlib as mpl
    mpl.rcParams.update(HOUSE_STYLE)


def _need(df, cols, what):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing column(s) {missing}; it has {list(df.columns)}")


def _levels(df, col, order):
    present = list(pd.unique(df[col]))
    if order is None:
        return present
    order = list(order)
    unlisted = [lv for lv in present if lv not in order]
    if unlisted:
        raise ValueError(f"levels {unlisted} of {col!r} are not in order={order} "
                         "(typo?); they would not be drawn.")
    # order entries absent from the data are skipped (e.g. 'placebo' in a
    # contrast plot), so one order list can serve every figure
    return [lv for lv in order if lv in present]


def _color(palette, key, default='#3B5BA5'):
    if palette is None:
        return default
    if isinstance(palette, dict):
        if key not in palette:
            raise ValueError(f"palette has no colour for {key!r}; it has {list(palette)}")
        return palette[key]
    return palette          # a single colour


def _label(level_labels, key):
    return (level_labels or {}).get(key, key)


def _select_parameter(df, parameter, what):
    if 'parameter' not in df.columns:
        return df
    d = df[df['parameter'] == parameter]
    if d.empty:
        raise ValueError(f"{what} has no rows for parameter {parameter!r}; "
                         f"parameters are {list(pd.unique(df['parameter']))}")
    return d


def _true_key(ax, x=1.04, y=0.92):
    tf = ax.transAxes
    ax.plot(x, y, 'o', color='0.35', ms=4.5, transform=tf, clip_on=False)
    ax.text(x + 0.05, y, 'Estimate', transform=tf, va='center', fontsize=6.5)
    ax.plot(x, y - 0.14, 'o', mfc='none', mec='0.15', ms=6, mew=0.8, transform=tf, clip_on=False)
    ax.text(x + 0.05, y - 0.14, 'True', transform=tf, va='center', fontsize=6.5)


def plot_forest(summary, parameters, by='condition', order=None, palette=None,
                level_labels=None, xlabels=None, ref=None, true_col=None,
                figsize=None, axes=None):
    """Group-level forest plot.

    One panel per parameter, one row per level of ``by`` (e.g. condition or
    contrast). Dot = posterior mean, horizontal line = 95% HDI (``lo``–``hi``),
    dashed vertical line at ``ref`` (use 0 for contrasts), open circles at
    ``true_col`` (simulations).

    Parameters
    ----------
    summary : DataFrame from :func:`bauer.utils.summarize_posterior` with
        columns ``parameter``, ``by``, ``mean``, ``lo``, ``hi``; one row per
        parameter × level.
    parameters : parameter name or list of names (one panel each).
    xlabels : dict parameter -> x-axis label.
    """
    parameters = [parameters] if isinstance(parameters, str) else list(parameters)
    _need(summary, ['parameter', by, 'mean', 'lo', 'hi'] + ([true_col] if true_col else []),
          'summary')
    for par in parameters:
        d = _select_parameter(summary, par, 'summary')
        if d.duplicated(by).any():
            raise ValueError(f"summary has several rows per {by!r} for {par!r} (a subject-level "
                             'summary?). Use plot_sorted_forest, or summarise by '
                             f"['parameter', {by!r}].")
    levels = _levels(summary, by, order)
    if axes is None:
        figsize = figsize or (2.5 * len(parameters), 0.35 * len(levels) + 0.8)
        fig, axes = plt.subplots(1, len(parameters), figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes)
    if len(axes) < len(parameters):
        raise ValueError(f'{len(parameters)} parameters but only {len(axes)} axes')
    fig = axes[0].figure
    for ax, par in zip(axes, parameters):
        d = summary[summary['parameter'] == par].set_index(by)
        for i, lev in enumerate(levels):
            if lev not in d.index:
                continue
            y, col = len(levels) - 1 - i, _color(palette, lev)
            ax.plot([d.loc[lev, 'lo'], d.loc[lev, 'hi']], [y, y], color=col, lw=1.6,
                    solid_capstyle='butt')
            ax.plot(d.loc[lev, 'mean'], y, 'o', color=col, ms=5, zorder=3)
            if true_col is not None:
                ax.plot(d.loc[lev, true_col], y, 'o', mfc='none', mec='0.15', ms=6.5,
                        mew=0.8, zorder=4)
        if ref is not None:
            ax.axvline(ref, color='0.6', lw=0.7, ls='--', zorder=0)
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels([_label(level_labels, lev) for lev in levels][::-1])
        for t, lev in zip(ax.get_yticklabels(), levels[::-1]):
            t.set_color(_color(palette, lev) if palette is not None else '0.15')
        ax.set_ylim(-0.6, len(levels) - 0.4)
        ax.set_xlabel((xlabels or {}).get(par, par))
        ax.tick_params(axis='y', length=0)
    for ax in axes[1:len(parameters)]:
        ax.set_yticklabels([])
    if true_col is not None:
        _true_key(axes[len(parameters) - 1])
    sns.despine(fig=fig, offset=4, left=True)
    for ax in axes[:len(parameters)]:
        _extend_ticks_to_data(ax)
    return fig


def _extend_ticks_to_data(ax):
    """Put the outer x ticks at or beyond the data, so a trimmed spine never
    ends inside an interval."""
    import matplotlib.ticker as mticker
    lo, hi = ax.get_xlim()
    ticks = [t for t in ax.get_xticks() if lo <= t <= hi]
    if len(ticks) < 2:
        return
    step = ticks[1] - ticks[0]
    data = [ln.get_xdata() for ln in ax.lines if len(ln.get_xdata())]
    data = np.concatenate([np.atleast_1d(d).astype(float) for d in data]) if data else np.array([])
    data = data[np.isfinite(data)]
    if data.size:
        while ticks[0] > data.min() + 1e-12:
            ticks.insert(0, ticks[0] - step)
        while ticks[-1] < data.max() - 1e-12:
            ticks.append(ticks[-1] + step)
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.set_xlim(min(lo, ticks[0]), max(hi, ticks[-1]))
    ax.spines['bottom'].set_bounds(ticks[0], ticks[-1])


def plot_sorted_forest(subject_summary, parameter, xlabel=None, by='condition', order=None,
                       sort_by=None, palette=None, level_labels=None, group_summary=None,
                       ref=None, true_col=None, figsize=None, subject_col='subject'):
    """Subject-level sorted forest plot (caterpillar), one panel per level of ``by``.

    Subjects are sorted by their posterior mean under ``sort_by`` (default: the
    first level) and keep that row in every panel, so a subject can be followed
    across conditions. Dot = posterior mean, line = 95% HDI, dashed line =
    group mean (``group_summary``, matched on ``by``) or ``ref``. Use
    ``by=None`` for a parameter with one value per subject.
    """
    _need(subject_summary, [subject_col, 'mean', 'lo', 'hi'], 'subject_summary')
    d = _select_parameter(subject_summary, parameter, 'subject_summary')
    if by is None:
        d = d.assign(_panel='')
        by = '_panel'
    _need(d, [by], 'subject_summary')
    if d.duplicated([subject_col, by]).any():
        raise ValueError(f'Several rows per subject{"" if by == "_panel" else " × " + by}: add the '
                         'missing column to summarize_posterior(by=...) or set by=.')
    levels = _levels(d, by, order)
    sort_by = levels[0] if sort_by is None else sort_by
    if sort_by not in levels:
        raise ValueError(f'sort_by={sort_by!r} is not a level of {by!r}; levels are {levels}')
    ranking = d[d[by] == sort_by].sort_values('mean')[subject_col].tolist()
    unranked = set(d[subject_col]) - set(ranking)
    if unranked:
        warnings.warn(f'{len(unranked)} subject(s) have no {sort_by!r} value and are placed '
                      'at the top.', UserWarning, stacklevel=2)
        ranking += sorted(unranked, key=str)
    rank = {s: i for i, s in enumerate(ranking)}
    figsize = figsize or (min(7.25, 1.9 * len(levels) + 0.6), 2.0)
    fig, axes = plt.subplots(1, len(levels), figsize=figsize, sharex=True, sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, lev in zip(axes, levels):
        dc = d[d[by] == lev].assign(_y=lambda x: x[subject_col].map(rank))
        if by == '_panel':
            col = palette if isinstance(palette, str) else '#3B5BA5'
        else:
            col = _color(palette, lev)
        if ref is not None:
            ax.axvline(ref, color='0.6', lw=0.7, ls='--', zorder=0)
        if group_summary is not None:
            g = _select_parameter(group_summary, parameter, 'group_summary')
            if by in g.columns:
                g = g[g[by] == lev]
            if len(g) != 1:
                raise ValueError(f'group_summary must have exactly one row per panel; found {len(g)} '
                                 f'for {lev!r}')
            ax.axvline(g['mean'].iloc[0], color='0.4', lw=0.7, ls='--', zorder=0)
        ax.hlines(dc['_y'], dc['lo'], dc['hi'], color=col, lw=0.9)
        ax.plot(dc['mean'], dc['_y'], 'o', color=col, ms=2.6, zorder=3)
        if true_col is not None:
            ax.plot(dc[true_col], dc['_y'], 'o', mfc='none', mec='0.15', ms=3.4, mew=0.6, zorder=4)
        if by != '_panel':
            ax.set_title(_label(level_labels, lev), color=col)
        ax.set_xlabel(xlabel or parameter)
        ax.set_yticks([])
    sorted_txt = '' if by == '_panel' else f' by {_label(level_labels, sort_by)}'
    axes[0].set_ylabel(f'Subject (sorted{sorted_txt})')
    sns.despine(fig=fig, offset=4, trim=True, left=True)
    return fig


def _check_ppc_frames(pred, obs, x, by):
    _need(pred, [x, 'pred', 'lo', 'hi'] + ([by] if by else []), 'pred')
    _need(obs, [x, 'obs'] + ([by] if by else []), 'obs')
    keys = [x] + ([by] if by else [])
    for name, df in [('pred', pred), ('obs', obs)]:
        if df.duplicated(keys).any():
            raise ValueError(f'{name} has several rows per {keys} (a subject-level summary? '
                             'use plot_subject_ppc).')


def plot_ppc_panels(pred, obs, x, by='condition', order=None, palette=None,
                    level_labels=None, xlabel=None, ylabel=None, ref=None, xref=None,
                    figsize=None, logx=False, xticks=None, yticks=None):
    """Group-level PPC, one panel per level of ``by``: observed data as dots,
    model as line + 95% band (from :func:`bauer.utils.ppc_summary` and
    :func:`bauer.utils.ppc_observed`).

    ``ref`` / ``xref``: dotted horizontal / vertical reference lines, e.g.
    ``ref=0.5`` (chance) and ``xref=0`` (no stimulus difference) for a
    psychometric curve."""
    _check_ppc_frames(pred, obs, x, by)
    levels = _levels(pred, by, order)
    figsize = figsize or (min(7.25, 1.8 * len(levels) + 0.5), 1.9)
    fig, axes = plt.subplots(1, len(levels), figsize=figsize, sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, lev in zip(axes, levels):
        b = pred[pred[by] == lev].sort_values(x)
        o, col = obs[obs[by] == lev], _color(palette, lev)
        ax.fill_between(b[x], b['lo'], b['hi'], color=col, alpha=0.25, lw=0)
        ax.plot(b[x], b['pred'], color=col)
        ax.plot(o[x], o['obs'], 'o', color=col, ms=3.2, mec='white', mew=0.4, zorder=3)
        _ref_lines(ax, ref, xref)
        ax.set_title(_label(level_labels, lev), color=col)
        ax.set_xlabel(xlabel or x)
        _ticks(ax, logx, xticks)
    axes[0].set_ylabel(ylabel or '')
    if yticks is not None:
        axes[0].set_yticks(yticks)
    sns.despine(fig=fig, offset=4, trim=True)
    return fig


def plot_ppc_overlay(pred, obs, x, by='condition', order=None, palette=None,
                     level_labels=None, xlabel=None, ylabel=None, ref=None, xref=None,
                     figsize=(3.4, 2.3), logx=False, xticks=None, yticks=None):
    """All levels of ``by`` on one axis (model line + data dots), with a
    coloured text key in the upper left."""
    _check_ppc_frames(pred, obs, x, by)
    levels = _levels(pred, by, order)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for i, lev in enumerate(levels):
        b = pred[pred[by] == lev].sort_values(x)
        o, col = obs[obs[by] == lev], _color(palette, lev)
        ax.plot(b[x], b['pred'], color=col)
        ax.plot(o[x], o['obs'], 'o', color=col, ms=2.6, mec='white', mew=0.3, zorder=3)
        ax.text(0.03, 0.97 - i * 0.09, _label(level_labels, lev), color=col,
                transform=ax.transAxes, va='top', fontsize=7)
    _ref_lines(ax, ref, xref)
    ax.set(xlabel=xlabel or x, ylabel=ylabel or '')
    _ticks(ax, logx, xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    sns.despine(offset=4, trim=True)
    return fig


def plot_subject_ppc(pred, obs, x, subject_col='subject', col_wrap=8, palette='#3B5BA5',
                     xlabel=None, ylabel=None, logx=False, logy=False, xticks=None,
                     yticks=None, identity=False, ref=None, xref=None, height=0.95,
                     aspect=1.2):
    """Subject-level PPC as small multiples: data dots, model line + 95% band.

    ``pred`` / ``obs`` from ``ppc_summary(..., subject_level=True)`` and
    ``ppc_observed(..., subject_level=True)``. Returns the Figure.
    """
    _need(pred, [subject_col, x, 'pred', 'lo', 'hi'], 'pred')
    _need(obs, [subject_col, x, 'obs'], 'obs')
    if pred.duplicated([subject_col, x]).any():
        raise ValueError(f'pred has several rows per {[subject_col, x]}: summarise per subject '
                         'with ppc_summary(..., subject_level=True) and only x in by.')
    subjects = list(pd.unique(pred[subject_col]))
    no_obs = [s for s in subjects if not (obs[subject_col] == s).any()]
    if no_obs:
        raise ValueError(f'No observed data for {len(no_obs)} subject(s), e.g. {no_obs[:3]} '
                         f'(subject id type mismatch? pred: {pred[subject_col].dtype}, '
                         f'obs: {obs[subject_col].dtype}).')
    n = len(subjects)
    ncol = min(col_wrap, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * height * aspect, nrow * height + 0.4),
                             sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax, subj in zip(axes.flat, subjects):
        b = pred[pred[subject_col] == subj].sort_values(x)
        o = obs[obs[subject_col] == subj]
        if identity:
            lim = [b[x].min(), b[x].max()]
            ax.plot(lim, lim, color='0.75', lw=0.5, ls='--', zorder=0)
        _ref_lines(ax, ref, xref)
        ax.fill_between(b[x], b['lo'], b['hi'], color=palette, alpha=0.25, lw=0)
        ax.plot(b[x], b['pred'], color=palette, lw=1)
        ax.plot(o[x], o['obs'], 'o', color='0.15', ms=1.8, zorder=3)
        ax.set_title(f'Subject {subj}', fontsize=6)
        _ticks(ax, logx, xticks)
        _ticks(ax, logy, yticks, axis='y')
    fig.supxlabel(xlabel or x, fontsize=8)
    fig.supylabel(ylabel or '', fontsize=8)
    sns.despine(fig=fig, offset=3, trim=True)
    return fig


def _ref_lines(ax, ref=None, xref=None):
    """Dotted reference lines behind the data (horizontal at ``ref``,
    vertical at ``xref``)."""
    if ref is not None:
        ax.axhline(ref, color='0.6', lw=0.7, ls=':', zorder=0)
    if xref is not None:
        ax.axvline(xref, color='0.6', lw=0.7, ls=':', zorder=0)


def _ticks(ax, log, ticks, axis='x'):
    import matplotlib.ticker as mticker
    ax_ = ax.xaxis if axis == 'x' else ax.yaxis
    if log:
        (ax.set_xscale if axis == 'x' else ax.set_yscale)('log')
        if ticks is not None:
            ax_.set_major_locator(mticker.FixedLocator(ticks))
        ax_.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax_.set_minor_formatter(mticker.NullFormatter())
    elif ticks is not None:
        (ax.set_xticks if axis == 'x' else ax.set_yticks)(ticks)


def plot_ppc_distributions(data, sim, x, value, sim_value=None, panels=None, palette='#3B5BA5',
                           level_labels=None, xlabel=None, bins=30, figsize=None, ref=True,
                           x_format='{:.2f}'):
    """Observed vs posterior-predictive **distributions**, one panel per value of ``x``.

    Grey bars = observed histogram (density), coloured line = histogram of the
    pooled posterior-predictive samples, dotted line at the stimulus value (when
    ``ref`` and ``x`` is numeric). For continuous responses (reproductions,
    estimates, RTs) per stimulus level.

    Parameters
    ----------
    data : observed trials with columns ``x`` and ``value``.
    sim : simulated trials (e.g. ``model.ppc(...)`` joined with the stimulus
        column) with columns ``x`` and ``sim_value`` (default ``value``).
    panels : the ``x`` levels to show (default: all, sorted).
    """
    sim_value = sim_value or value
    _need(data, [x, value], 'data')
    _need(sim, [x, sim_value], 'sim')
    levels = sorted(pd.unique(data[x])) if panels is None else list(panels)
    missing = [lv for lv in levels if not (np.isclose(data[x], lv) if np.issubdtype(
        np.asarray(levels).dtype, np.number) else data[x] == lv).any()]
    if missing:
        raise ValueError(f'No observed data for {x}={missing}; levels are {sorted(pd.unique(data[x]))}')
    figsize = figsize or (min(7.25, 1.05 * len(levels) + 0.4), 1.7)
    fig, axes = plt.subplots(1, len(levels), figsize=figsize, sharey=False,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    numeric = np.issubdtype(np.asarray(levels).dtype, np.number)
    for ax, lv in zip(axes, levels):
        sel_o = np.isclose(data[x], lv) if numeric else data[x] == lv
        sel_s = np.isclose(sim[x], lv) if numeric else sim[x] == lv
        o = data.loc[sel_o, value].astype(float)
        s = sim.loc[sel_s, sim_value].astype(float)
        if s.empty:
            raise ValueError(f'No simulated data for {x}={lv!r}')
        lo, hi = np.nanpercentile(np.concatenate([o, s]), [0.5, 99.5])
        edges = np.linspace(lo, hi, bins + 1)
        ax.hist(o, bins=edges, density=True, color='0.75', lw=0)
        dens, _ = np.histogram(s, bins=edges, density=True)
        ax.plot((edges[:-1] + edges[1:]) / 2, dens, color=palette, lw=1.2)
        if ref and numeric:
            ax.axvline(lv, color='0.6', lw=0.7, ls=':', zorder=0)
        title = _label(level_labels, lv)
        ax.set_title(x_format.format(title) if numeric and title == lv else title, fontsize=7)
        ax.set_yticks([])
    axes[0].set_ylabel('Density')
    fig.supxlabel(xlabel or value, fontsize=8)
    sns.despine(fig=fig, offset=3, left=True)
    return fig


def plot_ppc_stats(pred_by_stat, obs_by_stat, x, by='condition', order=None, palette=None,
                   level_labels=None, ylabels=None, xlabel=None, figsize=None, logx=False,
                   xticks=None, refs=None):
    """Several PPC statistics side by side (one panel per statistic), all
    levels of ``by`` overlaid: data dots, model line + band.

    ``pred_by_stat`` / ``obs_by_stat``: dicts statistic name -> output of
    :func:`bauer.utils.ppc_summary` / :func:`bauer.utils.ppc_observed` for that
    statistic (e.g. mean error, SD, CV, skewness of reproductions per duration).
    ``refs``: dict statistic -> y of a dotted reference line.
    """
    stats = list(pred_by_stat)
    missing = [s for s in stats if s not in obs_by_stat]
    if missing:
        raise ValueError(f'obs_by_stat lacks {missing}')
    figsize = figsize or (min(7.25, 2.4 * len(stats)), 2.1)
    fig, axes = plt.subplots(1, len(stats), figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, stat in zip(axes, stats):
        pred, obs = pred_by_stat[stat], obs_by_stat[stat]
        _check_ppc_frames(pred, obs, x, by)
        for lev in _levels(pred, by, order):
            b = pred[pred[by] == lev].sort_values(x)
            o, col = obs[obs[by] == lev], _color(palette, lev)
            ax.fill_between(b[x], b['lo'], b['hi'], color=col, alpha=0.15, lw=0)
            ax.plot(b[x], b['pred'], color=col)
            ax.plot(o[x], o['obs'], 'o', color=col, ms=2.8, mec='white', mew=0.3, zorder=3)
        _ref_lines(ax, (refs or {}).get(stat))
        ax.set(xlabel=xlabel or x, ylabel=(ylabels or {}).get(stat, stat))
        _ticks(ax, logx, xticks)
    for i, lev in enumerate(_levels(pred_by_stat[stats[0]], by, order)):
        axes[0].text(0.03, 0.97 - i * 0.09, _label(level_labels, lev),
                     color=_color(palette, lev), transform=axes[0].transAxes, va='top',
                     fontsize=7)
    sns.despine(fig=fig, offset=4, trim=True)
    return fig
