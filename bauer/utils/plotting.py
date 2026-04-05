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

    if exp_type  == 'magnitude':
        x = 'log(n2/n1)'

        if 'log(n2/n1)' not in ppc.index.names:
            if 'frac'  in ppc.index.names:
                ppc['log(n2/n1)'] = np.log(ppc.index.get_level_values('frac'))
            else:
                ppc['log(n2/n1)'] = np.log(ppc.index.get_level_values('n2')) - np.log(ppc.index.get_level_values('n1'))

            ppc.set_index('log(n2/n1)', append=True)

    assert (var_name in ['p', 'll_bernoulli'])

    ppc = ppc.xs(var_name, 0, 'variable').copy()


    df = df.copy()

    # Make sure that we group data from (Roughly) same fractions
    if not (df.groupby(['subject', x]).size().groupby('subject').size() < (n_clusters+1)).all():
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
        ppc = ppc.groupby(['subject']+groupby).mean()

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
    lows  = np.percentile(samples_flat, lo_prob,  axis=0)
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
                'lo':   np.percentile(samp_flat[:, si], lo_p),
                'hi':   np.percentile(samp_flat[:, si], hi_p),
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
    los   = data[lo_col].values
    his   = data[hi_col].values

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