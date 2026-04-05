import pandas as pd
import numpy as np
import pytensor.tensor as pt
import arviz as az
from scipy.special import expit

def get_posterior(mu1, sd1, mu2, sd2):
    var1, var2 = sd1**2, sd2**2
    return mu1 + (var1/(var1+var2))*(mu2 - mu1), pt.sqrt((var1*var2)/(var1+var2))

def get_posterior_np(mu1, sd1, mu2, sd2):
    var1, var2 = sd1**2, sd2**2
    return mu1 + (var1/(var1+var2))*(mu2 - mu1), np.sqrt((var1*var2)/(var1+var2))

def get_diff_dist(mu1, sd1, mu2, sd2):
    return mu2 - mu1, pt.sqrt(sd1**2+sd2**2)

def get_diff_dist_np(mu1, sd1, mu2, sd2):
    return mu2 - mu1, np.sqrt(sd1**2+sd2**2)

def cumulative_normal(x, mu, sd, s=pt.sqrt(2.)):
#     Cumulative distribution function for the standard normal distribution
    return pt.clip(0.5 + 0.5 *
                   pt.erf((x - mu) / (sd*s)), 1e-9, 1-1e-9)

def softplus(x): 
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0) 

def logistic(x): 
    return expit(x)

def summarize_ppc(ppc, groupby=None):
    """Single-step PPC summary (legacy).  Prefer summarize_ppc_group for group-level PPCs."""
    if groupby is not None:
        ppc = ppc.groupby(groupby).mean()

    e = ppc.mean(1).to_frame('p_predicted')
    hdi = pd.DataFrame(az.hdi(ppc.T.values), index=ppc.index,
                       columns=['hdi025', 'hdi975'])

    return pd.concat((e, hdi), axis=1)


def summarize_ppc_group(ppc, condition_cols, subject_col='subject', hdi_prob=0.95):
    """Two-step group-mean PPC summary.

    For each posterior sample, first averages simulated choices within each
    (subject, condition) cell, then averages those subject means across subjects.
    The HDI is then computed across the resulting per-sample group means.

    This correctly represents "where 95 % of plausible group-mean observations
    would fall under the model" — including both parameter uncertainty and
    Bernoulli noise — rather than the uncertainty on the per-trial probability.

    Parameters
    ----------
    ppc : pd.DataFrame, shape (trials, posterior_samples)
        Raw PPC samples from ``model.ppc().xs(var_name, level='variable')``.
        Posterior-sample columns can be any column not in *condition_cols* or
        *subject_col*.  All grouping keys (including *subject_col*) must be
        present either as index levels or as regular columns.
    condition_cols : list of str
        Condition dimensions to summarise over, e.g.
        ``['n1', 'log_ratio_bin']`` or
        ``['order', 'n_safe_bin', 'log_ratio_bin']``.
        These must be index levels or columns of *ppc*.
    subject_col : str
        Name of the subject dimension (default ``'subject'``).
    hdi_prob : float
        Posterior mass for HDI interval (default 0.95).

    Returns
    -------
    pd.DataFrame
        Index: *condition_cols*.
        Columns: ``p_predicted`` (posterior mean), ``hdi025``, ``hdi975``.

    Examples
    --------
    >>> ppc_ll = model.ppc(data, idata, var_names=['ll_bernoulli'])
    ...             .xs('ll_bernoulli', level='variable')
    >>> # add derived condition columns if needed
    >>> ppc_ll_flat = ppc_ll.reset_index()
    >>> ppc_ll_flat['bin'] = pd.cut(ppc_ll_flat['log(n2/n1)'], 12) \
    ...                        .apply(lambda x: x.mid).astype(float)
    >>> summary = summarize_ppc_group(ppc_ll_flat,
    ...                               condition_cols=['n1', 'bin'])
    >>> # summary has columns p_predicted, hdi025, hdi975 indexed by (n1, bin)
    """
    # After ppc_ll.reset_index(), pandas adds former-index columns as tuples
    # (name, '') because ppc_ll.columns is a MultiIndex(chain, draw).
    # Flatten those back to plain strings so the isinstance check works.
    if isinstance(ppc.columns, pd.MultiIndex):
        ppc = ppc.copy()
        ppc.columns = [c[0] if (isinstance(c, tuple) and len(c) >= 2 and c[1] == '') else c
                       for c in ppc.columns]

    # Posterior-sample columns have tuple names (chain, draw) — not strings.
    # Paradigm columns are strings.  Exclude condition and subject columns too.
    excluded = set(condition_cols) | {subject_col}
    sample_cols = [c for c in ppc.columns
                   if c not in excluded and not isinstance(c, str)]
    if not sample_cols:
        # Fallback when caller already renamed sample cols to integers
        sample_cols = ppc.select_dtypes('number').columns.difference(
            list(excluded)).tolist()

    # Step 1: average within (subject, condition) for each posterior sample
    ppc_subj = ppc.groupby([subject_col] + condition_cols)[sample_cols].mean()

    # Step 2: average subject means within each condition for each sample
    ppc_group = ppc_subj.groupby(condition_cols).mean()

    # Step 3: mean and HDI across samples
    e = ppc_group.mean(axis=1).to_frame('p_predicted')
    hdi_vals = pd.DataFrame(
        az.hdi(ppc_group.T.values, hdi_prob=hdi_prob),
        index=ppc_group.index,
        columns=['hdi025', 'hdi975'],
    )
    return pd.concat([e, hdi_vals], axis=1)


def get_posterior_np(mu1, sd1, mu2, sd2):
    var1, var2 = sd1**2, sd2**2
    return mu1 + (var1/(var1+var2))*(mu2 - mu1), np.sqrt((var1*var2)/(var1+var2))