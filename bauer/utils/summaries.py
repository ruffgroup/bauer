"""Posterior and posterior-predictive summaries in long (tidy) format.

The convention throughout: draws live in a long DataFrame with one row per
posterior draw (columns ``chain``, ``draw``, optional grouping columns such as
``parameter``, ``subject``, ``condition``, and the number in ``value``).
Everything is summarised as the **posterior mean** plus a **95% highest-density
interval** (HDI).

Typical use with a regression model::

    conds = pd.DataFrame({'condition': ['placebo', '5 mg', '10 mg'], 'session_c': 0.0})
    draws = model.conditionwise_draws(idata, conds, parameters=['nu'])
    group = summarize_posterior(draws, ['parameter', 'condition'])
    effects = summarize_posterior(contrast_draws(draws, 'condition', 'placebo'),
                                  ['parameter', 'condition'])
"""
import arviz as az
import numpy as np
import pandas as pd


def summarize_posterior(draws, by, value='value', hdi_prob=0.95):
    """Posterior mean, HDI and P(> 0) per group of a long draws DataFrame.

    Returns a DataFrame with the ``by`` columns plus ``mean``, ``lo``, ``hi``
    (HDI bounds) and ``p_positive`` (posterior probability that the value is
    above zero; useful for contrasts).
    """
    by = [by] if isinstance(by, str) else list(by)

    def _one(x):
        x = np.asarray(x, dtype=float)
        lo, hi = az.hdi(x, hdi_prob=hdi_prob)
        return pd.Series({'mean': x.mean(), 'lo': lo, 'hi': hi,
                          'p_positive': (x > 0).mean()})

    return (draws.groupby(by, observed=True, sort=False)[value]
                 .apply(_one).unstack().reset_index())


def contrast_draws(draws, column, reference, value='value'):
    """Per-draw differences from a reference level of ``column``.

    E.g. ``contrast_draws(draws, 'condition', 'placebo')`` gives, for every
    draw (and every parameter / subject in ``draws``), each condition minus
    placebo. The reference level itself is dropped.
    """
    keys = [c for c in draws.columns if c not in (column, value)]
    wide = draws.pivot_table(index=keys, columns=column, values=value, observed=True)
    out = wide.sub(wide[reference], axis=0).drop(columns=reference)
    out.columns.name = column
    return out.stack().rename(value).reset_index()


def posterior_draws(idata, var_names, transform=None):
    """Long draws for plain (non-regression) parameters, e.g. ``'nu'`` with
    dims ``(chain, draw, subject)`` or the group mean ``'nu_mu'``.

    ``transform`` is an optional callable applied to the values.
    """
    var_names = [var_names] if isinstance(var_names, str) else list(var_names)
    frames = []
    for v in var_names:
        d = idata.posterior[v].to_dataframe().reset_index().rename(columns={v: 'value'})
        if transform is not None:
            d['value'] = transform(d['value'].values)
        frames.append(d.assign(parameter=v))
    return pd.concat(frames, ignore_index=True)


# ─── Posterior predictive checks ─────────────────────────────────────────────

def ppc_summary(ppc, by, value, subject_col='subject', sample_col='ppc_sample',
                subject_level=False, interval=0.95):
    """Summarise simulated data the same way as the observed data.

    For every posterior-predictive sample: average ``value`` within each
    subject × ``by`` cell, then (unless ``subject_level``) across subjects.
    Returns, per cell, the mean over samples (``pred``) and the central
    ``interval`` over samples (``lo``, ``hi``). The band therefore contains
    parameter uncertainty **and** sampling noise, so ~95% of observed points
    should fall inside it.

    ``ppc`` is long: one row per trial × sample (``model.ppc(...)`` output after
    ``reset_index()`` and joining the condition columns).
    """
    by = [by] if isinstance(by, str) else list(by)
    per_draw = ppc.groupby([subject_col] + by + [sample_col], observed=True)[value].mean()
    keys = [subject_col] + by
    if not subject_level:
        per_draw = per_draw.groupby(by + [sample_col], observed=True).mean()
        keys = by
    g = per_draw.groupby(keys, observed=True)
    q = (1 - interval) / 2
    return pd.DataFrame({'pred': g.mean(), 'lo': g.quantile(q),
                         'hi': g.quantile(1 - q)}).reset_index()


def ppc_observed(data, by, value, subject_col='subject', subject_level=False):
    """The observed counterpart of :func:`ppc_summary` (column ``obs``)."""
    by = [by] if isinstance(by, str) else list(by)
    out = data.groupby([subject_col] + by, observed=True)[value].mean()
    if not subject_level:
        out = out.groupby(by, observed=True).mean()
    return out.rename('obs').reset_index()


def ppc_coverage(pred, obs, by):
    """Fraction of observed cells inside the predictive interval."""
    by = [by] if isinstance(by, str) else list(by)
    m = pred.merge(obs, on=by)
    return float(((m['obs'] >= m['lo']) & (m['obs'] <= m['hi'])).mean())
