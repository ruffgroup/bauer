"""Posterior and posterior-predictive summaries in long (tidy) format.

The convention throughout: draws live in a long DataFrame with one row per
posterior draw (columns ``chain``, ``draw``, grouping columns such as
``parameter``, ``subject``, ``condition``, and the number in ``value``).
Everything is summarised as the **posterior mean** plus a **95% highest-density
interval** (HDI).

Typical use with a regression model::

    conds = pd.DataFrame({'condition': ['placebo', '5 mg', '10 mg'], 'session_c': 0.0})
    draws = model.conditionwise_draws(idata, conds, parameters=['nu'])
    group = summarize_posterior(draws, ['parameter', 'condition'])
    effects = summarize_posterior(contrast_draws(draws, 'condition', 'placebo'),
                                  ['parameter', 'condition'])

All functions check their input and raise a ``ValueError`` that says what is
wrong rather than returning an empty or silently averaged result.
"""
import warnings

import arviz as az
import numpy as np
import pandas as pd

_DRAW_KEYS = ('parameter', 'subject', 'chain', 'draw')


def _as_list(x):
    return [x] if isinstance(x, str) else list(x)


def _require(df, cols, what):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing column(s) {missing}; it has {list(df.columns)}")


def summarize_posterior(draws, by, value='value', hdi_prob=0.95, min_draws=100):
    """Posterior mean, 95% HDI and P(> 0) per group of a long draws DataFrame.

    Parameters
    ----------
    draws : DataFrame with one row per posterior draw (e.g. from
        ``model.conditionwise_draws``, :func:`contrast_draws` or
        :func:`posterior_draws`).
    by : column name(s) to summarise per, e.g. ``['parameter', 'condition']``.
        Every column that distinguishes rows *within* a draw must be listed,
        otherwise different quantities are pooled into one summary.

    Returns
    -------
    DataFrame with the ``by`` columns plus ``mean``, ``lo``, ``hi`` (HDI
    bounds) and ``p_positive`` (posterior probability that the value is above
    zero; meaningful for contrasts).
    """
    by = _as_list(by)
    _require(draws, by + [value], 'draws')
    if draws[value].isna().any():
        raise ValueError(f"{int(draws[value].isna().sum())} NaN values in {value!r}; "
                         'a NaN draw usually means a failed transform or a bad merge.')
    if draws[by].isna().any().any():
        bad = draws[by].columns[draws[by].isna().any()].tolist()
        raise ValueError(f'NaN in grouping column(s) {bad}: those rows would be dropped. '
                         'Summarise them separately.')
    sizes = draws.groupby(by, observed=True, sort=False).size()
    if sizes.min() < min_draws:
        warnings.warn(f'Only {sizes.min()} draws in some group; the {hdi_prob:.0%} HDI is '
                      'unreliable with fewer than ~100 draws.', UserWarning, stacklevel=2)
    if 'chain' in draws.columns and 'draw' in draws.columns:
        per = draws.groupby(by + ['chain', 'draw'], observed=True, sort=False).size()
        if per.max() > 1:
            warnings.warn('Several rows per draw within a group: summarize_posterior is '
                          f'pooling different quantities. Add the distinguishing column to by={by}.',
                          UserWarning, stacklevel=2)

    def _one(x):
        x = np.asarray(x, dtype=float)
        lo, hi = az.hdi(x, hdi_prob=hdi_prob)
        return pd.Series({'mean': x.mean(), 'lo': lo, 'hi': hi,
                          'p_positive': (x > 0).mean()})

    return (draws.groupby(by, observed=True, sort=False)[value]
                 .apply(_one).unstack().reset_index())


def contrast_draws(draws, column, reference, value='value', keys=None):
    """Per-draw differences from a reference level of ``column``.

    ``contrast_draws(draws, 'condition', 'placebo')`` gives, for every draw
    (and every parameter / subject), each condition minus placebo. The
    reference level itself is dropped. Level order is preserved.

    Rows are matched on ``keys`` (default: whichever of ``parameter``,
    ``subject``, ``chain``, ``draw`` are present). Any other column must be
    constant (it is dropped) or listed in ``keys``; a column that varies, such
    as ``session_c`` with different values per condition, raises.
    """
    _require(draws, [column, value, 'chain', 'draw'], 'draws')
    keys = list(keys) if keys is not None else [k for k in _DRAW_KEYS if k in draws.columns]
    levels = list(pd.unique(draws[column]))
    if reference not in levels:
        raise ValueError(f'reference {reference!r} is not a level of {column!r}; levels are {levels}')
    others = [c for c in draws.columns if c not in keys + [column, value]]
    varying = [c for c in others if draws[c].nunique(dropna=False) > 1]
    if varying:
        raise ValueError(f'Column(s) {varying} vary between rows but are not part of the '
                         f'contrast. Drop them (e.g. draws.drop(columns={varying})) or pass '
                         f'keys=[...] including them.')
    if draws.duplicated(keys + [column]).any():
        raise ValueError(f'Several rows per {keys + [column]}: the contrast would average them. '
                         'Check for duplicate condition rows or concatenated fits '
                         '(add a distinguishing column to keys).')
    wide = draws.pivot(index=keys, columns=column, values=value)
    out = wide.sub(wide[reference], axis=0).drop(columns=reference)
    out = out.reset_index().melt(id_vars=keys, var_name=column, value_name=value)
    out[column] = pd.Categorical(out[column], [lv for lv in levels if lv != reference])
    if 'parameter' in out.columns:
        out['parameter'] = pd.Categorical(out['parameter'], list(pd.unique(draws['parameter'])))
    out = out.sort_values([k for k in ['parameter', column] if k in out.columns], kind='stable')
    for c in [column, 'parameter']:
        if c in out.columns:
            out[c] = out[c].astype(object)
    return out.reset_index(drop=True)


def posterior_draws(idata, var_names, transform=None):
    """Long draws for plain (non-regression) variables, e.g. ``'nu'`` with
    dims ``(chain, draw, subject)`` or the group mean ``'nu_mu'``.

    For regression parameters (a ``*_regressors`` dimension) use
    ``model.conditionwise_draws`` instead: the raw coefficients are on the
    untransformed scale. ``transform`` is an optional callable applied to the
    values.
    """
    frames = []
    for v in _as_list(var_names):
        if v not in idata.posterior:
            raise ValueError(f'{v!r} not in the posterior; variables are {list(idata.posterior.data_vars)}')
        dims = [d for d in idata.posterior[v].dims if d.endswith('_regressors')]
        if dims:
            raise ValueError(f'{v!r} is a regression parameter (dimension {dims[0]}); its values '
                             'are untransformed coefficients. Use model.conditionwise_draws().')
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
    ``interval`` over samples (``lo``, ``hi``). The band contains parameter
    uncertainty **and** sampling noise, so ~95% of observed cells should fall
    inside it (see :func:`ppc_coverage`).

    ``ppc`` is long: one row per trial × sample, e.g. ``model.ppc(...)`` after
    ``reset_index()`` and joining the condition columns.
    """
    by = _as_list(by)
    ppc = ppc.reset_index() if subject_col not in ppc.columns else ppc
    _require(ppc, [subject_col, sample_col, value] + by, 'ppc')
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
    by = _as_list(by)
    data = data.reset_index() if subject_col not in data.columns else data
    _require(data, [subject_col, value] + by, 'data')
    out = data.groupby([subject_col] + by, observed=True)[value].mean()
    if not subject_level:
        out = out.groupby(by, observed=True).mean()
    return out.rename('obs').reset_index()


def ppc_coverage(pred, obs, by):
    """Fraction of observed cells inside the predictive interval.

    ``by`` must identify cells uniquely in both frames (pass ``['subject', ...]``
    for subject-level summaries), and every cell must be in both.
    """
    by = _as_list(by)
    _require(pred, by + ['lo', 'hi'], 'pred')
    _require(obs, by + ['obs'], 'obs')
    for name, df in [('pred', pred), ('obs', obs)]:
        if df.duplicated(by).any():
            raise ValueError(f'{name} has several rows per {by}: add the missing key '
                             "(e.g. 'subject' for subject-level summaries).")
    m = pred.merge(obs, on=by, how='outer', indicator=True)
    unmatched = int((m['_merge'] != 'both').sum())
    if unmatched:
        raise ValueError(f'{unmatched} cells are in only one of pred / obs on {by} '
                         '(different levels, or float/int/str key mismatch).')
    return float(((m['obs'] >= m['lo']) & (m['obs'] <= m['hi'])).mean())
