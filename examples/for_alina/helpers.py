"""Shared plotting / summary helpers for the example notebooks.

All summaries use the **posterior mean** as the point estimate (and a 94 %
HDI for the interval). Natural-scale extraction goes through
``model.get_conditionwise_parameters``, which applies the link function.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az


GAIN, LOSS = '#1f78b4', '#d95f02'
CONDITIONS = pd.DataFrame({'domain': ['gain', 'loss']})

# Risk-neutral indifference point in log(risky/safe) space.
# With p_risky = 0.55, log(1/p_risky) ≈ 0.598.
EV_NEUTRAL_LOG = float(np.log(1 / 0.55))


def mark_ev_neutral(ax, label='EV-neutral'):
    """Vertical dashed line at the risk-neutral indifference log-ratio."""
    ax.axvline(EV_NEUTRAL_LOG, ls=':', c='#a50026', lw=1.5, label=label)


def prepare(df):
    """Flip ``choice`` on loss trials so the gain/loss decision shares a
    drift direction. Keeps ``n1, n2, p1, p2`` in presentation order.
    """
    out = df.copy()
    is_loss = out['domain'] == 'loss'
    out.loc[is_loss, 'choice'] = ~out.loc[is_loss, 'choice'].astype(bool)
    return out


def natural_summary(model, idata, params, conditions=CONDITIONS):
    """Tidy long-form: parameter | domain | mean | hdi_lo | hdi_hi."""
    cw = model.get_conditionwise_parameters(idata, conditions, group=True)
    rows = []
    for p in params:
        s = cw.xs(p, level='parameter')
        for dom in conditions['domain']:
            v = s[dom].values
            lo, hi = az.hdi(v, hdi_prob=0.94)
            rows.append({'parameter': p, 'domain': dom,
                         'mean': v.mean(), 'hdi_lo': lo, 'hdi_hi': hi})
    return pd.DataFrame(rows)


def per_subject_summary(model, idata, params, conditions=CONDITIONS):
    """Tidy long-form: parameter | domain | subject | mean | hdi_lo | hdi_hi."""
    cw = model.get_conditionwise_parameters(idata, conditions, group=False)
    rows = []
    for p in params:
        s = cw.xs(p, level='parameter')
        for subj in s.index.get_level_values('subject').unique():
            sub = s.xs(subj, level='subject')
            for dom in sub.columns:
                v = sub[dom].values
                lo, hi = az.hdi(v, hdi_prob=0.94)
                rows.append({'parameter': p, 'domain': dom, 'subject': subj,
                             'mean': v.mean(), 'hdi_lo': lo, 'hdi_hi': hi})
    return pd.DataFrame(rows)


def plot_forest(summary, title, scale_hints=None):
    params = summary['parameter'].unique()
    fig, axes = plt.subplots(1, len(params), figsize=(3.8 * len(params), 3.2))
    if len(params) == 1: axes = [axes]
    for ax, p in zip(axes, params):
        d = summary[summary['parameter'] == p]
        for i, dom in enumerate(['gain', 'loss']):
            row = d[d['domain'] == dom].iloc[0]
            colour = GAIN if dom == 'gain' else LOSS
            ax.errorbar(row['mean'], i,
                        xerr=[[row['mean']-row['hdi_lo']],
                              [row['hdi_hi']-row['mean']]],
                        fmt='o', ms=10, color=colour, capsize=4, lw=2.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Gain', 'Loss'])
        ax.set_xlabel(scale_hints.get(p, p) if scale_hints else p)
        ax.set_title(p); ax.invert_yaxis()
        sns.despine(ax=ax, left=True); ax.tick_params(left=False)
    fig.suptitle(title, y=1.04); plt.tight_layout()


def plot_contrast(model, idata, params, title, conditions=CONDITIONS):
    """KDE of the loss − gain contrast on the natural scale for each param."""
    cw = model.get_conditionwise_parameters(idata, conditions, group=True)
    fig, axes = plt.subplots(1, len(params), figsize=(3.8 * len(params), 3.2))
    if len(params) == 1: axes = [axes]
    for ax, p in zip(axes, params):
        s = cw.xs(p, level='parameter')
        contrast = (s['loss'] - s['gain']).values
        mean = contrast.mean()
        lo, hi = az.hdi(contrast, hdi_prob=0.94)
        az.plot_kde(contrast, plot_kwargs={'color': '#525252', 'lw': 2}, ax=ax)
        ax.axvline(0,    ls=':', c='#a50026', lw=1.5)
        ax.axvline(mean, ls='-', c='#525252', lw=1)
        ax.set_title(f'{p}\nMean={mean:+.3f}, 94 % HDI [{lo:+.3f}, {hi:+.3f}]',
                     fontsize=10)
        ax.set_xlabel('Loss − gain'); sns.despine(ax=ax)
    fig.suptitle(title, y=1.06); plt.tight_layout()


def plot_subject_panels(subj_df, title):
    """One panel per parameter; x = subject, y = mean ± 94 % HDI, color = domain."""
    g = sns.FacetGrid(subj_df, col='parameter', col_wrap=3, height=3.2,
                      aspect=1.2, sharey=False)
    def _draw(data, **kwargs):
        ax = plt.gca()
        subjects = sorted(data['subject'].unique())
        for dom in ['gain', 'loss']:
            d = data[data['domain'] == dom].set_index('subject').loc[subjects]
            xs = np.arange(len(subjects)) + 0.15 * (1 if dom == 'loss' else -1)
            colour = GAIN if dom == 'gain' else LOSS
            ax.errorbar(xs, d['mean'],
                        yerr=[d['mean']-d['hdi_lo'], d['hdi_hi']-d['mean']],
                        fmt='o', ms=8, capsize=3, color=colour, label=dom)
        ax.set_xticks(np.arange(len(subjects))); ax.set_xticklabels(subjects)
        if not ax.get_legend(): ax.legend(title='Domain', fontsize=8)
        sns.despine(ax=ax)
    g.map_dataframe(_draw)
    g.set_axis_labels('Subject', 'Posterior mean (94 % HDI)')
    g.set_titles(col_template='{col_name}')
    g.figure.suptitle(title, y=1.04)
    plt.tight_layout()


def per_subject_natural_from_files(model_factory, subject_idata_paths,
                                     params, conditions=CONDITIONS):
    """Walk a dict ``{subject: path_to_subject.nc}`` of non-hierarchical
    per-subject fits, build a model per subject (via ``model_factory(df_subj)``
    where df_subj is each subject's data — passed by the caller; this helper
    just owns the per-condition extraction), and return the same
    long-form table as :func:`per_subject_summary`:

        parameter | domain | subject | mean | hdi_lo | hdi_hi

    The ``model_factory`` must return a built model whose
    ``get_conditionwise_parameters(idata, conditions, group=False)`` call
    is valid (i.e. ``build_estimation_model`` already called on the
    subject's data with ``hierarchical=False``).
    """
    import arviz as az
    rows = []
    for subj, path in subject_idata_paths.items():
        idata = az.from_netcdf(path)
        model = model_factory(subj)
        # Non-hierarchical → variables are bare ``parameter`` (no _mu)
        cw = model.get_conditionwise_parameters(idata, conditions, group=False)
        for p in params:
            s = cw.xs(p, level='parameter')
            for dom in conditions['domain']:
                v = s[dom].values
                lo, hi = az.hdi(v, hdi_prob=0.94)
                rows.append({'parameter': p, 'domain': dom, 'subject': subj,
                             'mean': v.mean(), 'hdi_lo': lo, 'hdi_hi': hi})
    return pd.DataFrame(rows)


def chose_risky(df, choice_col='choice', flipped=False):
    """Return a Series: True iff subject chose the risky/gamble option.

    Set ``flipped=True`` when ``choice_col`` is already loss-flipped (the
    model's frame). The returned Series is on the *actual* "chose risky"
    axis, suitable for plotting against ``log(n_risky/n_safe)``.
    """
    risky_first = df['p1'] < df['p2']
    raw_chose_2 = df[choice_col].astype(bool)
    if flipped:
        # Loss trials were flipped before fitting; un-flip them.
        raw_chose_2 = np.where(df['domain'] == 'loss', ~raw_chose_2, raw_chose_2)
    # chose risky ⟺ chose option 2 XOR (risky was option 1)
    return pd.Series(raw_chose_2 ^ risky_first.values, index=df.index)


def log_risky_safe(df):
    """log(n_risky / n_safe) regardless of presentation order."""
    return np.where(df['p1'] < df['p2'],
                    np.log(df['n1'] / df['n2']),
                    np.log(df['n2'] / df['n1']))
