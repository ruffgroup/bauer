from . import cluster_offers
from .bayes import summarize_ppc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az

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

def plot_subjectwise_parameters(idata, parameter, sort_subjects=True, plot_group_mean=True, **kwargs):

    d = idata.posterior[parameter].to_dataframe()
    means = d.groupby('subject').mean()
    means.columns = pd.MultiIndex.from_tuples([(parameter, 'mean')])

    # Get 95% CI using arviz
    cis = d[parameter].groupby('subject').apply(get_hdi).unstack(-1)
    cis.columns = pd.MultiIndex.from_tuples([(parameter, 'hdi_low'), (parameter, 'hdi_high')])
    print(cis)
    summarized_pars = means.join(cis)

    # summarized_pars.sort_values((parameter, 'mean'), inplace=True)

    if sort_subjects:
        summarized_pars['mean_order'] = summarized_pars[(parameter, 'mean')].rank()
        x = 'mean_order'
    else:
        x = 'subject'
        summarized_pars = summarized_pars.reset_index()

    # Plot the results
    g = sns.scatterplot(x=x, y=(parameter, 'mean'), data=summarized_pars, color='k')
    plt.errorbar(x=summarized_pars[x], y=summarized_pars[(parameter, 'mean')], yerr=[summarized_pars[(parameter, 'mean')] - summarized_pars[(parameter, 'hdi_low')], summarized_pars[(parameter, 'hdi_high')] - summarized_pars[(parameter, 'mean')]], fmt='o', color='k',)

    g.set_ylabel(parameter)
    g.set_xlabel('Subject')
    
    # plot errorbars around the scatters using hd_low and hdi_high
    sns.despine()

    if plot_group_mean:
        group_mean = idata.posterior[parameter+'_mu'].to_dataframe()
        plt.axhline(group_mean[parameter+'_mu'].mean(), c='k', ls='--')