"""Pull cluster runtimes TSV, parse, and plot backend/device comparison.

Reads ~/logs/bauer_runtimes.tsv on sciencecluster (rsync'd locally) and
produces:
  1. A parsed DataFrame with (dataset, model, flex, regression, n_subj,
     env, host, elapsed_sec) columns.
  2. A bar chart comparing pymc-CPU / numpyro-CPU / numpyro-GPU-L4 /
     numpyro-GPU-H100 wall times for the JAX experiment at N=8 and N=64.
  3. A "production fits landed" summary table.

Usage (local):
    python notebooks/runtimes_analysis.py
"""
# %%
import os, os.path as op, re, subprocess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

LOCAL_TSV = '/tmp/bauer_runtimes.tsv'
OUT = '/Users/gdehol/git/bauer/notebooks/figures/runtimes'
os.makedirs(OUT, exist_ok=True)

sns.set_theme(context='notebook', style='whitegrid', palette='deep')


def pull_tsv():
    """rsync the TSV from cluster."""
    subprocess.run(
        ['rsync', '-az', 'sciencecluster:~/logs/bauer_runtimes.tsv', LOCAL_TSV],
        check=True,
    )


def parse_jobname(name):
    """Split 'jax_exp_numpyro_gpu_n8' / 'garcia_rdm_flex' / 'tms_choice_reg'
    into structured fields.
    """
    info = {'job_class': 'production', 'backend': None, 'device': None,
            'dataset': None, 'model': None, 'flex': False, 'regression': False,
            'n_subj_hint': None}

    parts = name.split('_')
    if name.startswith('jax_exp_'):
        info['job_class'] = 'jax_experiment'
        # forms: jax_exp_pymc_cpu_n8, jax_exp_numpyro_gpu_l4_n64, ...
        for token in parts[2:]:
            if token in ('pymc', 'numpyro', 'blackjax'):
                info['backend'] = token
            elif token == 'cpu':
                info['device'] = 'cpu'
            elif token == 'gpu':
                info['device'] = 'gpu'
            elif token in ('l4', 'h100'):
                info['device'] = f'gpu-{token.upper()}'
            elif re.match(r'^n\d+$', token):
                info['n_subj_hint'] = int(token[1:])
        info['dataset'] = 'garcia'
        info['model'] = 'ddm'
        return info

    # Production: e.g. garcia_rdm_flex / tms_ddm_reg / dh_dc_choice
    if name.startswith('garcia_'):
        info['dataset'] = 'garcia'
        rest = name[len('garcia_'):]
    elif name.startswith('dh_dc_'):
        info['dataset'] = 'dehollander_dotcloud'
        rest = name[len('dh_dc_'):]
    elif name.startswith('dh_sym_'):
        info['dataset'] = 'dehollander_symbolic'
        rest = name[len('dh_sym_'):]
    elif name.startswith('tms_'):
        info['dataset'] = 'tms'
        rest = name[len('tms_'):]
    else:
        info['model'] = name
        return info

    if rest.endswith('_reg'):
        info['regression'] = True
        rest = rest[:-4]
    if '_flex' in rest:
        info['flex'] = True
        rest = rest.replace('_flex', '')
    if rest.startswith('rdm_flex_so'):
        info['model'] = 'rdm'; info['flex'] = True
    elif rest in ('choice', 'ddm', 'rdm'):
        info['model'] = rest
    else:
        info['model'] = rest
    return info


def load():
    if not op.exists(LOCAL_TSV):
        pull_tsv()
    df = pd.read_csv(LOCAL_TSV, sep='\t')
    if df.empty:
        return df
    parsed = pd.DataFrame([parse_jobname(j) for j in df['job_name']])
    df = pd.concat([df, parsed], axis=1)
    df['elapsed_min'] = df['elapsed_sec'] / 60
    return df


# %% Pull + show
pull_tsv()
df = load()
print(f'{len(df)} rows')
if df.empty:
    print('TSV empty — wait for fits to land.')
else:
    print(df[['job_name', 'env', 'device', 'backend', 'elapsed_min', 'exit']].to_string(index=False))


# %% JAX experiment: backend comparison at N=8 / N=64
exp = df[df['job_class'] == 'jax_experiment'].copy() if not df.empty else pd.DataFrame()
if len(exp) >= 2:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    label = exp['backend'].astype(str) + '-' + exp['device'].astype(str)
    sns.barplot(data=exp.assign(label=label), x='label', y='elapsed_min',
                 hue='n_subj_hint', ax=ax)
    ax.set_xlabel('backend × device'); ax.set_ylabel('elapsed (min)')
    ax.set_title('Garcia DDM wall time across backends')
    plt.tight_layout()
    fig.savefig(op.join(OUT, 'jax_experiment_bars.png'), dpi=140)
    plt.close(fig)
    print(f'jax experiment bar chart saved')
else:
    print('jax experiment has <2 rows — not enough for a comparison plot yet.')


# %% Production fits landed
prod = df[df['job_class'] == 'production'].copy() if not df.empty else pd.DataFrame()
if not prod.empty:
    print('\n=== production fits ===')
    print(prod[['dataset', 'model', 'flex', 'regression',
                'env', 'device', 'elapsed_min']].to_string(index=False))
