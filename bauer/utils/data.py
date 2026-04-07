from importlib.resources import files
import os
import os.path as op
import pandas as pd
import numpy as np

def load_garcia2022(task='magnitude', remove_non_responses=True):
    """Return behavioral data from Barreto Garcia et al. (2022) as a multi-indexed DataFrame.

    Parameters
    ----------
    task : {'magnitude', 'risk'}
        Which task dataset to load.
    remove_non_responses : bool
        If True, drop trials with missing choices and cast the ``choice`` column to bool.
    """

    if task == 'magnitude':
        fn = 'garcia2022_magnitude.csv'
    elif task == 'risk':
        fn = 'garcia2022_risk.csv'

    with (files(__package__) / f'../data/{fn}').open('rb') as f:
        df = pd.read_csv(f, index_col=[0, 1, 2, 3])

    if remove_non_responses:
        df = df[~df['choice'].isnull()]
        df['choice'] = df['choice'].astype(bool)
    # df['log(n2/n1)'] = np.log(df['n2'] / df['n1'])
    # df['trial_nr'] = df.groupby(['subject'], group_keys=False).apply(lambda d: pd.Series(np.arange(len(d))+1, index=d.index))
    return df


def load_dehollander2024(task='dotcloud', sessions=None,
                         bids_folder='/data/ds-risk',
                         symbolic_folder='/data/ds-symbolicrisk',
                         remove_non_responses=True):
    """Return behavioral data from de Hollander et al. (2024) as a multi-indexed DataFrame.

    Parameters
    ----------
    task : {'dotcloud', 'symbolic'}
        Which task dataset to load. ``'dotcloud'`` uses the fMRI dot-cloud gamble
        task (ds-risk); ``'symbolic'`` uses the Arabic-numeral behavioural task
        (ds-symbolicrisk).
    sessions : list of str or None
        Sessions to include for the dotcloud task (default ``['3t2', '7t2']``).
        Ignored for the symbolic task.
    bids_folder : str
        Root of the ds-risk BIDS dataset.
    symbolic_folder : str
        Root of the ds-symbolicrisk dataset.
    remove_non_responses : bool
        If True, drop trials with missing choices and cast the ``choice`` column
        to bool.
    """
    if task == 'dotcloud':
        return _load_dehollander2024_dotcloud(
            sessions=sessions,
            bids_folder=bids_folder,
            remove_non_responses=remove_non_responses,
        )
    elif task == 'symbolic':
        return _load_dehollander2024_symbolic(
            symbolic_folder=symbolic_folder,
            remove_non_responses=remove_non_responses,
        )
    else:
        raise ValueError(f"task must be 'dotcloud' or 'symbolic', got {task!r}")


def _load_dehollander2024_dotcloud(sessions=None, bids_folder='/data/ds-risk',
                                    remove_non_responses=True):
    if sessions is None:
        sessions = ['3t2', '7t2']

    # sub-02 … sub-32, excluding sub-24
    subject_ids = ['%02d' % i for i in range(2, 33)]
    subject_ids.remove('24')

    rows = []
    for subject in subject_ids:
        for session in sessions:
            n_runs = 8  # sessions ending in '2' always have 8 runs
            for run in range(1, n_runs + 1):
                fn = op.join(
                    bids_folder,
                    f'sub-{subject}', f'ses-{session}', 'func',
                    f'sub-{subject}_ses-{session}_task-task_run-{run}_events.tsv',
                )
                if not op.exists(fn):
                    continue
                d = pd.read_csv(fn, sep='\t', index_col=['trial_nr', 'trial_type'])
                d = d.unstack('trial_type')

                n1 = d[('n1', 'stimulus 1')]
                n2 = d[('n2', 'stimulus 1')]
                p1 = d[('prob1', 'stimulus 1')]
                p2 = d[('prob2', 'stimulus 1')]
                raw_choice = d[('choice', 'choice')]

                trial = pd.DataFrame({
                    'n1': n1,
                    'n2': n2,
                    'p1': p1,
                    'p2': p2,
                    'choice': raw_choice,
                    'risky_first': p1 == 0.55,
                    'subject': subject,
                    'session': session,
                    'run': run,
                })
                rows.append(trial)

    df = pd.concat(rows)
    df = df.reset_index().set_index(['subject', 'session', 'run', 'trial_nr'])

    if remove_non_responses:
        df = df[~df['choice'].isnull()]
        df['choice'] = (df['choice'] == 2.0)
    return df


def load_abstract_values_pilot(bids_folder='/data/ds-abstract_values_pilot',
                               subjects=None, phase='test',
                               remove_non_responses=True):
    """Load pilot orientation-to-value estimation data.

    Parameters
    ----------
    bids_folder : str
        Root of the BIDS dataset.
    subjects : list of int or None
        Subject IDs to load. Default: all usable subjects (2-7, 9-16).
    phase : {'test', 'training', 'both'}
        Which task phase to load. 'test' = estimation runs only.
    remove_non_responses : bool
        Drop trials with NaN responses.

    Returns
    -------
    pd.DataFrame
        Multi-indexed (subject, session, mapping, run, trial_nr) with columns:
        orientation, value, response, mapping, response_time, etc.
    """
    if subjects is None:
        subjects = [2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16]

    all_dfs = []

    for subject in subjects:
        for session in [1, 2]:
            behavior_dir = op.join(bids_folder, 'sourcedata', 'behavior',
                                   f'sub-{subject}', f'session-{session:02d}')
            if not op.exists(behavior_dir):
                continue

            # Find mapping for this session
            for mapping in ['cdf', 'inverse_cdf']:
                for run in range(1, 9):
                    fn = op.join(behavior_dir,
                                f'sub-{subject}_ses-{session:02d}_run-{run:02d}'
                                f'_task-estimate.{mapping}_events.tsv')
                    if not op.exists(fn):
                        continue

                    d = pd.read_csv(fn, sep='\t')
                    # Filter to feedback events (contain response)
                    d = d[d['event_type'] == 'feedback'].copy()
                    d['subject'] = subject
                    d['session'] = session
                    d['mapping'] = mapping
                    d['run'] = run
                    all_dfs.append(d)

    if not all_dfs:
        raise FileNotFoundError(f'No data found in {bids_folder}')

    df = pd.concat(all_dfs, ignore_index=True)
    df['response'] = pd.to_numeric(df['response'], errors='coerce')

    if remove_non_responses:
        df = df.dropna(subset=['response'])

    df = df.set_index(['subject', 'session', 'mapping', 'run', 'trial_nr'])
    df = df.sort_index()

    return df


def load_neuralpriors(bids_folder=None, subjects=None, remove_non_responses=True):
    """Load numerosity estimation data from the neural_priors experiment.

    If bids_folder is None, loads the bundled CSV from the bauer package.
    Otherwise reads directly from TSV files (no nilearn dependency required).

    Parameters
    ----------
    bids_folder : str
        Root of the BIDS dataset.
    subjects : list of str or None
        Subject IDs (e.g., ['01', '02']). Default: all subjects with 2 sessions.
    remove_non_responses : bool
        Drop trials with NaN responses.

    Returns
    -------
    pd.DataFrame
        Multi-indexed (subject, session, run, trial_nr) with columns:
        n, response, range, response_time, error, abs_error, etc.
    """
    # If no bids_folder, load the bundled CSV
    if bids_folder is None:
        with (files(__package__) / '../data/neuralpriors.csv').open('rb') as f:
            df = pd.read_csv(f, index_col=[0, 1, 2, 3])
        if subjects is not None:
            df = df[df.index.get_level_values('subject').isin(subjects)]
        if remove_non_responses:
            df = df.dropna(subset=['response'])
        df['error'] = df['response'] - df['n']
        df['abs_error'] = np.abs(df['error'])
        return df

    if subjects is None:
        # Default: all subjects with 2 sessions (same as neural_priors package)
        import yaml
        subjects_yml = op.join(bids_folder, 'sourcedata', 'behavior')
        # Discover subjects by listing sub-XX directories
        all_subs = sorted([
            d.replace('sub-', '') for d in os.listdir(subjects_yml)
            if d.startswith('sub-') and d[4:].isdigit()
            and op.exists(op.join(subjects_yml, d, 'ses-2'))
        ])
        subjects = all_subs

    all_dfs = []
    for subject in subjects:
        for session in [1, 2]:
            behavior_dir = op.join(bids_folder, 'sourcedata', 'behavior',
                                   f'sub-{subject}', f'ses-{session}')
            if not op.exists(behavior_dir):
                continue

            for run in range(1, 9):
                fn = op.join(behavior_dir,
                             f'sub-{subject}_ses-{session}_task-estimation_task'
                             f'_run-{run}_events.tsv')
                if not op.exists(fn):
                    continue

                d = pd.read_csv(fn, sep='\t')
                d = d[d['event_type'] == 'feedback'].copy()

                d['n'] = d['n'].astype(float)
                d['response'] = pd.to_numeric(d['response'], errors='coerce')

                # Determine range condition
                d['range'] = 'wide' if (d['n'] > 25).any() else 'narrow'

                d['subject'] = subject
                d['session'] = session
                d['run'] = run

                all_dfs.append(d)

    if not all_dfs:
        raise FileNotFoundError(f'No data found in {bids_folder}')

    df = pd.concat(all_dfs, ignore_index=True)

    if remove_non_responses:
        df = df.dropna(subset=['response'])

    df['response'] = df['response'].astype(float)
    df['error'] = df['response'] - df['n']
    df['abs_error'] = np.abs(df['error'])

    df = df.set_index(['subject', 'session', 'run', 'trial_nr'])
    df = df.sort_index()

    return df


def _load_dehollander2024_symbolic(symbolic_folder='/data/ds-symbolicrisk',
                                    remove_non_responses=True):
    logs_dir = op.join(symbolic_folder, 'sourcedata', 'logs')
    subject_dirs = sorted(
        d for d in os.listdir(logs_dir)
        if d.startswith('sub-') and op.isdir(op.join(logs_dir, d))
    )

    rows = []
    for sub_dir in subject_dirs:
        subject = sub_dir.replace('sub-', '')
        fn = op.join(logs_dir, sub_dir, f'{sub_dir}_task-numeral_gambles_events.tsv')
        if not op.exists(fn):
            continue
        d = pd.read_csv(fn, sep='\t')
        d = d[d['event_type'] == 'choice'].copy()
        d['subject'] = subject
        rows.append(d[['subject', 'run', 'trial_nr', 'n1', 'n2', 'prob1', 'prob2', 'choice']])

    df = pd.concat(rows, ignore_index=True)
    df = df.rename(columns={'prob1': 'p1', 'prob2': 'p2'})
    df['risky_first'] = df['p1'] == 0.55
    df = df.set_index(['subject', 'run', 'trial_nr'])

    if remove_non_responses:
        df = df[~df['choice'].isnull()]
        df['choice'] = (df['choice'] == 2.0)
    return df
