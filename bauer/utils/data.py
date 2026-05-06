from importlib.resources import files
import os
import os.path as op
import pandas as pd
import numpy as np

def load_garcia2022(task='magnitude', remove_non_responses=True, min_rt=0.15,
                    max_rt=None):
    """Return behavioral data from Barreto Garcia et al. (2022) as a multi-indexed DataFrame.

    For the magnitude task, the raw CSV stores ``rt`` in milliseconds with
    ``-1`` as a non-response sentinel. This loader converts ``rt`` to seconds
    and (when ``remove_non_responses=True``) drops trials with ``rt <= 0`` so
    the column is directly consumable by DDM models. Implausibly fast trials
    (RT < 150 ms by default) are also dropped — these are typically motor
    anticipations / fast guesses that distort DDM non-decision-time estimates.

    Parameters
    ----------
    task : {'magnitude', 'risk'}
        Which task dataset to load.
    remove_non_responses : bool
        If True, drop trials with missing choices (and, for magnitude, with
        ``rt <= 0`` or RTs outside ``[min_rt, max_rt]``) and cast ``choice``
        to bool.
    min_rt : float
        Minimum RT in seconds (default 0.15). Trials faster than this are
        dropped. Set to ``0`` to disable.
    max_rt : float or None
        Maximum RT in seconds. If None (default), no upper cut.
    """

    if task == 'magnitude':
        fn = 'garcia2022_magnitude.csv'
    elif task == 'risk':
        fn = 'garcia2022_risk.csv'

    with (files(__package__) / f'../data/{fn}').open('rb') as f:
        df = pd.read_csv(f, index_col=[0, 1, 2, 3])

    if 'rt' in df.columns:
        if remove_non_responses:
            df = df[df['rt'] > 0]
        df['rt'] = df['rt'] / 1000.0  # ms -> s
        if remove_non_responses:
            if min_rt is not None and min_rt > 0:
                df = df[df['rt'] >= min_rt]
            if max_rt is not None:
                df = df[df['rt'] <= max_rt]

    if remove_non_responses:
        df = df[~df['choice'].isnull()]
        df['choice'] = df['choice'].astype(bool)
    return df


def load_dehollander_tms_risk(stimulation_conditions=None, sessions=None,
                                tms_only=True, remove_non_responses=True,
                                min_rt=0.15, max_rt=None):
    """Return behavioral data from the de Hollander TMS-risk experiment.

    73 subjects total, but **only 35 of them completed the TMS sessions
    (sessions 2 and 3)**; the remaining 38 only did the baseline session
    (session 1). For TMS analyses you typically want only the 35 TMS
    subjects and only sessions 2/3 — the ``tms_only=True`` default does
    exactly that. Set ``tms_only=False`` to get all 73 subjects across
    all sessions (useful for behavioral baselines or pilot analyses).

    Each trial has two lotteries (one risky at ``p=0.55``, one safe at
    ``p=1.0``) and a binary choice. RT is in seconds. ``choice = True``
    means option 2 was chosen (bauer's risk convention). The TMS condition
    is in the ``stimulation_condition`` column ∈ {baseline, vertex, ips};
    pass it as a regression covariate to fit how stimulation modulates
    noise/aversion.

    Loads from bundled CSV (``bauer/data/dehollander_tms_risk.csv``).

    Parameters
    ----------
    stimulation_conditions : list of str or None
        Subset of ``['baseline', 'vertex', 'ips']`` to keep. Default: all.
    sessions : list of int or None
        Subset of ``[1, 2, 3]`` to keep. Default: ``[2, 3]`` if
        ``tms_only=True``, otherwise all.
    tms_only : bool
        If True (default), keep only the 35 subjects who completed TMS
        sessions, and only sessions 2 and 3. If False, return all 73
        subjects across all sessions.
    remove_non_responses : bool
        Drop trials with missing RT or choice; apply RT cutoffs.
    min_rt : float
        Lower RT cutoff in seconds (default 0.15). Set to 0 to disable.
    max_rt : float or None
        Upper RT cutoff in seconds (default None).
    """
    fn = 'dehollander_tms_risk.csv'
    with (files(__package__) / f'../data/{fn}').open('rb') as f:
        df = pd.read_csv(f)

    if tms_only:
        if sessions is None:
            sessions = [2, 3]
        # Keep only subjects who appear in the TMS sessions.
        tms_subjects = df[df['session'].isin([2, 3])]['subject'].unique()
        df = df[df['subject'].isin(tms_subjects)]

    if stimulation_conditions is not None:
        df = df[df['stimulation_condition'].isin(stimulation_conditions)]
    if sessions is not None:
        df = df[df['session'].isin(sessions)]

    if remove_non_responses:
        df = df.dropna(subset=['rt', 'choice'])
        if min_rt is not None and min_rt > 0:
            df = df[df['rt'] >= min_rt]
        if max_rt is not None:
            df = df[df['rt'] <= max_rt]

    df['choice'] = (df['choice'] == 2.0)
    df = df.set_index(['subject', 'session', 'stimulation_condition',
                        'run', 'trial_nr']).sort_index()
    return df


def load_dehollander2024_symbolic(remove_non_responses=True,
                                   min_rt=0.15, max_rt=None):
    """Return behavioral data from de Hollander et al. (2024 Nat Comms)
    *symbolic* (Arabic-numeral) risky-choice task as a multi-indexed DataFrame.

    Loads from the bundled CSV (``bauer/data/dehollander2024_symbolic.csv``).
    58 subjects, ~256 trials each. RT is in seconds. Unlike the dotcloud task,
    n1/n2 are continuous (each trial samples a different number on a fine
    grid, range ~5-100), making this a stronger test of stimulus-dependent
    encoding-noise models. ``choice = True`` means option 2 was chosen.

    Parameters
    ----------
    remove_non_responses : bool
        Drop rows with missing choices and trials with RT outside
        ``[min_rt, max_rt]``.
    min_rt : float
        Lower RT cutoff in seconds (default 0.15). Set to 0 to disable.
    max_rt : float or None
        Upper RT cutoff in seconds (default None).
    """
    fn = 'dehollander2024_symbolic.csv'
    with (files(__package__) / f'../data/{fn}').open('rb') as f:
        df = pd.read_csv(f, dtype={'subject': str})

    if remove_non_responses:
        df = df.dropna(subset=['rt', 'choice'])
        if min_rt is not None and min_rt > 0:
            df = df[df['rt'] >= min_rt]
        if max_rt is not None:
            df = df[df['rt'] <= max_rt]

    df['choice'] = df['choice'].astype(bool)
    df = df.set_index(['subject', 'run', 'trial_nr']).sort_index()
    return df


def load_dehollander2024_risk(sessions=None, remove_non_responses=True,
                               min_rt=0.15, max_rt=None):
    """Return behavioral data from de Hollander et al. (2024 Nat Comms) risky-choice
    task as a multi-indexed DataFrame.

    Loads from the bundled CSV (``bauer/data/dehollander2024_risk.csv``), which
    contains one row per trial across 30 subjects (subject IDs as zero-padded
    strings ``'02'`` … ``'32'``, sub-24 excluded) and two sessions (3T MRI,
    7T MRI). RT is in seconds. ``choice = True`` means the participant chose
    option 2 (the second-presented option), matching bauer's risk-model
    convention. Trials with no response (RT or choice missing) are dropped by
    default.

    Parameters
    ----------
    sessions : list of str or None
        Sessions to include (default: all). Valid values: ``['3t2', '7t2']``.
    remove_non_responses : bool
        Drop rows with missing RT/choice and trials with RT outside
        ``[min_rt, max_rt]``.
    min_rt : float
        Lower RT cutoff in seconds (default 0.15). Set to 0 to disable.
    max_rt : float or None
        Upper RT cutoff in seconds (default None).

    Returns
    -------
    pd.DataFrame
        Multi-indexed by (subject, session, run, trial_nr). Columns include
        ``n1``, ``n2``, ``p1``, ``p2``, ``risky_first``, ``choice`` (bool),
        ``rt`` (s), ``log_risky_safe``, ``chose_risky``, ``certainty``.
    """
    fn = 'dehollander2024_risk.csv'
    with (files(__package__) / f'../data/{fn}').open('rb') as f:
        df = pd.read_csv(f, dtype={'subject': str})

    if sessions is not None:
        df = df[df['session'].isin(sessions)]

    if remove_non_responses:
        df = df.dropna(subset=['rt', 'choice'])
        if min_rt is not None and min_rt > 0:
            df = df[df['rt'] >= min_rt]
        if max_rt is not None:
            df = df[df['rt'] <= max_rt]

    df['choice'] = df['choice'].astype(bool)
    df = df.set_index(['subject', 'session', 'run', 'trial_nr']).sort_index()
    return df


def load_dehollander2024(task='dotcloud', sessions=None,
                         bids_folder=None,
                         symbolic_folder='/data/ds-symbolicrisk',
                         remove_non_responses=True):
    """Return behavioral data from de Hollander et al. (2024) as a multi-indexed DataFrame.

    Parameters
    ----------
    task : {'dotcloud', 'symbolic'}
        Which task dataset to load. ``'dotcloud'`` defaults to the **bundled CSV**
        (no BIDS folder needed); pass ``bids_folder`` to read raw events.tsvs
        from the ds-risk dataset instead. ``'symbolic'`` uses the Arabic-numeral
        behavioural task at ``symbolic_folder``.
    sessions : list of str or None
        Sessions to include for the dotcloud task (default: all sessions in the
        bundled CSV, i.e. ``['3t2', '7t2']``). Ignored for the symbolic task.
    bids_folder : str or None
        If given (and ``task='dotcloud'``), reads from raw BIDS events.tsvs
        instead of the bundled CSV. Default ``None`` → use the bundled CSV.
    symbolic_folder : str
        Root of the ds-symbolicrisk dataset.
    remove_non_responses : bool
        If True, drop trials with missing choices and cast the ``choice`` column
        to bool.

    Notes
    -----
    Prefer :func:`load_dehollander2024_risk` for new code — it has cleaner
    defaults and an RT cutoff. This function is kept for back-compat with
    notebooks and tutorials that import it under the old name.
    """
    if task == 'dotcloud':
        if bids_folder is None:
            df = load_dehollander2024_risk(
                sessions=sessions, remove_non_responses=remove_non_responses,
                min_rt=0.0,  # back-compat: no RT cutoff in old behaviour
            )
            # Match the column subset the legacy loader returned.
            keep = [c for c in ['n1', 'n2', 'p1', 'p2', 'choice', 'risky_first']
                    if c in df.columns]
            return df[keep]
        return _load_dehollander2024_dotcloud(
            sessions=sessions,
            bids_folder=bids_folder,
            remove_non_responses=remove_non_responses,
        )
    elif task == 'symbolic':
        if symbolic_folder == '/data/ds-symbolicrisk' and not op.isdir(symbolic_folder):
            # Default BIDS path doesn't exist → use bundled CSV.
            df = load_dehollander2024_symbolic(
                remove_non_responses=remove_non_responses, min_rt=0.0,
            )
            keep = [c for c in ['n1', 'n2', 'p1', 'p2', 'choice', 'risky_first']
                    if c in df.columns]
            return df[keep]
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
