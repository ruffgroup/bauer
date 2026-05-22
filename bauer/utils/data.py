"""Data loaders for bauer's bundled experiment CSVs.

All loaders share the same RT/choice cleaning convention: drop trials with
missing rt/choice and trials with rt outside ``[min_rt, max_rt]``, then cast
``choice`` to bool. Pass ``remove_non_responses=False`` to skip cleaning.
"""
from importlib.resources import files
import pandas as pd


def _read(fn, **read_kwargs):
    """Open a bundled CSV under ``bauer/data/``."""
    with (files(__package__) / f'../data/{fn}').open('rb') as f:
        return pd.read_csv(f, **read_kwargs)


def _clean_rt_choice(df, remove_non_responses=True, min_rt=0.15, max_rt=None):
    """Apply the canonical RT/choice cleaning to ``df`` in place.

    - Drop rows with NaN rt or choice
    - Apply ``[min_rt, max_rt]`` window
    - Cast ``choice`` to bool
    """
    if not remove_non_responses:
        return df
    cols_present = [c for c in ('rt', 'choice') if c in df.columns]
    df = df.dropna(subset=cols_present)
    if 'rt' in df.columns and min_rt is not None and min_rt > 0:
        df = df[df['rt'] >= min_rt]
    if 'rt' in df.columns and max_rt is not None:
        df = df[df['rt'] <= max_rt]
    if 'choice' in df.columns and df['choice'].dtype != bool:
        # 1.0/2.0 numeric coding → True if option 2 chosen
        if df['choice'].dtype.kind in 'fi':
            df = df.assign(choice=(df['choice'] == 2.0))
        else:
            df = df.assign(choice=df['choice'].astype(bool))
    return df


# ---------------------------------------------------------------------------
# Garcia 2022
# ---------------------------------------------------------------------------

def load_garcia2022(task='magnitude', remove_non_responses=True, min_rt=0.15,
                    max_rt=None):
    """Behavioural data from Barreto-Garcia et al. (2022).

    For the magnitude task, the raw CSV stores rt in milliseconds; this loader
    converts to seconds. Implausibly fast trials (rt < 150 ms) are dropped by
    default — typical motor anticipations distort DDM non-decision times.

    The magnitude-task dataframe carries an **``isi``** column (seconds)
    extracted from the original BIDS events.tsv files — the inter-stimulus
    interval between offset of n1 and onset of n2. The design jitters ISI
    over seven half-second levels {6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0}; useful
    for testing whether memory-load duration changes encoding noise or
    response caution (see docs/tutorial/lesson8.ipynb).

    Parameters
    ----------
    task : {'magnitude', 'risk'}
    remove_non_responses : bool
    min_rt, max_rt : float
        RT cutoffs in seconds (post-conversion). Set ``min_rt=0`` to disable.
    """
    fn = f'garcia2022_{task}.csv'
    df = _read(fn, index_col=[0, 1, 2, 3])
    if 'rt' in df.columns:
        if remove_non_responses:
            df = df[df['rt'] > 0]   # raw -1 sentinel
        df = df.assign(rt=df['rt'] / 1000.0)  # ms → s
    return _clean_rt_choice(df, remove_non_responses, min_rt, max_rt)


# ---------------------------------------------------------------------------
# de Hollander 2024 — dotcloud (risky)
# ---------------------------------------------------------------------------

def load_dehollander2024_risk(sessions=None, remove_non_responses=True,
                              min_rt=0.15, max_rt=None):
    """De Hollander et al. (2024 Nature Communications) dotcloud risky-choice
    task — N=30 subjects across 3T and 7T sessions, ~256 trials/subject.

    ``choice = True`` means option 2 chosen (bauer's risk convention).
    The risky lottery is at p=0.55, the safe at p=1. Derived columns
    (``risky_first``, ``chose_risky``, etc.) are not bundled — compute on
    the fly from ``p1, p2``.
    """
    df = _read('dehollander2024_risk.csv', dtype={'subject': str})
    if sessions is not None:
        df = df[df['session'].isin(sessions)]
    df = _clean_rt_choice(df, remove_non_responses, min_rt, max_rt)
    return df.set_index(['subject', 'session', 'run', 'trial_nr']).sort_index()


# ---------------------------------------------------------------------------
# de Hollander 2024 — symbolic (Arabic-numeral risky)
# ---------------------------------------------------------------------------

def load_dehollander2024_symbolic(remove_non_responses=True, min_rt=0.15,
                                  max_rt=None):
    """De Hollander 2024 *symbolic* (Arabic-numeral) risky-choice task —
    N=58 subjects, ~256 trials each. Unlike the dotcloud task, n1/n2 are
    continuous (range ~5–100), making this a stronger test of stimulus-
    dependent encoding-noise (flex) models.
    """
    df = _read('dehollander2024_symbolic.csv', dtype={'subject': str})
    df = _clean_rt_choice(df, remove_non_responses, min_rt, max_rt)
    return df.set_index(['subject', 'run', 'trial_nr']).sort_index()


# ---------------------------------------------------------------------------
# de Hollander TMS-risk
# ---------------------------------------------------------------------------

def load_bedi2026(remove_non_responses=True):
    """Bedi 2026 abstract-value estimation pilot — orientation→value mapping.

    13 subjects across 2 sessions × 2 mapping conditions ('cdf' /
    'inverse_cdf') × 8 runs. On each trial the participant sees an oriented
    Gabor and estimates its associated value (CHF) on a continuous scale; a
    BDM-auction-derived ``value`` is the ground truth and ``reward`` is what
    they actually earn.

    This is a continuous-response task — use the continuous-response models
    (``EstimationBaseModel`` family) rather than the discrete-choice family.

    Bundled CSV: ``bauer/data/bedi2026.csv`` with columns subject, session,
    mapping, run, trial_nr, orientation, value, reward, response,
    response_time.
    """
    df = _read('bedi2026.csv', dtype={'subject': int})
    if remove_non_responses:
        df = df.dropna(subset=['response'])
    df = df.set_index(['subject', 'session', 'mapping', 'run', 'trial_nr']).sort_index()
    return df


def load_dehollander_tms_risk(stimulation_conditions=None, sessions=None,
                              tms_only=True, remove_non_responses=True,
                              min_rt=0.15, max_rt=None):
    """De Hollander TMS-risk experiment — 73 subjects total but only 35 of
    them completed the TMS sessions (sessions 2 and 3); the remaining 38
    only did the baseline session.

    For TMS analyses you usually want only the 35 TMS subjects (sessions 2/3)
    — that's the ``tms_only=True`` default. Set ``tms_only=False`` to get all
    73 subjects across all sessions.

    Parameters
    ----------
    stimulation_conditions : list of str or None
        Subset of {'baseline', 'vertex', 'ips'} to keep. Default: all.
    sessions : list of int or None
        Subset of {1, 2, 3}. Default: ``[2, 3]`` if ``tms_only=True``,
        else all.
    tms_only : bool
        If True, keep only the 35 TMS-completing subjects in sessions 2/3.
    """
    df = _read('dehollander_tms_risk.csv')
    if tms_only:
        if sessions is None:
            sessions = [2, 3]
        tms_subjects = df[df['session'].isin([2, 3])]['subject'].unique()
        df = df[df['subject'].isin(tms_subjects)]
    if stimulation_conditions is not None:
        df = df[df['stimulation_condition'].isin(stimulation_conditions)]
    if sessions is not None:
        df = df[df['session'].isin(sessions)]
    df = _clean_rt_choice(df, remove_non_responses, min_rt, max_rt)
    return df.set_index(['subject', 'session', 'stimulation_condition',
                         'run', 'trial_nr']).sort_index()
