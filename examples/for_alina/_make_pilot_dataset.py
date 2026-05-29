"""Load Alina's BIDS-style risky-choice pilot data into a clean trial DataFrame.

Source format: BIDS-events TSVs at
    data/pilot/sub-pilXX/ses-N/sub-pilXX_ses-N_task-numloss_run-K_events.tsv

Each TSV has one row per event (fixation, cue, piechart, array, response,
choice, feedback, iti, pulses, ...). We need one row per *trial*, with:

    subject, session, run, trial, n1, n2, p1, p2, choice, rt, domain

Conventions used here:

- ``n1, n2`` in Alina's events files are **signed** — positive for gains,
  negative for losses. ``domain = 'gain' if n1 > 0 else 'loss'`` (both
  options in a trial always have the same sign).
- ``n1, n2`` returned by this loader are **absolute values**, matching
  bauer's convention (positive magnitudes; domain in a separate column).
- ``rt`` in the events ``feedback`` row is the absolute experiment time
  when the response key was pressed. We compute **decision time** as
  ``feedback.rt − (response-phase-start onset)`` so the loaded ``rt``
  column is what bauer's DDM expects (time from response cue to keypress,
  in seconds).
- ``choice`` is the value from the ``choice`` event (1.0 → ``True``).
- Trials with no response (``choice`` is NaN) are dropped — bauer cannot
  fit non-responses in this setup.

Usage::

    python load_pilot.py                              # writes data/pilot_data.csv
    python load_pilot.py --pilot-dir /path/to/data    # custom source
"""
import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).parent
DEFAULT_PILOT_DIR = HERE / 'data' / 'pilot'
DEFAULT_OUT       = HERE / 'data' / 'pilot_data.csv'

# Regex on filename:  sub-pilXX_ses-N_task-numloss_run-K_events.tsv
FILENAME_RE = re.compile(
    r'sub-(?P<subject>[\w-]+)_'
    r'ses-(?P<session>\d+)_'
    r'task-numloss_'
    r'run-(?P<run>\d+)_events\.tsv$'
)


def _load_one(path):
    """Convert a single events.tsv into one row per trial."""
    df = pd.read_csv(path, sep='\t')

    # Response-phase start: event_type='response' AND response is NaN (the cue
    # that the response window opened — no key has been pressed yet).
    resp_start = (df[df['event_type'] == 'response']
                  .groupby('trial_nr')['onset'].first())

    # Feedback row: one per trial, holds final n1, n2, p1, p2, choice, and
    # the absolute time of the keypress (in the ``rt`` column).
    fb = df[df['event_type'] == 'feedback'].copy()
    fb = fb.set_index('trial_nr')

    # Decision time = absolute response press − response-phase-start onset.
    # NaN-safe: trials without a response have NaN rt anyway.
    fb['decision_time'] = fb['rt'] - resp_start.reindex(fb.index)

    # Build the trial-level frame
    out = pd.DataFrame({
        'n1':     fb['n1'].abs(),     # |magnitude|; domain stored separately
        'n2':     fb['n2'].abs(),
        'p1':     fb['p1'],
        'p2':     fb['p2'],
        'choice': fb['choice'].astype('boolean'),    # nullable bool
        'rt':     fb['decision_time'],
        'domain': np.where(fb['n1'] >= 0, 'gain', 'loss'),
    }).reset_index().rename(columns={'trial_nr': 'trial_in_run'})

    return out


def load_pilot(pilot_dir=DEFAULT_PILOT_DIR, glob_pattern='sub-*/ses-*/*events*.tsv'):
    """Walk ``pilot_dir`` for events TSVs and concatenate into one DataFrame."""
    pilot_dir = Path(pilot_dir)
    files = sorted(glob.glob(str(pilot_dir / glob_pattern)))
    if not files:
        raise FileNotFoundError(
            f"No events files matched {pilot_dir}/{glob_pattern}. "
            f"rsync the pilot data into examples/for_alina/data/pilot first."
        )

    rows = []
    for f in files:
        m = FILENAME_RE.search(f)
        if m is None:
            print(f'  skipped (filename does not match): {f}', file=sys.stderr)
            continue
        info = m.groupdict()
        trials = _load_one(f)
        trials['subject'] = info['subject']
        trials['session'] = int(info['session'])
        trials['run']     = int(info['run'])
        rows.append(trials)

    full = pd.concat(rows, ignore_index=True)

    # Drop trials with no response
    n_pre = len(full)
    full = full.dropna(subset=['choice', 'rt']).copy()
    full['choice'] = full['choice'].astype(bool)
    full = full[full['rt'] > 0]
    n_post = len(full)
    if n_post < n_pre:
        print(f'Dropped {n_pre - n_post} no-response/invalid-RT trials '
              f'({n_post} remaining).')

    # Build a single monotonically-increasing trial index per subject
    full = full.sort_values(['subject', 'session', 'run', 'trial_in_run']).reset_index(drop=True)
    full['trial'] = full.groupby('subject').cumcount() + 1
    full = full.set_index(['subject', 'trial'])
    full = full[['session', 'run', 'trial_in_run',
                 'n1', 'n2', 'p1', 'p2', 'domain', 'choice', 'rt']]
    return full


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--pilot-dir', type=Path, default=DEFAULT_PILOT_DIR)
    parser.add_argument('--out',       type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = load_pilot(args.pilot_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)

    print(f'\nWrote {len(df)} trials × '
          f'{df.index.get_level_values("subject").nunique()} subjects → {args.out}')
    print('\nTrials per (subject, domain):')
    print(df.groupby(['subject', 'domain']).size().unstack(fill_value=0))
    print('\nP(choice=True) by (subject, domain):')
    print(df.groupby(['subject', 'domain'])['choice'].mean().round(3).unstack())
    print('\nMean RT by (subject, domain):')
    print(df.groupby(['subject', 'domain'])['rt'].mean().round(3).unstack())


if __name__ == '__main__':
    sys.exit(main())
