"""Fit MagnitudeComparisonModel and DDMMagnitudeComparisonModel hierarchically
on Barreto-Garcia et al. (2022). Saves InferenceData to netcdf so the companion
notebook can load cached results.

Defaults (small example fit, ~10 min on a laptop):
    python notebooks/fit_ddm_garcia.py

Full fit on all subjects and trials (slow — hours on a laptop):
    python notebooks/fit_ddm_garcia.py --full

Custom slice:
    python notebooks/fit_ddm_garcia.py --n-subjects 12 --n-trials 300
"""

import argparse
import os
import os.path as op
import warnings

warnings.filterwarnings('ignore')

from bauer.utils.data import load_garcia2022
from bauer.models import (
    MagnitudeComparisonModel,
    DDMMagnitudeComparisonModel,
    RaceDiffusionMagnitudeComparisonModel,
)


HERE = op.dirname(op.abspath(__file__))


# PyMC's rich progress bar is interactive-only — it emits ~2 frames total when
# stdout is piped to a log. Tiny callback gives us visible progress in the log.
# Module-level so it pickles for multiprocess sampling.
def _progress(trace, draw):  # noqa: ARG001 — pymc passes trace= as kwarg
    if draw.draw_idx % 200 == 0:
        phase = 'tune' if draw.tuning else 'draw'
        print(f'  chain {draw.chain} {phase} {draw.draw_idx}', flush=True)


def _safe_to_netcdf(idata, path):
    """Atomic netcdf write — survives the target file being open elsewhere
    (e.g. by a running notebook). Writes to ``path.tmp`` then renames; readers
    keep their old inode until they re-open."""
    tmp = path + '.tmp'
    idata.to_netcdf(tmp)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-subjects', type=int, default=6,
                        help='Number of subjects to include (default 6).')
    parser.add_argument('--n-trials', type=int, default=200,
                        help='Trials per subject; None or 0 for all (default 200).')
    parser.add_argument('--full', action='store_true',
                        help='Shorthand: all subjects, all trials. Output to results_full/.')
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=2)
    parser.add_argument('--cores', type=int, default=2)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip fits whose .nc file already exists in output dir.')
    args = parser.parse_args()

    if args.full:
        n_subjects = None
        n_trials = None
        out_dir = args.output_dir or op.join(HERE, 'results_full')
    else:
        n_subjects = args.n_subjects
        n_trials = args.n_trials if args.n_trials and args.n_trials > 0 else None
        out_dir = args.output_dir or op.join(HERE, 'results')
    os.makedirs(out_dir, exist_ok=True)

    df = load_garcia2022(task='magnitude')
    if n_subjects is not None:
        subs = df.index.get_level_values('subject').unique()[:n_subjects]
        df = df.loc[df.index.get_level_values('subject').isin(subs)]
    if n_trials is not None:
        df = df.groupby('subject', group_keys=False).head(n_trials)
    df = df.copy()
    print(f'Fitting {len(df)} trials, '
          f'{df.index.get_level_values("subject").nunique()} subjects')

    choice_path = op.join(out_dir, 'garcia_choice_idata.nc')
    ddm_path = op.join(out_dir, 'garcia_ddm_idata.nc')
    race_path = op.join(out_dir, 'garcia_race_idata.nc')

    sample_kwargs = dict(draws=args.draws, tune=args.tune,
                         chains=args.chains, cores=args.cores,
                         progressbar=False, callback=_progress)

    if args.skip_existing and op.exists(choice_path):
        print(f'\n=== choice-only: SKIPPED (exists at {choice_path}) ===', flush=True)
    else:
        print('\n=== choice-only model ===', flush=True)
        m_choice = MagnitudeComparisonModel(paradigm=df, fit_seperate_evidence_sd=True)
        m_choice.build_estimation_model(data=df, hierarchical=True)
        idata_choice = m_choice.sample(target_accept=0.9, random_seed=0, **sample_kwargs)
        _safe_to_netcdf(idata_choice, choice_path)
        print(f'saved -> {choice_path}', flush=True)

    if args.skip_existing and op.exists(ddm_path):
        print(f'\n=== DDM: SKIPPED (exists at {ddm_path}) ===', flush=True)
    else:
        print('\n=== DDM model (v_scale fixed to 1) ===', flush=True)
        m_ddm = DDMMagnitudeComparisonModel(paradigm=df, fit_seperate_evidence_sd=True,
                                             fit_v_scale=False)
        m_ddm.build_estimation_model(data=df, hierarchical=True)
        idata_ddm = m_ddm.sample(target_accept=0.95, random_seed=0, **sample_kwargs)
        _safe_to_netcdf(idata_ddm, ddm_path)
        print(f'saved -> {ddm_path}', flush=True)

    if args.skip_existing and op.exists(race_path):
        print(f'\n=== race: SKIPPED (exists at {race_path}) ===', flush=True)
        print('\nDone.')
        return

    print('\n=== Race-diffusion model (fit_prior=True, v_scale fixed to 1) ===', flush=True)
    m_race = RaceDiffusionMagnitudeComparisonModel(
        paradigm=df, fit_seperate_evidence_sd=True, fit_v_scale=False,
        fit_prior=True,
    )
    m_race.build_estimation_model(data=df, hierarchical=True)
    idata_race = m_race.sample(target_accept=0.95, random_seed=0, **sample_kwargs)
    _safe_to_netcdf(idata_race, race_path)
    print(f'saved -> {race_path}', flush=True)

    print('\nDone.')


if __name__ == '__main__':
    main()
