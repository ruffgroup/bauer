"""Fit FlexibleNoiseComparisonModel and DDMFlexibleNoiseComparisonModel
hierarchically on Barreto-Garcia et al. (2022). Saves InferenceData to netcdf.

Defaults (small smoke test, ~5 min):
    python notebooks/fit_flexible_ddm_garcia.py --n-subjects 4 --n-trials 200

Full fit on all subjects, all trials (slow — hours):
    python notebooks/fit_flexible_ddm_garcia.py --full
"""

import argparse
import os
import os.path as op
import warnings

warnings.filterwarnings('ignore')

from bauer.utils.data import load_garcia2022
from bauer.models import (
    FlexibleNoiseComparisonModel,
    DDMFlexibleNoiseComparisonModel,
)


HERE = op.dirname(op.abspath(__file__))


def _progress(trace, draw):  # noqa: ARG001 — pymc passes trace= as kwarg
    if draw.draw_idx % 200 == 0:
        phase = 'tune' if draw.tuning else 'draw'
        print(f'  chain {draw.chain} {phase} {draw.draw_idx}', flush=True)


def _safe_to_netcdf(idata, path):
    tmp = path + '.tmp'
    idata.to_netcdf(tmp)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-subjects', type=int, default=4)
    parser.add_argument('--n-trials', type=int, default=200)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--polynomial-order', type=int, default=5)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=2)
    parser.add_argument('--cores', type=int, default=2)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    if args.full:
        n_subjects = None
        n_trials = None
        out_dir = args.output_dir or op.join(HERE, 'results_flexible_full')
    else:
        n_subjects = args.n_subjects
        n_trials = args.n_trials if args.n_trials and args.n_trials > 0 else None
        out_dir = args.output_dir or op.join(HERE, 'results_flexible')
    os.makedirs(out_dir, exist_ok=True)

    df = load_garcia2022(task='magnitude')
    if n_subjects is not None:
        subs = df.index.get_level_values('subject').unique()[:n_subjects]
        df = df.loc[df.index.get_level_values('subject').isin(subs)]
    if n_trials is not None:
        df = df.groupby('subject', group_keys=False).head(n_trials)
    df = df.copy()
    print(f'Fitting {len(df)} trials, '
          f'{df.index.get_level_values("subject").nunique()} subjects, '
          f'polynomial_order={args.polynomial_order}')

    choice_path = op.join(out_dir, 'garcia_flex_choice_idata.nc')
    ddm_path = op.join(out_dir, 'garcia_flex_ddm_idata.nc')

    sample_kwargs = dict(draws=args.draws, tune=args.tune,
                         chains=args.chains, cores=args.cores,
                         progressbar=False, callback=_progress)

    print('\n=== flexible-noise choice-only model ===', flush=True)
    m_choice = FlexibleNoiseComparisonModel(
        paradigm=df, fit_seperate_evidence_sd=True,
        polynomial_order=args.polynomial_order,
    )
    m_choice.build_estimation_model(paradigm=df, hierarchical=True)
    idata_choice = m_choice.sample(target_accept=0.9, random_seed=0, **sample_kwargs)
    _safe_to_netcdf(idata_choice, choice_path)
    print(f'saved -> {choice_path}', flush=True)

    print('\n=== flexible-noise DDM model (v_scale fixed to 1) ===', flush=True)
    m_ddm = DDMFlexibleNoiseComparisonModel(
        paradigm=df, fit_seperate_evidence_sd=True,
        polynomial_order=args.polynomial_order, fit_v_scale=False,
    )
    m_ddm.build_estimation_model(paradigm=df, hierarchical=True)
    idata_ddm = m_ddm.sample(target_accept=0.95, random_seed=0, **sample_kwargs)
    _safe_to_netcdf(idata_ddm, ddm_path)
    print(f'saved -> {ddm_path}', flush=True)

    print('\nDone.')


if __name__ == '__main__':
    main()
