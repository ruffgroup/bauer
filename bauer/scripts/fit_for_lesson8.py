"""Fit the probit + DDM caches used by docs/tutorial/lesson8.ipynb.

Standalone (not nbconvert), so the notebook runs locally in seconds once
the .nc files exist in ~/.bauer_tutorial_cache/. Matches lesson 8's
fit_or_load exactly: same model kwargs, same sample kwargs, same cache
key (`garcia_n{n_subj}_rtmin{ms}_{name}.nc`).

Usage (local):
    python -m bauer.scripts.fit_for_lesson8

Usage (cluster sbatch wrapper):
    sbatch bauer/scripts/slurm_jobs/lesson8_cache.sh
"""
import os
import time
import threading
import warnings
import argparse
import pandas as pd

warnings.filterwarnings('ignore')


def heartbeat(stop_event, label, interval=60):
    """Print '[heartbeat] <label>: <min> min elapsed' every <interval> s.
    Needed because pmpc/numpyro progressbars are silent in non-TTY logs."""
    t0 = time.time()
    while not stop_event.wait(interval):
        print(f'[heartbeat] {label}: {(time.time()-t0)/60:.1f} min elapsed',
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rt-min', type=float, default=0.20,
                    help='Drop trials below this RT in seconds (default 0.20).')
    ap.add_argument('--backend', default='numpyro',
                    choices=['pymc', 'numpyro', 'blackjax'])
    ap.add_argument('--chain-method', default=None,
                    choices=['vectorized', 'parallel', 'sequential'],
                    help='JAX-backend chain method (numpyro/blackjax only). '
                         'Default = vectorized (bauer default).')
    ap.add_argument('--cache-dir', default='~/.bauer_tutorial_cache')
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--tune', type=int, default=1000)
    ap.add_argument('--chains', type=int, default=4)
    ap.add_argument('--target-accept', type=float, default=0.95)
    args = ap.parse_args()

    import numpy as np
    from bauer.utils.data import load_garcia2022
    from bauer.models import (
        MagnitudeComparisonModel, DDMMagnitudeComparisonModel,
        DDMMagnitudeComparisonRegressionModel,
    )

    cache_dir = os.path.expanduser(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    df = load_garcia2022(task='magnitude')
    n_subj = df.index.get_level_values('subject').nunique()
    n_before = len(df)
    df = df[df['rt'] >= args.rt_min].copy()
    print(f"Garcia magnitude: {n_subj} subjects.  Dropped "
          f"{n_before - len(df)}/{n_before} trials with rt < {args.rt_min:.2f}s "
          f"({100*(n_before - len(df))/n_before:.1f}%).  min(rt) = "
          f"{df['rt'].min():.3f}s.",
          flush=True)

    # Categorical ISI for the regression-DDM demo (median split).
    df['isi_cat'] = pd.Categorical(
        np.where(df['isi'] >= df['isi'].median(), 'long', 'short'),
        categories=['short', 'long'],   # 'short' is reference level
    )

    tag = f'garcia_n{n_subj}_rtmin{int(args.rt_min*1000)}'

    def fit_one(name, build):
        path = os.path.join(cache_dir, f'{tag}_{name}.nc')
        if os.path.exists(path):
            print(f"{name}: cached, skipping ({path})", flush=True)
            return
        m = build()
        m.build_estimation_model(data=df, hierarchical=True)
        stop = threading.Event()
        hb = threading.Thread(target=heartbeat, args=(stop, name), daemon=True)
        hb.start()
        t0 = time.time()
        sample_kw = dict(draws=args.draws, tune=args.tune,
                         chains=args.chains,
                         target_accept=args.target_accept,
                         backend=args.backend)
        if args.chain_method is not None and args.backend != 'pymc':
            sample_kw['chain_method'] = args.chain_method
        if args.backend == 'blackjax':
            # blackjax 1.x + HSSM progress bar = JAX vmap/cond crash
            # ("IO effect not supported in vmap-of-cond"). The heartbeat
            # thread above gives us tail-able progress regardless.
            sample_kw['progressbar'] = False
        try:
            idata = m.sample(**sample_kw)
        finally:
            stop.set(); hb.join(timeout=2)
        dt = time.time() - t0
        idata.to_netcdf(path)
        print(f"{name}: saved in {dt/60:.1f} min -> {path}", flush=True)

    fit_one('probit', lambda: MagnitudeComparisonModel(
        paradigm=df, fit_separate_evidence_sd=True, fit_prior=True))
    fit_one('ddm', lambda: DDMMagnitudeComparisonModel(
        paradigm=df, fit_separate_evidence_sd=True, fit_prior=True))
    fit_one('ddm_isi', lambda: DDMMagnitudeComparisonRegressionModel(
        paradigm=df, fit_separate_evidence_sd=True, fit_prior=True,
        regressors={'n1_evidence_sd': 'isi_cat'}))


if __name__ == '__main__':
    main()
