"""n=64 (production) arm of the DDM sampler experiment.

Same questions as the pil03 arm, but on the real target: Garcia 2022
magnitude comparison, **n=64, hierarchical**, basic DDM vs the ISI-regression
DDM. One (model, backend, chain_method, tune, seed) cell per invocation, so it
can be fanned out as independent sbatch jobs (each n=64 fit is slow — hours for
the regression model — so we run the decisive cells, not a full crossing).

Each run appends one row to a per-cell TSV under --out-dir (filename encodes
the config, so parallel jobs don't collide). Aggregate the rows afterwards.

    python run_ddm_sampler_experiment_n64.py \
        --model regression --backend numpyro --chain-method parallel \
        --tune 2000 --draws 1000 --seed 0 --out-dir notes/experiments/n64_cells
"""
import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az

from bauer.utils.data import load_garcia2022
from bauer.models import (DDMMagnitudeComparisonModel,
                          DDMMagnitudeComparisonRegressionModel)

DIAG_VARS = ['n1_evidence_sd_mu', 'n2_evidence_sd_mu', 'a_mu', 't0_mu',
             'prior_mu_mu', 'prior_sd_mu']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=('basic', 'regression'), required=True)
    ap.add_argument('--backend', choices=('numpyro', 'blackjax'), default='numpyro')
    ap.add_argument('--chain-method', choices=('vectorized', 'parallel'),
                    default='vectorized')
    ap.add_argument('--tune', type=int, default=2000)
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--target-accept', type=float, default=0.99)
    ap.add_argument('--rt-min', type=float, default=0.20)
    ap.add_argument('--out-dir', type=Path,
                    default=Path(__file__).parent / 'n64_cells')
    args = ap.parse_args()

    df = load_garcia2022(task='magnitude')
    n_subj = df.index.get_level_values('subject').nunique()
    df = df[df['rt'] >= args.rt_min].copy()
    df['isi_cat'] = pd.Categorical(
        np.where(df['isi'] >= df['isi'].median(), 'long', 'short'),
        categories=['short', 'long'])
    print(f"Garcia n={n_subj}, {len(df)} trials (rt>={args.rt_min}).", flush=True)

    if args.model == 'basic':
        m = DDMMagnitudeComparisonModel(
            paradigm=df, fit_separate_evidence_sd=True, fit_prior=True)
    else:
        m = DDMMagnitudeComparisonRegressionModel(
            paradigm=df, fit_separate_evidence_sd=True, fit_prior=True,
            regressors={'n1_evidence_sd': 'isi_cat'})
    m.build_estimation_model(data=df, hierarchical=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{args.backend}_{args.chain_method}_t{args.tune}_s{args.seed}"
    out = args.out_dir / f"{tag}.tsv"
    rec = dict(model=args.model, backend=args.backend,
               chain_method=args.chain_method, tune=args.tune,
               draws=args.draws, seed=args.seed, n_subjects=n_subj,
               max_rhat=np.nan, min_ess=np.nan, divergences=np.nan,
               walltime_s=np.nan, status='')
    try:
        sk = dict(draws=args.draws, tune=args.tune, chains=4,
                  target_accept=args.target_accept, backend=args.backend,
                  chain_method=args.chain_method, random_seed=args.seed)
        if args.backend == 'blackjax':
            sk['progressbar'] = False
        t0 = time.time()
        idata = m.sample(**sk)
        rec['walltime_s'] = round(time.time() - t0, 1)
        have = [v for v in DIAG_VARS if v in idata.posterior]
        s = az.summary(idata, var_names=have or None, round_to=4)
        rec['max_rhat'] = float(s['r_hat'].max())
        rec['min_ess'] = int(s['ess_bulk'].min())
        rec['divergences'] = int(idata.sample_stats['diverging'].sum()) \
            if 'diverging' in idata.sample_stats else -1
        rec['status'] = 'ok'
        print(f"{tag}: r_hat={rec['max_rhat']:.3f} ess={rec['min_ess']} "
              f"div={rec['divergences']} ({rec['walltime_s']:.0f}s)", flush=True)
    except Exception as e:
        rec['status'] = f"FAILED: {type(e).__name__}: {str(e)[:120]}"
        print(rec['status'], flush=True)
        traceback.print_exc()

    pd.DataFrame([rec]).to_csv(out, sep='\t', index=False)
    print(f"-> wrote {out}", flush=True)


if __name__ == '__main__':
    main()
