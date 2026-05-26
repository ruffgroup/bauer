"""Controlled sampler experiment for bauer risky-choice DDMs.

Question: why does a *regression* DDM fail to converge on numpyro
`chain_method='vectorized'` while a *basic* DDM is fine — and is the fix
really `chain_method`, or is it warmup length / backend / seed luck?

Testbed: the `pil03` pilot subject (examples/for_alina/data/pilot_data.tsv),
a near-chance-in-losses risky-choice subject with RTs — the proven-hard
regression case, small enough (~314 trials) to fit in minutes on a GPU so we
can sweep a real grid. We fit the SAME data with a basic vs a regression DDM
and vary backend × chain_method × tune × seed, recording r̂ / ESS / divergences
/ wall-time into a TSV.

Run (cluster GPU):
    python notes/experiments/run_ddm_sampler_experiment.py \
        --out notes/experiments/ddm_sampler_results.tsv

Each cell is wrapped in try/except so one crash (e.g. a blackjax/JAX issue)
doesn't abort the sweep — the failure is recorded in the `status` column.
"""
import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az

from bauer.models import DDMRiskModel, DDMRiskRegressionModel

# (model, backend, chain_method, tune, draws, seed)
# Crossed to answer: (1) basic vs regression under identical settings;
# (2) tune 2000 vs 4000; (3) vectorized vs parallel at fixed tune;
# (4) seed sensitivity at the suspect cell; (5) blackjax (with progressbar
# off, the documented workaround).
CONFIGS = [
    ('regression', 'numpyro',  'vectorized', 2000, 1000, 0),
    ('regression', 'numpyro',  'vectorized', 4000, 2000, 0),
    ('regression', 'numpyro',  'vectorized', 4000, 2000, 1),
    ('regression', 'numpyro',  'vectorized', 4000, 2000, 2),
    ('regression', 'numpyro',  'parallel',   4000, 2000, 0),
    ('regression', 'numpyro',  'parallel',   4000, 2000, 1),
    ('regression', 'blackjax', 'vectorized', 4000, 2000, 0),
    ('regression', 'blackjax', 'parallel',   4000, 2000, 0),
    ('basic',      'numpyro',  'vectorized', 2000, 1000, 0),
    ('basic',      'numpyro',  'vectorized', 4000, 2000, 0),
    ('basic',      'numpyro',  'parallel',   4000, 2000, 0),
]

# Parameters to base convergence on (intersected with what the model has).
DIAG_VARS = ['n1_evidence_sd', 'n2_evidence_sd', 'a', 't0',
             'safe_prior_mu', 'risky_prior_mu', 'safe_prior_sd', 'risky_prior_sd']


def prepare(df):
    """Flip choice on loss trials; keep presentation order (Alina convention)."""
    out = df.copy()
    is_loss = out['domain'] == 'loss'
    out.loc[is_loss, 'choice'] = ~out.loc[is_loss, 'choice'].astype(bool)
    return out


def build(model_kind, df):
    common = dict(prior_estimate='full', fit_seperate_evidence_sd=True,
                  fit_v_scale=False, fix_z=True)
    if model_kind == 'basic':
        m = DDMRiskModel(paradigm=df.reset_index(), **common)
    else:
        m = DDMRiskRegressionModel(
            paradigm=df.reset_index(),
            regressors={'n1_evidence_sd': 'C(domain)',
                        'n2_evidence_sd': 'C(domain)',
                        'a':              'C(domain)'},
            **common)
    m.build_estimation_model(data=df, hierarchical=False)
    return m


def diagnostics(idata):
    have = [v for v in DIAG_VARS if v in idata.posterior]
    s = az.summary(idata, var_names=have or None, round_to=4)
    nd = int(idata.sample_stats['diverging'].sum()) \
        if 'diverging' in idata.sample_stats else -1
    return float(s['r_hat'].max()), int(s['ess_bulk'].min()), nd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path,
                    default=Path(__file__).resolve().parents[1].parent
                    / 'examples' / 'for_alina' / 'data' / 'pilot_data.tsv')
    ap.add_argument('--subject', default='pil03')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent / 'ddm_sampler_results.tsv')
    ap.add_argument('--target-accept', type=float, default=0.99)
    args = ap.parse_args()

    df = pd.read_csv(args.data, sep='\t').set_index(['subject', 'trial'])
    df = df.loc[df.index.get_level_values('subject') == args.subject]
    df = df[(df['rt'] >= 0.20) & (df['rt'] <= 5.0)]
    df = prepare(df)
    print(f"Testbed: subject={args.subject}, {len(df)} trials, "
          f"P(choice|domain after flip)={df.groupby('domain')['choice'].mean().round(3).to_dict()}",
          flush=True)

    cols = ['model', 'backend', 'chain_method', 'tune', 'draws', 'seed',
            'max_rhat', 'min_ess', 'divergences', 'walltime_s', 'status']
    rows = []
    for (model_kind, backend, chain_method, tune, draws, seed) in CONFIGS:
        tag = f"{model_kind}/{backend}/{chain_method}/tune={tune}/seed={seed}"
        print(f"\n=== {tag} ===", flush=True)
        rec = dict(model=model_kind, backend=backend, chain_method=chain_method,
                   tune=tune, draws=draws, seed=seed,
                   max_rhat=np.nan, min_ess=np.nan, divergences=np.nan,
                   walltime_s=np.nan, status='')
        try:
            m = build(model_kind, df)
            sk = dict(draws=draws, tune=tune, chains=4,
                      target_accept=args.target_accept, backend=backend,
                      chain_method=chain_method, random_seed=seed)
            if backend == 'blackjax':
                sk['progressbar'] = False   # documented vmap-of-cond workaround
            t0 = time.time()
            idata = m.sample(**sk)
            rec['walltime_s'] = round(time.time() - t0, 1)
            rec['max_rhat'], rec['min_ess'], rec['divergences'] = diagnostics(idata)
            rec['status'] = 'ok'
            print(f"  r_hat={rec['max_rhat']:.3f} ess={rec['min_ess']} "
                  f"div={rec['divergences']} ({rec['walltime_s']:.0f}s)", flush=True)
        except Exception as e:
            rec['status'] = f"FAILED: {type(e).__name__}: {str(e)[:120]}"
            print(f"  {rec['status']}", flush=True)
            traceback.print_exc()
        rows.append(rec)
        # Write incrementally so partial results survive a crash/timeout.
        pd.DataFrame(rows, columns=cols).to_csv(args.out, sep='\t', index=False)
        print(f"  -> wrote {args.out}", flush=True)

    print("\nDONE. Results:")
    print(pd.DataFrame(rows, columns=cols).to_string(index=False))


if __name__ == '__main__':
    main()
