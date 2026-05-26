"""One (model, backend, chain_method, tune, seed) cell of the pil03 DDM
sampler experiment — designed to be fanned out as a SLURM array so we can run
MANY seeds per config and measure convergence *rates*, not single runs.

Writes one TSV row to --out-dir/<tag>.tsv (filename encodes the config+seed so
array tasks never collide). Aggregate the rows afterwards.
"""
import argparse, time, traceback
from pathlib import Path
import numpy as np, pandas as pd, arviz as az
from bauer.models import DDMRiskModel, DDMRiskRegressionModel

DIAG = ['n1_evidence_sd', 'n2_evidence_sd', 'a', 't0',
        'safe_prior_mu', 'risky_prior_mu', 'safe_prior_sd', 'risky_prior_sd']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=('basic', 'regression'), required=True)
    ap.add_argument('--backend', choices=('numpyro', 'blackjax'), default='numpyro')
    ap.add_argument('--chain-method', choices=('vectorized', 'parallel'), default='vectorized')
    ap.add_argument('--tune', type=int, default=2000)
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--target-accept', type=float, default=0.99)
    ap.add_argument('--subject', default='pil03')
    ap.add_argument('--data', type=Path,
                    default=Path(__file__).resolve().parents[1].parent
                    / 'examples' / 'for_alina' / 'data' / 'pilot_data.tsv')
    ap.add_argument('--out-dir', type=Path,
                    default=Path(__file__).parent / 'seed_cells')
    args = ap.parse_args()

    df = pd.read_csv(args.data, sep='\t').set_index(['subject', 'trial'])
    df = df.loc[df.index.get_level_values('subject') == args.subject]
    df = df[(df['rt'] >= 0.20) & (df['rt'] <= 5.0)].copy()
    is_loss = df['domain'] == 'loss'
    df.loc[is_loss, 'choice'] = ~df.loc[is_loss, 'choice'].astype(bool)

    common = dict(prior_estimate='full', fit_seperate_evidence_sd=True,
                  fit_v_scale=False, fix_z=True)
    if args.model == 'basic':
        m = DDMRiskModel(paradigm=df.reset_index(), **common)
    else:
        m = DDMRiskRegressionModel(
            paradigm=df.reset_index(),
            regressors={'n1_evidence_sd': 'C(domain)',
                        'n2_evidence_sd': 'C(domain)', 'a': 'C(domain)'}, **common)
    m.build_estimation_model(data=df, hierarchical=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.subject}_{args.model}_{args.backend}_{args.chain_method}_t{args.tune}_s{args.seed}"
    rec = dict(subject=args.subject, model=args.model, backend=args.backend,
               chain_method=args.chain_method, tune=args.tune, draws=args.draws,
               seed=args.seed, max_rhat=np.nan, min_ess=np.nan,
               divergences=np.nan, walltime_s=np.nan, status='')
    try:
        sk = dict(draws=args.draws, tune=args.tune, chains=4,
                  target_accept=args.target_accept, backend=args.backend,
                  chain_method=args.chain_method, random_seed=args.seed)
        if args.backend == 'blackjax':
            sk['progressbar'] = False
        t0 = time.time()
        idata = m.sample(**sk)
        rec['walltime_s'] = round(time.time() - t0, 1)
        have = [v for v in DIAG if v in idata.posterior]
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
        traceback.print_exc()
    pd.DataFrame([rec]).to_csv(args.out_dir / f"{tag}.tsv", sep='\t', index=False)
    print(f"-> wrote {args.out_dir / (tag + '.tsv')}", flush=True)


if __name__ == '__main__':
    main()
