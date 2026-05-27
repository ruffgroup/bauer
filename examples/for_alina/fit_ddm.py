"""Fit ``DDMRiskRegressionModel`` to the simulated gain/loss data.

Same data convention as ``fit_psychometric_bayes.py``: ``n1, n2, p1, p2``
in **presentation order**, ``choice`` flipped on loss trials. The flip
unifies the two domains under one drift formula:

    v = v_scale * ((post_log_n_2 - post_log_n_1) + log(p_2/p_1)) / √(σ_1² + σ_2²)

For *gains*, ``v > 0`` ⟺ option 2 has higher EU ⟺ EV-optimal subject picks
option 2 ⟺ ``choice = True``. For *losses* under the flip,
``flipped_choice = True`` ⟺ the subject picked option 1 in the actual data
⟺ the smaller-loss option ⟺ the EV-preferred option in losses. So the
same drift formula's "v > 0 ⟹ choice = True" semantics still hold.

Probabilities enter as a deterministic additive shift of the log-magnitude
comparison — same threshold as the static probit, no multiplicative EV
weighting. Bauer's risky-DDM has **no explicit risk premium parameter**;
apparent risk attitudes emerge from encoding noise × prior shrinkage (KLW).

Regressors on ``domain`` for: ``n1_evidence_sd``, ``n2_evidence_sd``, and
``a`` (boundary separation).

Usage::

    python fit_ddm.py \\
        --data data/pilot_data.tsv \\
        --out  results/ddm_idata.nc \\
        --draws 1000 --tune 2000 --chains 4 --backend numpyro
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pymc as pm
from bauer.models import DDMRiskRegressionModel


def prepare(df):
    """Flip ``choice`` in loss trials. Keeps presentation order intact."""
    out = df.copy()
    is_loss = out['domain'] == 'loss'
    out.loc[is_loss, 'choice'] = ~out.loc[is_loss, 'choice'].astype(bool)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data', type=Path,
                        default=Path(__file__).parent / 'data' / 'pilot_data.tsv')
    parser.add_argument('--out',  type=Path,
                        default=Path(__file__).parent / 'results' / 'ddm_idata.nc')
    parser.add_argument('--draws',  type=int, default=1000)
    parser.add_argument('--tune',   type=int, default=1500)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--target-accept', type=float, default=0.95)
    parser.add_argument('--backend', choices=('pymc', 'numpyro', 'blackjax'),
                        default='numpyro')
    parser.add_argument('--per-subject', action='store_true',
                        help='Fit each subject separately (complete no-pooling). '
                             'Rarely what you want for real data. DEFAULT is a '
                             'hierarchical fit (partial pooling) over all '
                             'subjects — the realistic choice, and what the '
                             'starting-point finder makes converge even at small N.')
    parser.add_argument('--n-subjects', type=int, default=None,
                        help='If set, fit only the first N subjects (for debugging).')
    parser.add_argument('--subject', default=None,
                        help='If set, fit only this specific subject (e.g. pil03). '
                             'Overrides --n-subjects.')
    parser.add_argument('--chain-method', default='vectorized',
                        choices=('vectorized', 'parallel'),
                        help='numpyro chain_method. parallel is more robust to '
                             'bad seeds (independent RNG per chain) but uses more VRAM.')
    parser.add_argument('--slice', action='store_true',
                        help=('Use pm.Slice (gradient-free, CPU-only) instead of '
                              'NUTS. Immune to the WFPT LOGP_LB flat-floor '
                              'pathology that traps NUTS when t0 wanders above '
                              'min(rt). Slow (hours) but bulletproof; use as a '
                              'reference when NUTS will not converge.'))
    parser.add_argument('--t0-domain', action='store_true',
                        help=('Also regress t0 on C(domain). OFF by default: t0 is '
                              'motor/sensory delay with no strong reason to differ '
                              'gain-vs-loss, and regressing it leaves the model '
                              'under-identified on low-N / near-chance subjects '
                              '(this is what broke pil03 — see README). Only enable '
                              'with enough data to identify a domain-specific t0.'))
    parser.add_argument('--rt-floor', type=float, default=0.20,
                        help='Drop trials with rt below this (seconds). '
                             'Required to avoid the HSSM t0 gradient floor.')
    parser.add_argument('--rt-ceiling', type=float, default=5.0,
                        help='Drop trials with rt above this (seconds). '
                             'Removes slow outliers/disengaged trials.')
    args = parser.parse_args()

    df = pd.read_csv(args.data, sep='\t').set_index(['subject', 'trial'])
    if args.subject is not None:
        df = df.loc[df.index.get_level_values('subject') == args.subject]
        print(f"Filtering to subject {args.subject} ({len(df)} trials).")
    elif args.n_subjects is not None:
        keep = sorted(df.index.get_level_values('subject').unique())[:args.n_subjects]
        df = df.loc[df.index.get_level_values('subject').isin(keep)]
        print(f"Subsetting to first {args.n_subjects} subjects ({len(df)} trials).")

    # Trim RT outliers
    pre = len(df)
    df = df[(df['rt'] >= args.rt_floor) & (df['rt'] <= args.rt_ceiling)]
    print(f"Dropped {pre - len(df)} RT outliers "
          f"(kept {args.rt_floor:.2f} ≤ rt ≤ {args.rt_ceiling:.2f}s; "
          f"{len(df)} remaining).")

    df = prepare(df)

    n_subj = df.index.get_level_values('subject').nunique()
    print(f"\nFitting DDMRiskRegressionModel on {n_subj} subjects × {len(df)} "
          f"trials ({'per-subject' if args.per_subject else 'HIERARCHICAL'}).")
    print("P(choice=True) by domain after flip:",
          df.groupby('domain')['choice'].mean().round(3).to_dict())

    # Regress the cognitive parameters that can plausibly differ by domain:
    # n1/n2 encoding noise and the boundary a. t0 (motor delay) is NOT
    # regressed by default — regressing it under-identifies weak subjects;
    # add it with --t0-domain only if you have a reason and the data to.
    import arviz as az
    regressors = {'n1_evidence_sd': 'C(domain)',
                  'n2_evidence_sd': 'C(domain)',
                  'a':              'C(domain)'}
    if args.t0_domain:
        regressors['t0'] = 'C(domain)'
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def make_model(paradigm):
        return DDMRiskRegressionModel(
            paradigm=paradigm.reset_index(), regressors=regressors,
            prior_estimate='full', fit_seperate_evidence_sd=True,
            fit_v_scale=False, fix_z=True)

    def fit(model):
        if args.slice:                       # gradient-free CPU reference
            with model.estimation_model:
                return pm.sample(draws=args.draws, tune=args.tune,
                                 chains=args.chains, step=pm.Slice(),
                                 return_inferencedata=True, progressbar=False)
        sk = dict(draws=args.draws, tune=args.tune, chains=args.chains,
                  target_accept=args.target_accept, backend=args.backend)
        if args.backend in ('numpyro', 'blackjax'):
            sk['chain_method'] = args.chain_method
        return model.sample(**sk)            # starting-point finder is on by default

    if not args.per_subject:
        # ── Default: ONE hierarchical model (partial pooling) — realistic ──
        model = make_model(df)
        model.build_estimation_model(data=df, hierarchical=True)
        t0 = time.time(); idata = fit(model)
        print(f"  done in {time.time()-t0:.1f}s")
        s = az.summary(idata, var_names=['n1_evidence_sd_mu', 'n2_evidence_sd_mu',
                                         'a_mu', 't0_mu'], round_to=3)
        print(f"  max r̂={s['r_hat'].max():.3f}, min ESS={s['ess_bulk'].min():.0f}")
        idata.to_netcdf(args.out)
        print(f"  saved → {args.out}")
    else:
        # ── Pilot (tiny N): fit each subject alone; one netCDF per subject ──
        for subj in sorted(df.index.get_level_values('subject').unique()):
            df_subj = df.xs(subj, level='subject', drop_level=False)
            print(f"\n--- subject={subj}  ({len(df_subj)} trials) ---")
            model = make_model(df_subj)
            model.build_estimation_model(data=df_subj, hierarchical=False)
            t0 = time.time(); idata_s = fit(model)
            print(f"  done in {time.time()-t0:.1f}s")
            s = az.summary(idata_s, var_names=['n1_evidence_sd', 'n2_evidence_sd',
                                               'a', 't0'], round_to=3)
            print(f"  max r̂={s['r_hat'].max():.3f}, min ESS={s['ess_bulk'].min():.0f}")
            out_path = args.out.with_name(f'ddm_idata_{subj}.nc')
            idata_s.to_netcdf(out_path)
            print(f"  saved → {out_path}")


if __name__ == '__main__':
    sys.exit(main())
