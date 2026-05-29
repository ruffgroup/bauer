"""Fit ``RiskLapseRegressionModel`` to the simulated gain/loss data.

Data convention used here (matches the user's experimental setup):

- ``n1, n2, p1, p2, choice`` are kept in **presentation order** — ``n1`` is the
  first-presented option, regardless of whether it's the safe or risky one.
  This lets bauer's separate ``n1_evidence_sd`` vs ``n2_evidence_sd`` absorb
  working-memory degradation on the first option.

- **Loss trials**: ``choice`` is flipped (``choice_new = NOT choice``). This
  unifies the perceptual decision across domains. From the subject's POV,
  gain trials ask "pick the option with the larger EV" and loss trials ask
  "pick the option with the smaller |EV|" — exact opposites. Flipping
  ``choice`` in losses re-aligns these so "chose option 2" consistently
  means "took the action that *raises* expected value". The slope of the
  psychometric (going up with ``log(EU_2/EU_1)``) is then consistent across
  domains and a single bauer ``RiskModel`` likelihood fits both with shared
  parameters.

Regressors:

- ``n1_evidence_sd``  — encoding noise on first-presented option (per domain)
- ``n2_evidence_sd``  — encoding noise on second-presented option (per domain)
- ``p_lapse``         — lapse rate (per domain)

Risk attitudes are captured implicitly via the Bayesian observer with
``prior_estimate='full'`` (KLW mechanism: asymmetric prior shrinkage between
safe and risky options).

Usage::

    python fit_psychometric.py \\
        --data data/pilot_data.tsv \\
        --out  results/psychometric_idata.nc \\
        --draws 1000 --tune 1500 --chains 4 --backend numpyro
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bauer.models import RiskLapseRegressionModel


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
                        default=Path(__file__).parent / 'results' / 'psychometric_bayes_idata.nc')
    parser.add_argument('--draws',  type=int, default=1000)
    parser.add_argument('--tune',   type=int, default=1500)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--target-accept', type=float, default=0.95)
    parser.add_argument('--backend', choices=('pymc', 'numpyro', 'blackjax'),
                        default='numpyro')
    parser.add_argument('--n-subjects', type=int, default=None,
                        help='If set, fit only the first N subjects (for debugging).')
    parser.add_argument('--prior-estimate', default='full',
                        choices=('objective', 'shared', 'full', 'klw'))
    args = parser.parse_args()

    df = pd.read_csv(args.data, sep='\t').set_index(['subject', 'trial'])
    if args.n_subjects is not None:
        keep = sorted(df.index.get_level_values('subject').unique())[:args.n_subjects]
        df = df.loc[df.index.get_level_values('subject').isin(keep)]
        print(f"Subsetting to first {args.n_subjects} subjects ({len(df)} trials).")

    df = prepare(df)

    print(f"\nFitting RiskLapseRegressionModel on "
          f"{df.index.get_level_values('subject').nunique()} subjects × "
          f"{len(df)} trials.")
    print("Domain levels:", sorted(df['domain'].unique()))
    print("P(choice=True) by domain after flip:",
          df.groupby('domain')['choice'].mean().round(3).to_dict())

    model = RiskLapseRegressionModel(
        paradigm=df.reset_index(),
        regressors={
            'n1_evidence_sd': 'C(domain)',
            'n2_evidence_sd': 'C(domain)',
            'p_lapse':        'C(domain)',
        },
        prior_estimate=args.prior_estimate,
    )
    model.build_estimation_model(data=df, hierarchical=True)

    print(f"\nSampling: backend={args.backend}, "
          f"draws={args.draws}, tune={args.tune}, chains={args.chains}, "
          f"target_accept={args.target_accept}")
    t0 = time.time()
    idata = model.sample(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        backend=args.backend,
    )
    elapsed = time.time() - t0
    print(f"Sampling done in {elapsed:.1f} s.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(args.out)
    print(f"Saved idata → {args.out}")

    import arviz as az
    summary = az.summary(idata,
                         var_names=['n1_evidence_sd_mu', 'n2_evidence_sd_mu',
                                    'p_lapse_mu'],
                         round_to=3)
    print("\nGroup-level posterior summary:")
    print(summary)


if __name__ == '__main__':
    sys.exit(main())
