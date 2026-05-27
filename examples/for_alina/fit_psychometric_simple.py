"""Fit a simple ``PsychophysicalLapseRegressionModel`` to the gain/loss data.

The Bayesian-observer model in :mod:`fit_psychometric_bayes` is the
"theoretically correct" model for this data. This script provides the
**simplest possible** psychometric fit as a complement — useful as a
sanity check, a starting point, and a quick model that runs in seconds.

The trick to making ``PsychophysicalLapseRegressionModel`` work for risky
choice is to encode the stimulus axis as **log expected utility**:

    x1 = log(n1 * p1)     # log-EV of first-presented option
    x2 = log(n2 * p2)     # log-EV of second-presented option

Then the model fits

    P(chose option 2) = (1-λ) Φ((x2 - x1 + bias) / (√2 ν)) + λ/2

and ``bias`` has a clean interpretation: deviation from EV-optimality in the
direction of preferring option 1. For a risk-neutral subject, ``bias = 0``.
Negative ``bias`` means "less likely to choose the higher-EV option" — i.e.
risk-averse in gains (with the loss-trial choice flip, also risk-seeking in
losses gets folded into the same negative bias under the unified coding).

This is the same data convention as the Bayesian model: keep ``n1, n2``
in **presentation order**, and **flip ``choice`` for loss trials** so that
the gain/loss task becomes the same perceptual question (pick the option
with higher EU).

Note the trade-off versus the Bayesian model: the psychometric model has a
single ``nu`` (no separation between first- and second-presented option
noise), so working-memory order effects are not captured. Use the Bayesian
model when that matters.

Usage::

    python fit_psychometric_simple.py \\
        --data data/pilot_data.tsv \\
        --out  results/psychometric_simple_idata.nc \\
        --draws 1000 --tune 1000 --chains 4 --backend numpyro
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bauer.models import PsychophysicalLapseRegressionModel


def prepare(df):
    """Compute log-EU stimulus axes; flip ``choice`` in loss trials."""
    out = df.copy()
    is_loss = out['domain'] == 'loss'
    out.loc[is_loss, 'choice'] = ~out.loc[is_loss, 'choice'].astype(bool)
    out['x1'] = np.log(out['n1'].astype(float) * out['p1'].astype(float))
    out['x2'] = np.log(out['n2'].astype(float) * out['p2'].astype(float))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data', type=Path,
                        default=Path(__file__).parent / 'data' / 'pilot_data.tsv')
    parser.add_argument('--out',  type=Path,
                        default=Path(__file__).parent / 'results' / 'psychometric_simple_idata.nc')
    parser.add_argument('--draws',  type=int, default=1000)
    parser.add_argument('--tune',   type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--target-accept', type=float, default=0.95)
    parser.add_argument('--backend', choices=('pymc', 'numpyro', 'blackjax'),
                        default='numpyro')
    parser.add_argument('--n-subjects', type=int, default=None,
                        help='If set, fit only the first N subjects (for debugging).')
    args = parser.parse_args()

    df = pd.read_csv(args.data, sep='\t').set_index(['subject', 'trial'])
    if args.n_subjects is not None:
        keep = sorted(df.index.get_level_values('subject').unique())[:args.n_subjects]
        df = df.loc[df.index.get_level_values('subject').isin(keep)]
        print(f"Subsetting to first {args.n_subjects} subjects ({len(df)} trials).")

    df = prepare(df)

    print(f"\nFitting PsychophysicalLapseRegressionModel on "
          f"{df.index.get_level_values('subject').nunique()} subjects × "
          f"{len(df)} trials.")
    print("P(choice=True) by domain after flip:",
          df.groupby('domain')['choice'].mean().round(3).to_dict())

    model = PsychophysicalLapseRegressionModel(
        paradigm=df.reset_index(),
        regressors={
            'nu':      'C(domain)',
            'bias':    'C(domain)',
            'p_lapse': 'C(domain)',
        },
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
    summary = az.summary(idata, var_names=['nu_mu', 'bias_mu', 'p_lapse_mu'],
                         round_to=3)
    print("\nGroup-level posterior summary:")
    print(summary)


if __name__ == '__main__':
    sys.exit(main())
