"""Unified fitter for Barreto-Garcia 2022 magnitude-comparison data.

CLI:
    python -m bauer.scripts.fit_garcia <model> [--flex] [--v-scale {free,fixed}]
        [--n-subjects N|all] [--out-dir results/garcia/]
        [--draws 1000] [--tune 1000] [--chains 2] [--cores 2]
        [--target-accept 0.99] [--seed 0]

``model`` ∈ {choice, ddm, rdm}.

Canonical kwargs applied to every fit:
- ``fit_seperate_evidence_sd=True`` — asymmetric encoding noise drives the
  order effect (σ_1 > σ_2)
- ``fit_prior=True`` — fitted Bayesian prior on log(n) (or
  ``fit_prior_mu_only=True`` for RDM, to close the σ_p×ν ridge)
- ``fix_z=True`` — DDM starting point fixed at 0.5
- ``unit_sigma=True`` — RDM uses Tillman σ_acc=1 with v_scale-driven drift
- soft tight prior on ``a`` centered at 1.0; ``t0`` hard-capped at
  0.95·min(rt) per subject; drift positivity guard.

Output: ``<out_dir>/<n>subj/{model}{_flex}{_freescale|_fixedscale}.nc``
with per-trial PPC stored as a deterministic group on the same idata.
"""
import argparse
import os
import os.path as op
import warnings
warnings.filterwarnings('ignore')

from bauer.utils.data import load_garcia2022


def _progress(trace, draw):  # noqa: ARG001 — pymc passes trace as kwarg
    if draw.draw_idx % 200 == 0:
        phase = 'tune' if draw.tuning else 'draw'
        print(f'  chain {draw.chain} {phase} {draw.draw_idx}', flush=True)


def _safe_to_netcdf(idata, path):
    os.makedirs(op.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    idata.to_netcdf(tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', choices=['choice', 'ddm', 'rdm'])
    ap.add_argument('--flex', action='store_true',
                     help='Use flexible-noise (B-spline) variant')
    ap.add_argument('--v-scale', choices=['free', 'fixed'], default=None,
                     help='free or fixed v_scale. Default: free for static '
                          'DDM/RDM, fixed for flex (spline absorbs scale).')
    ap.add_argument('--n-subjects', default='8',
                     help='Number of subjects: integer or "all"')
    ap.add_argument('--out-dir', default='results/garcia',
                     help='Output directory (under cwd)')
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--tune', type=int, default=1000)
    ap.add_argument('--chains', type=int, default=2)
    ap.add_argument('--cores', type=int, default=2)
    ap.add_argument('--target-accept', type=float, default=0.99)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-advantage', action='store_true',
                     help='RDM only: turn OFF advantage decomposition '
                          '(default is advantage=True).')
    ap.add_argument('--backend', choices=['pymc', 'numpyro', 'blackjax'],
                     default='pymc',
                     help='Sampler backend. numpyro/blackjax run via JAX, '
                          'often 2-10x faster especially with --gres=gpu:1.')
    ap.add_argument('--no-ppc', action='store_true',
                     help='Skip PPC computation after sampling')
    args = ap.parse_args()

    # Defaults for v_scale: free in static, fixed in flex
    if args.v_scale is None:
        args.v_scale = 'fixed' if args.flex else 'free'
    fit_v_scale = (args.v_scale == 'free')

    df = load_garcia2022(task='magnitude')
    if args.n_subjects != 'all':
        n = int(args.n_subjects)
        subs = df.index.get_level_values('subject').unique()[:n]
        df = df.loc[df.index.get_level_values('subject').isin(subs)].copy()
    n_subj = df.index.get_level_values('subject').nunique()
    print(f'{len(df)} trials, {n_subj} subjects, model={args.model}, '
          f'flex={args.flex}, v_scale={args.v_scale}', flush=True)

    # Build the model
    if args.model == 'choice':
        if args.flex:
            from bauer.models import FlexibleNoiseComparisonModel as Cls
            kwargs = dict(paradigm=df, fit_seperate_evidence_sd=True,
                          polynomial_order=5, fit_prior=True)
        else:
            from bauer.models import MagnitudeComparisonModel as Cls
            kwargs = dict(paradigm=df, fit_seperate_evidence_sd=True,
                          fit_prior=True)
    elif args.model == 'ddm':
        if args.flex:
            from bauer.models import DDMFlexibleNoiseComparisonModel as Cls
            kwargs = dict(paradigm=df, fit_seperate_evidence_sd=True,
                          polynomial_order=5, fit_prior=True,
                          fit_v_scale=fit_v_scale)
        else:
            from bauer.models import DDMMagnitudeComparisonModel as Cls
            kwargs = dict(paradigm=df, fit_seperate_evidence_sd=True,
                          fit_prior=True, fit_v_scale=fit_v_scale)
    elif args.model == 'rdm':
        adv = not args.no_advantage
        if args.flex:
            from bauer.models import RaceDiffusionFlexibleNoiseComparisonModel as Cls
            kwargs = dict(paradigm=df, fit_seperate_evidence_sd=True,
                          polynomial_order=5, fit_prior=True, advantage=adv)
        else:
            from bauer.models import RaceDiffusionMagnitudeComparisonModel as Cls
            kwargs = dict(paradigm=df, fit_seperate_evidence_sd=True,
                          fit_prior=True, advantage=adv)
    m = Cls(**kwargs)

    # Build the PyMC model. Some classes use ``data=``, others ``paradigm=``.
    try:
        m.build_estimation_model(data=df, hierarchical=True)
    except TypeError:
        m.build_estimation_model(paradigm=df, hierarchical=True)

    print(f'Sampling (backend={args.backend})...', flush=True)
    if args.backend == 'pymc':
        idata = m.sample(
            draws=args.draws, tune=args.tune, chains=args.chains, cores=args.cores,
            target_accept=args.target_accept, random_seed=args.seed,
            progressbar=False, callback=_progress,
        )
    else:
        # JAX-backed NUTS via numpyro or blackjax. Both run all chains in
        # parallel within a single process; honors --chains but ignores
        # --cores / --callback (no per-step Python callback hooks).
        from pymc.sampling.jax import sample_numpyro_nuts, sample_blackjax_nuts
        sampler = sample_numpyro_nuts if args.backend == 'numpyro' \
                                       else sample_blackjax_nuts
        with m.estimation_model:
            idata = sampler(
                draws=args.draws, tune=args.tune, chains=args.chains,
                target_accept=args.target_accept, random_seed=args.seed,
                progressbar=False,
            )

    # Compose output filename
    flex_tag = '_flex' if args.flex else ''
    if args.model == 'ddm':
        suffix = f'_{args.v_scale}scale'
    elif args.model == 'rdm':
        suffix = '_noadvantage' if args.no_advantage else ''
    else:
        suffix = ''
    out_path = op.join(args.out_dir, f'{n_subj}subj',
                        f'{args.model}{flex_tag}{suffix}.nc')
    _safe_to_netcdf(idata, out_path)
    print(f'idata -> {out_path}', flush=True)

    if not args.no_ppc:
        print('Computing PPC...', flush=True)
        if args.model == 'choice':
            ppc = m.ppc(df, idata, progressbar=False)
        else:
            ppc = m.ppc(df, idata, n_posterior_samples=200,
                         inner_samples=1, random_seed=args.seed,
                         progressbar=False)
        ppc_path = out_path.replace('.nc', '_ppc.parquet')
        ppc.to_parquet(ppc_path)
        print(f'ppc -> {ppc_path}', flush=True)

    print('Done.', flush=True)


if __name__ == '__main__':
    main()
