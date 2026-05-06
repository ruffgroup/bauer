"""Unified fitter for de Hollander et al. 2024 risky-choice data.

CLI:
    python -m bauer.scripts.fit_dehollander2024 <model> [--task {dotcloud,symbolic}]
        [--prior-estimate {objective,shared,full,klw}]
        [--v-scale {free,fixed}] [--n-subjects N|all]
        [--out-dir results/dehollander2024/]
        [--draws 1000] [--tune 1000] [--chains 2] [--cores 2]
        [--target-accept 0.99] [--seed 0]

``model`` ∈ {choice, ddm, rdm}. ``task`` selects dot-cloud (continuous-noisy
n) vs symbolic (Arabic-numeral) variant; default ``dotcloud``.

For risky choice the natural prior is ``--prior-estimate full`` (separate
fitted means+SDs for risky vs safe options) per the paper.

Output: ``<out_dir>/<task>/<n>subj/{model}{_freescale|_fixedscale}.nc``.
"""
import argparse
import os
import os.path as op
import warnings
warnings.filterwarnings('ignore')

from bauer.utils.data import load_dehollander2024_risk, load_dehollander2024_symbolic


def _progress(trace, draw):  # noqa: ARG001
    if draw.draw_idx % 100 == 0:
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
    ap.add_argument('--task', choices=['dotcloud', 'symbolic'], default='dotcloud')
    ap.add_argument('--prior-estimate', default='full',
                     choices=['objective', 'shared', 'full', 'klw'])
    ap.add_argument('--v-scale', choices=['free', 'fixed'], default='free')
    ap.add_argument('--n-subjects', default='all',
                     help='Number of subjects: integer or "all"')
    ap.add_argument('--out-dir', default='results/dehollander2024')
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--tune', type=int, default=1000)
    ap.add_argument('--chains', type=int, default=4)
    ap.add_argument('--cores', type=int, default=4)
    ap.add_argument('--target-accept', type=float, default=0.95)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--backend', choices=['pymc', 'numpyro', 'blackjax'],
                     default='pymc')
    ap.add_argument('--no-ppc', action='store_true')
    ap.add_argument('--flex', action='store_true',
                     help='Use flexible-noise (B-spline) variant')
    ap.add_argument('--spline-order', type=int, default=5)
    args = ap.parse_args()

    fit_v_scale = (args.v_scale == 'free')

    if args.task == 'dotcloud':
        df = load_dehollander2024_risk()
    else:
        df = load_dehollander2024_symbolic()
    if args.n_subjects != 'all':
        n = int(args.n_subjects)
        subs = df.index.get_level_values('subject').unique()[:n]
        df = df.loc[df.index.get_level_values('subject').isin(subs)].copy()
    n_subj = df.index.get_level_values('subject').nunique()
    print(f'{len(df)} trials, {n_subj} subjects, task={args.task}, '
          f'model={args.model}, prior={args.prior_estimate}, '
          f'v_scale={args.v_scale}', flush=True)

    common = dict(paradigm=df, prior_estimate=args.prior_estimate,
                  fit_seperate_evidence_sd=True)
    if args.flex:
        common['spline_order'] = args.spline_order
    if args.model == 'choice':
        if args.flex:
            from bauer.models import FlexibleNoiseRiskModel as Cls
        else:
            from bauer.models import RiskModel as Cls
    elif args.model == 'ddm':
        if args.flex:
            from bauer.models import DDMFlexibleNoiseRiskModel as Cls
        else:
            from bauer.models import DDMRiskModel as Cls
        common['fit_v_scale'] = fit_v_scale
    elif args.model == 'rdm':
        if args.flex:
            from bauer.models import RaceDiffusionFlexibleNoiseRiskModel as Cls
        else:
            from bauer.models import RaceDiffusionRiskModel as Cls
    m = Cls(**common)

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
        from pymc.sampling.jax import sample_numpyro_nuts, sample_blackjax_nuts
        sampler = sample_numpyro_nuts if args.backend == 'numpyro' \
                                       else sample_blackjax_nuts
        with m.estimation_model:
            nuts_kwargs = dict(getattr(m, 'recommended_nuts_kwargs', {}))
            print(f'  nuts_kwargs: {nuts_kwargs or "(numpyro defaults)"}', flush=True)
            idata = sampler(
                draws=args.draws, tune=args.tune, chains=args.chains,
                target_accept=args.target_accept, random_seed=args.seed,
                chain_method='vectorized',
                nuts_kwargs=nuts_kwargs or None,
                progressbar=True,
            )

    flex_tag = '_flex' if args.flex else ''
    scale_tag = f'_{args.v_scale}scale' if args.model in ('ddm', 'rdm') else ''
    out_path = op.join(args.out_dir, args.task, f'{n_subj}subj',
                        f'{args.model}{flex_tag}{scale_tag}_{args.prior_estimate}.nc')
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
