"""Unified fitter for the de Hollander TMS-risk experiment.

CLI:
    python -m bauer.scripts.fit_dehollander_tms <model> [--regression]
        [--prior-estimate {objective,shared,full,klw}]
        [--v-scale {free,fixed}] [--n-subjects N|all]
        [--out-dir results/dehollander_tms/]
        [--all-subjects-no-tms]   # fall back to all 73 subjects, all sessions
        [--draws 1000] [--tune 1000] [--chains 2] [--cores 2]
        [--target-accept 0.99] [--seed 0]

``model`` ∈ {choice, ddm, rdm}.

By default loads the 35 TMS subjects in sessions 2/3 (``tms_only=True``).
Pass ``--all-subjects-no-tms`` to use all 73 subjects across all sessions.

``--regression`` activates the regression variant of the choice model with
the TMS condition as a covariate on the noise/aversion parameters. Currently
only implemented for ``choice`` (DDM/RDM regression versions to come).
"""
import argparse
import os
import os.path as op
import warnings
warnings.filterwarnings('ignore')

from bauer.utils.data import load_dehollander_tms_risk


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
    ap.add_argument('--regression', action='store_true',
                     help='Use the RegressionModel variant with stimulation_condition '
                          'as a covariate. Choice + non-flex available; '
                          'flex variants of DDM/RDM regression now supported.')
    ap.add_argument('--flex', action='store_true',
                     help='Use flexible-noise (B-spline) variant')
    ap.add_argument('--spline-order', type=int, default=5)
    ap.add_argument(
        '--reg-on', default='n1_evidence_sd,n2_evidence_sd',
        help='Comma-separated list of parameters to regress on stimulation_condition. '
             'Default: only encoding noise (n1/n2_evidence_sd). Pass a longer list '
             '(e.g. "n1_evidence_sd,n2_evidence_sd,risky_prior_mu,risky_prior_sd,'
             'safe_prior_mu,safe_prior_sd") for the most permissive spec.'
    )
    ap.add_argument('--prior-estimate', default='full',
                     choices=['objective', 'shared', 'full', 'klw'])
    ap.add_argument('--v-scale', choices=['free', 'fixed'], default='free')
    ap.add_argument('--n-subjects', default='all',
                     help='Number of subjects: integer or "all"')
    ap.add_argument('--all-subjects-no-tms', action='store_true',
                     help='Use all 73 subjects across all sessions '
                          '(default: 35 TMS subjects, sessions 2/3)')
    ap.add_argument('--out-dir', default='results/dehollander_tms')
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--tune', type=int, default=1000)
    ap.add_argument('--chains', type=int, default=4)
    ap.add_argument('--cores', type=int, default=4)
    ap.add_argument('--target-accept', type=float, default=0.95)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--backend', choices=['pymc', 'numpyro', 'blackjax'],
                     default='pymc')
    ap.add_argument('--no-ppc', action='store_true')
    args = ap.parse_args()

    fit_v_scale = (args.v_scale == 'free')
    df = load_dehollander_tms_risk(tms_only=not args.all_subjects_no_tms)
    if args.n_subjects != 'all':
        n = int(args.n_subjects)
        subs = df.index.get_level_values('subject').unique()[:n]
        df = df.loc[df.index.get_level_values('subject').isin(subs)].copy()
    n_subj = df.index.get_level_values('subject').nunique()
    print(f'{len(df)} trials, {n_subj} subjects, model={args.model}, '
          f'regression={args.regression}, prior={args.prior_estimate}, '
          f'v_scale={args.v_scale}', flush=True)

    # Patsy regression needs `stimulation_condition` as a column, not an
    # index level. Promote it for regression fits.
    df_use = df.reset_index().set_index('subject') if args.regression else df

    common = dict(paradigm=df_use, prior_estimate=args.prior_estimate,
                  fit_seperate_evidence_sd=True)
    if args.flex:
        common['spline_order'] = args.spline_order

    if args.regression:
        reg_params = [p.strip() for p in args.reg_on.split(',') if p.strip()]
        regressors = {p: 'stimulation_condition' for p in reg_params}
        common['regressors'] = regressors

        if args.model == 'choice':
            if args.flex:
                from bauer.models import FlexibleNoiseRiskRegressionModel as Cls
            else:
                from bauer.models import RiskRegressionModel as Cls
        elif args.model == 'ddm':
            if not args.flex:
                raise NotImplementedError(
                    'Non-flex DDM regression on risk not implemented yet. '
                    'Use --flex.'
                )
            from bauer.models import DDMFlexibleNoiseRiskRegressionModel as Cls
            common['fit_v_scale'] = fit_v_scale
        elif args.model == 'rdm':
            if not args.flex:
                raise NotImplementedError(
                    'Non-flex RDM regression on risk not implemented yet. '
                    'Use --flex.'
                )
            from bauer.models import RaceDiffusionFlexibleNoiseRiskRegressionModel as Cls
    else:
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
        m.build_estimation_model(data=df_use, hierarchical=True)
    except TypeError:
        m.build_estimation_model(paradigm=df_use, hierarchical=True)

    print(f'Sampling (backend={args.backend})...', flush=True)
    sample_kwargs = dict(
        draws=args.draws, tune=args.tune, chains=args.chains,
        target_accept=args.target_accept, random_seed=args.seed,
        backend=args.backend,
    )
    if args.backend == 'pymc':
        sample_kwargs.update(cores=args.cores, progressbar=False, callback=_progress)
    else:
        sample_kwargs.update(chain_method='vectorized', progressbar=True)
    idata = m.sample(**sample_kwargs)

    flex_tag = '_flex' if args.flex else ''
    reg_tag = '_reg' if args.regression else ''
    scale_tag = f'_{args.v_scale}scale' if args.model in ('ddm', 'rdm') else ''
    out_path = op.join(args.out_dir, f'{n_subj}subj',
                        f'{args.model}{flex_tag}{reg_tag}{scale_tag}_{args.prior_estimate}.nc')
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
        ppc_path = out_path.replace('.nc', '_ppc.pickle')
        ppc.to_pickle(ppc_path)
        print(f'ppc -> {ppc_path}', flush=True)

    print('Done.', flush=True)


if __name__ == '__main__':
    main()
