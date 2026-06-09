"""Incremental DDM convergence debugging on the dyscalculic magnitude data.

This is the Task-B driver for getting the bauer DDM (and DDM+lapse) to CONVERGE
on the real dyscalculic magnitude-comparison data. It walks the debugging ladder
one config at a time (smallest/cheapest first), reporting r-hat / divergences /
ESS for each, and is meant to be run on a GPU compute node (never the login
node). Each ``--step`` is one self-contained fit.

Data
----
``magjudge_behavior_DNumRisk.csv`` (path via ``--data``). Filtered to
``rt >= 0.20``; ``choice = chose_n2``; multi-indexed by ``[subject, run,
trial_nr]``. Use ``--n-subjects 8`` while debugging, then scale to all 66.

Ladder (``--step``)
-------------------
1  ``plain``       Plain DDM, no lapse. Team's ``ddm_group`` config:
                   fit_separate_evidence_sd=True, memory_model='shared_perceptual_noise',
                   v_scale NEVER fit. mapjitter init. Baseline.
2  ``plain_pf``    Same as (1) but ``find_init='pathfinder'``. The key test.
3  ``indep``       Same as (1/2) but memory_model='independent' (rotates the
                   -0.81 sigma_perc<->sigma_mem ridge onto axes). ``--init`` picks
                   mapjitter|pathfinder.
4  ``lapse``       Add the RT-aware lapse (Beta group prior, regularized DOWN via
                   ``--beta-mu-mean`` ~0.02-0.03) on top of whatever converged
                   best. ``--memory-model`` and ``--init`` configurable.
5  scale           Re-run the winning config with ``--n-subjects 0`` (= all 66).

Guardrails (do NOT violate)
---------------------------
- v_scale stays FIXED (fit_v_scale=False) — perfect v_scale<->evidence_sd degeneracy.
- Plain DDM only; never the sv-DDM.
- target_accept defaults to 0.99 (hierarchical DDM); dense_mass via the model's
  recommended_nuts_kwargs.

Usage (on a GPU compute node)::

    PY=~/data/conda/envs/bauer_cuda/bin/python
    $PY fit_real_ddm.py --step plain    --n-subjects 8 --out ~/ddm_plain.json
    $PY fit_real_ddm.py --step plain_pf --n-subjects 8 --out ~/ddm_plain_pf.json
    $PY fit_real_ddm.py --step indep    --init pathfinder --n-subjects 8 \
        --out ~/ddm_indep_pf.json
    $PY fit_real_ddm.py --step lapse    --init pathfinder \
        --memory-model independent --beta-mu-mean 0.02 --n-subjects 8 \
        --out ~/ddm_lapse.json
    # scale the winner to all subjects:
    $PY fit_real_ddm.py --step indep --init pathfinder --n-subjects 0 \
        --out ~/ddm_indep_pf_all.json
"""
import argparse
import json
import os
import sys
import time

# Import THIS checkout's bauer (the feature/ddm-lapse submodule), not whatever
# is pip-installed in the env (e.g. ~/git/bauer), which lacks the lapse classes.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import arviz as az

from bauer.models import (
    DDMMagnitudeComparisonRegressionModel,
    DDMMagnitudeComparisonLapseRegressionModel,
)


def load_data(path, n_subjects, rt_floor=0.20):
    """Load + filter the dyscalculic magnitude data into bauer's convention."""
    df = pd.read_csv(path)
    df['choice'] = df['chose_n2'].astype(bool)
    df = df[df['rt'] >= rt_floor].copy()
    df = df.set_index(['subject', 'run', 'trial_nr'])
    if n_subjects and n_subjects > 0:
        keep = sorted(df.index.get_level_values('subject').unique())[:n_subjects]
        df = df.loc[df.index.get_level_values('subject').isin(keep)]
    n_subj = df.index.get_level_values('subject').nunique()
    print(f"Loaded {len(df)} trials x {n_subj} subjects "
          f"(rt>={rt_floor}); P(chose_n2)={df['choice'].mean():.3f}")
    return df


def build_model(df, step, memory_model, beta_mu_mean):
    """Construct the model for the requested ladder step.

    Regressors mirror the original blow-up config: perceptual & memory noise +
    boundary a. We regress an intercept-only (``'1'``) design here for the
    convergence ladder (no covariate) — swap in your group covariate once the
    base geometry converges. v_scale stays FIXED (guardrail).
    """
    # Intercept-only regressors keep the geometry identical to the plain
    # hierarchical model while exercising the regression code path the real
    # analysis uses. Replace '1' with e.g. 'C(group)' for the final analysis.
    regressors = {'n1_evidence_sd': '1', 'n2_evidence_sd': '1', 'a': '1'}

    common = dict(
        paradigm=df.reset_index(), regressors=regressors,
        fit_prior=True, fit_separate_evidence_sd=True,
        memory_model=memory_model, fit_v_scale=False, fix_z=True,
    )

    if step == 'lapse':
        m = DDMMagnitudeComparisonLapseRegressionModel(**common)
        # lapse_group / lapse_mu_mean are class attributes, not __init__ kwargs.
        # Set them, then re-derive free_parameters (computed in __init__).
        m.lapse_group = 'beta'
        m.lapse_mu_mean = beta_mu_mean
        m.free_parameters = m.get_free_parameters()
        return m
    return DDMMagnitudeComparisonRegressionModel(**common)


def diagnostics(idata, var_names):
    """Return max r-hat, total divergences, min bulk ESS over var_names."""
    s = az.summary(idata, var_names=var_names, round_to=4)
    max_rhat = float(s['r_hat'].max())
    min_ess = float(s['ess_bulk'].min())
    div = int(idata.sample_stats['diverging'].values.sum()) \
        if 'diverging' in idata.sample_stats else -1
    return max_rhat, div, min_ess, s


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--step', required=True,
                   choices=('plain', 'plain_pf', 'indep', 'lapse'),
                   help="Ladder step (see module docstring).")
    p.add_argument('--data', required=True,
                   help='Path to magjudge_behavior_DNumRisk.csv')
    p.add_argument('--out', required=True, help='Output JSON path for results.')
    p.add_argument('--nc-out', default=None,
                   help='Optional: also save the full idata netCDF here.')
    p.add_argument('--n-subjects', type=int, default=8,
                   help='First N subjects; 0 = all.')
    p.add_argument('--init', default=None,
                   choices=('mapjitter', 'pathfinder', 'priorjitter'),
                   help='Override starting-point finder. Defaults: plain/indep '
                        '-> mapjitter, plain_pf -> pathfinder.')
    p.add_argument('--memory-model', default=None,
                   help="Override memory model. Defaults: plain/plain_pf/lapse "
                        "-> shared_perceptual_noise; indep -> independent.")
    p.add_argument('--beta-mu-mean', type=float, default=0.02,
                   help='Lapse Beta group-mean prior (regularize DOWN). ~0.02-0.03.')
    p.add_argument('--draws', type=int, default=1000)
    p.add_argument('--tune', type=int, default=2000)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--target-accept', type=float, default=0.99)
    p.add_argument('--backend', default='numpyro',
                   choices=('pymc', 'numpyro', 'blackjax'))
    p.add_argument('--chain-method', default='vectorized',
                   choices=('vectorized', 'parallel'))
    p.add_argument('--random-seed', type=int, default=None)
    p.add_argument('--group-sd-dist', default='halfcauchy',
                   choices=('halfcauchy', 'halfnormal'),
                   help="Group-SD prior family (halfnormal tames the funnel).")
    args = p.parse_args()

    # Resolve per-step defaults for init and memory model.
    init = args.init
    if init is None:
        init = 'pathfinder' if args.step == 'plain_pf' else 'mapjitter'
    memory_model = args.memory_model
    if memory_model is None:
        memory_model = ('independent' if args.step == 'indep'
                        else 'shared_perceptual_noise')

    df = load_data(args.data, args.n_subjects)
    print(f"\n=== step={args.step} | init={init} | memory_model={memory_model} "
          f"| ta={args.target_accept} | n_subj="
          f"{df.index.get_level_values('subject').nunique()} ===")

    model = build_model(df, args.step, memory_model, args.beta_mu_mean)
    model.group_sd_dist = args.group_sd_dist
    model.build_estimation_model(data=df, hierarchical=True)

    sample_kwargs = dict(
        draws=args.draws, tune=args.tune, chains=args.chains,
        target_accept=args.target_accept, backend=args.backend,
        find_init=init,
    )
    if args.random_seed is not None:
        sample_kwargs['random_seed'] = args.random_seed
    if args.backend in ('numpyro', 'blackjax'):
        sample_kwargs['chain_method'] = args.chain_method

    t0 = time.time()
    idata = model.sample(**sample_kwargs)
    elapsed = time.time() - t0

    # Report on the group-level (_mu) parameters — the convergence verdict.
    var_names = ['n1_evidence_sd_mu', 'n2_evidence_sd_mu', 'a_mu', 't0_mu']
    if args.step == 'lapse':
        var_names.append('p_outlier_mu')
    var_names = [v for v in var_names if v in idata.posterior]

    max_rhat, div, min_ess, summary = diagnostics(idata, var_names)
    print(f"\n--- RESULT step={args.step} ---")
    print(f"  elapsed         : {elapsed:.1f}s")
    print(f"  max r-hat (_mu) : {max_rhat:.4f}")
    print(f"  divergences     : {div} / {args.chains * args.draws}")
    print(f"  min bulk ESS    : {min_ess:.0f}")
    print(summary[['mean', 'sd', 'r_hat', 'ess_bulk']].to_string())

    result = {
        'step': args.step, 'init': init, 'memory_model': memory_model,
        'group_sd_dist': args.group_sd_dist,
        'target_accept': args.target_accept, 'backend': args.backend,
        'n_subjects': int(df.index.get_level_values('subject').nunique()),
        'n_trials': int(len(df)), 'beta_mu_mean': args.beta_mu_mean,
        'elapsed_sec': elapsed,
        'max_rhat_mu': max_rhat, 'divergences': div, 'min_ess_bulk': min_ess,
        'n_post_draws': int(args.chains * args.draws),
        'random_seed': args.random_seed,
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote results -> {args.out}")

    if args.nc_out:
        idata.to_netcdf(args.nc_out)
        print(f"Wrote idata    -> {args.nc_out}")

    # Crude verdict for at-a-glance log scanning.
    ok = (max_rhat <= 1.05) and (div <= 0.01 * args.chains * args.draws) \
        and (min_ess >= 400)
    print(f"\nVERDICT: {'CONVERGED' if ok else 'NOT converged'} "
          f"(r-hat<=1.05, div<=1%, ESS>=400)")
    return 0 if ok else 0  # always exit 0 so SLURM array continues


if __name__ == '__main__':
    sys.exit(main())
