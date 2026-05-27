# Fitting DDM models in bauer

A short, prescriptive recipe for fitting drift-diffusion (and race) models.
Runnable examples: `bauer/scripts/fit_for_lesson8.py` (magnitude comparison
with a regressor) and `examples/for_alina/fit_ddm.py` (risky choice, gain/loss
regression).

## Recipe

```python
df = df[df['rt'] >= 0.20].copy()          # 1. drop fast RTs (see notes)
m = DDMRiskRegressionModel(               # 2. pick the class (table below)
        paradigm=df, fit_separate_evidence_sd=True, prior_estimate='full',
        regressors={'n1_evidence_sd': 'C(domain)'})   # only params you expect to move
m.build_estimation_model(data=df, hierarchical=True)  # 3. build
idata = m.sample(draws=1000, tune=2000, chains=4,     # 4. sample
                 target_accept=0.99, backend='numpyro')
# 5. CHECK before interpreting: max r̂ ≤ 1.01, min ESS ≥ 400, few divergences
```

`choice` is boolean (`True` = option 2); `rt` is in seconds; hierarchical fits
need `subject` in the index or as a column. Paradigm columns: magnitude
`n1/n2`, risk `n1/n2/p1/p2`.

| Task | Class |
|---|---|
| Magnitude comparison | `DDMMagnitudeComparisonModel` (`…RegressionModel` to add a covariate) |
| Risky choice | `DDMRiskModel` (`…RegressionModel` to add a covariate) |
| Stimulus-dependent (B-spline) noise | the `*FlexibleNoise*` variants |

Use `fit_separate_evidence_sd=True` (lets `n1`/`n2` encoding noise differ) and
`fit_prior=True` / `prior_estimate='full'` (estimate the Bayesian-observer
prior). For regression, only regress a parameter on a covariate you have a
prior reason to expect it to move — don't data-mine every parameter.

## What matters for convergence

1. **Filter fast RTs (`rt < 0.20 s`).** The WFPT likelihood has a flat,
   zero-gradient region when the non-decision time `t0` exceeds the fastest
   RT, and the sampler gets stuck there. Dropping implausibly fast trials
   keeps it out. Tune the cutoff to your task's motor floor.
2. **Good starting points (on by default).** bauer initialises each chain at a
   data-informed plausible point and disperses the chains around it (see next
   section). This is what makes DDM/regression posteriors converge reliably —
   leave it on.
3. **`tune=2000`, `target_accept=0.99`** for hierarchical DDMs. `numpyro`
   `chain_method='vectorized'` on a GPU is the fast default (~45 min for
   n≈64 on an L4). Then **always check r̂ and ESS** before trusting a fit.

## Starting points (initial values)

A DDM/regression posterior is a long, curved ridge, so *where the chains
start* strongly affects whether they converge. bauer handles this for you:
`BaseModel.get_initial_points` (enabled by default on DDM/Race via
`recommended_init='mapjitter'`) starts each chain at a **data-informed
plausible centre** — the posterior mode from `find_MAP`, falling back to the
prior-central point — and **disperses the chains around it by a fraction of
each parameter's prior SD**. Chains sit around the typical set rather than all
at the mode, so r̂ stays meaningful.

It works for **every parameter automatically** — core DDM (`a`, `t0`),
front-end (`prior_mu/sd`, `evidence_sd`), B-spline noise coefficients, lapse —
with no per-parameter tuning. Pass `find_init=False` to turn it off, or your
own `initvals` to override.

For flexible-noise models, note the B-spline **basis is fixed at model
construction from the paradigm** (not the fit data), so build the model with a
paradigm that spans your stimulus range; the spline coefficients are defined
relative to that basis.

## If a fit won't converge

1. Confirm you filtered `rt < 0.20 s`.
2. **One stuck chain on a big hierarchical fit?** Even with the finder, a single
   chain can occasionally wander off and stay there (chance), inflating r̂ across
   many parameters at once. Diagnose with an outlier-chain tally (if ~one chain
   accounts for most of the high-r̂ params, that's it); the other chains are
   fine. Just re-run with a different `random_seed` (and/or a longer `tune`).
   Judge convergence on the **group-level** (`*_mu`) r̂; a handful of
   weakly-identified subject params with mild r̂ (≤~1.05) is normal — don't
   chase it.
3. **Weak data?** If `P(choice) ≈ 0.5` in some condition, the drift carries
   little information and the posterior is genuinely broad — no sampler setting
   fixes that. Drop the regression on parameters the covariate shouldn't move
   (often `a`, `t0`), and/or fit a pilot subject as-is and re-fit
   hierarchically once more subjects are available.

`backend='numpyro'` is the default; avoid `blackjax` (incompatible with HSSM's
progress bar on GPU). `pm.Slice` (CPU, gradient-free) is a slow robustness
check, not a default.

## Cluster (sciencecluster)

```bash
sbatch --job-name=myfit --gres=gpu:L4:1 --partition=lowprio --time=04:00:00 \
    bauer/scripts/slurm_jobs/run_fit.sh bauer_cuda \
    bauer.scripts.fit_for_lesson8 --tune 2000 --target-accept 0.99
```

`git pull` on the cluster first; the GPU env is `bauer_cuda`.

---

*Background:* the empirical convergence experiment behind these defaults is in
`experiments/ddm_sampler_experiments.md`; the longer history is in
`ddm_convergence_lessons.md`.
