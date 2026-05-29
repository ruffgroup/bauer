# Hierarchical DDM convergence — what we learned

Notes from getting bauer's `DDMMagnitudeComparisonModel` (and its
regression variant) to converge cleanly on Barreto-Garcia 2022
(n=64 subjects, ~210 trials/subject after cleaning). Distilled in
May 2026 after several days of trial-and-error.

## TL;DR — root causes, in retrospect

After all the iteration, **two actual root causes** explain everything we
saw:

1. **RT-filter pathology (§1).** `t0 > min(rt)` hits HSSM's flat
   `LOGP_LB = -66.1` floor with zero gradient. Fixed by dropping fast
   trials. Real, general, well-documented.
2. **Bug in `RegressionModel.build_hierarchical_nodes` (§3c, fixed
   2026-05-23).** The softplus branch was missing from the
   transform-dispatch, silently zeroing out every regression
   parameter's `mu_intercept`. For DDM `t0` specifically this set
   the prior mean to `softplus(0) = 0.69 s` — larger than most RTs,
   guaranteed to land in the LOGP_LB dead zone above. **This single
   bug accounts for every regression-DDM convergence failure we saw**
   (intercept-only or with a real regressor), plus probably
   contaminates every previously-fit regression model in this codebase
   (risk_experiment, TMS analyses).

The HSSM-style prior tightening on `a`/`t0` (§2) was a real but
secondary improvement to the **basic** DDM. The "tighten the front-end
slope SD" hand-wave (now removed from the notes) was correct-by-accident
— it slightly helped because of how it interacted with bug §3c, but
isn't independently load-bearing.

## 1. Filter fast RTs before anything else

bauer reuses HSSM's analytical `logp_ddm`. When the sampler proposes
$t_0 > \min(\text{rt})$ for any subject, HSSM **floors the per-trial
log-likelihood at `LOGP_LB = -66.1`** instead of returning $-\infty$.
Sounds defensive — but **the floor is flat**: empirically verified that
$\frac{\partial \log p}{\partial t_0} = 0$ exactly in the invalid region.
NUTS sees a gradient of zero and the chain stops moving.

This is the most common cause of "DDM chains stuck for no reason."

**Recipe:** drop trials with `rt < 0.20 s` (or your task's plausible
motor floor) before fitting. Garcia loses ~2 % of trials; bauer now
warns to stderr if you forget.

Reference: HSSM `likelihoods/analytical.py:logp_ddm`, lines 289-302.

## 2. HSSM-style priors on `a` and `t0`

For hierarchical fits, bauer's old defaults (`sigma_intercept=0.5`,
`cauchy_sigma_intercept=0.25`) gave $\hat r \approx 1.6$ on n=64 even
after the RT filter — chains landed in different basins of the
$a \leftrightarrow t_0 \leftrightarrow \text{drift}$ identifiability
ridge. Tightening to **wide group mean + tight group SD** following
`hssm.prior.HDDM_MU` / `HDDM_SIGMA` fixed it:

```python
# bauer/models/ddm.py — DDMMixin.get_free_parameters
pars['a']  = {'mu_intercept': inverse_softplus_np(1.2),
              'sigma_intercept': 1.0,
              'cauchy_sigma_intercept': 0.1,
              'transform': 'softplus', 'min_value': 0.3}
pars['t0'] = {'mu_intercept': inverse_softplus_np(0.2),
              'sigma_intercept': 1.0,
              'cauchy_sigma_intercept': 0.2,
              'transform': 'softplus'}
```

Result: $\hat r$ 1.6 → 1.0, ESS 7 → 1400+, 0 divergences.

Reference: HDDM convention from Wiecki, Sofer & Frank (2013).

## 3. Don't tighten the *front-end* intercept priors

`cauchy_sigma_intercept=0.1` works for `a` and `t0` because subjects
genuinely have similar caution / non-decision time. Applying the same
tightening to `n*_evidence_sd`, `prior_mu`, `prior_sd` **broke** the
basic DDM ($\hat r \to 2.9$): forcing all subjects to have nearly
identical encoding noise creates pressure on the likelihood that
fragments the chains.

For regression-model variants (`DDMMagnitudeComparisonRegressionModel`
etc.), **only the per-subject *slope* SD** is worth tightening — that
keeps subjects' regression slopes close to the group slope without
constraining their absolute noise levels:

```python
# bauer/models/magnitude.py — MagnitudeComparisonModel.get_free_parameters
TIGHT_SLOPE = {'cauchy_sigma_regressors': 0.1}  # NOT cauchy_sigma_intercept
free_parameters['n1_evidence_sd'] = {'mu_intercept': -1., 'transform': 'softplus',
                                      **TIGHT_SLOPE}
```

This is bauer-specific judgment, not a published convention.

## 3c. THE BUG — missing softplus branch in `RegressionModel`

(Fixed 2026-05-23 in `bauer/core.py:736–752` and `:768–786`.)

`RegressionModel.build_hierarchical_nodes` and `build_prior` had a
transform-conditional dispatch that handled `'identity'` and
`'logistic'` but **silently fell through for `'softplus'`** with a
stale `# Possibly use inverse of softplus` comment. Concretely:

```python
# Before (broken):
if transform == 'identity':
    mu[0] = mu_intercept
    sigma[0] = sigma_intercept
elif transform == 'logistic':
    mu[0] = mu_intercept
    sigma[0] = sigma_intercept
# Possibly use inverse of softplus

# After (fixed):
# Regression operates on the *untransformed* scale (softplus / logistic /
# identity applied later in get_trialwise_variable), so the Intercept
# prior is Normal(mu_intercept, sigma_intercept) for ALL transforms.
mu[0] = mu_intercept
sigma[0] = sigma_intercept
```

For every **softplus**-transformed parameter (`n*_evidence_sd`, `a`,
`t0`, `v_scale`, flexible-noise spline coefs, …) — which is essentially
every important parameter in this codebase — the regression model was
silently using `Normal(0, sigma_regressors)` on the untransformed scale
instead of the model's declared `mu_intercept` / `sigma_intercept`.

Empirical impact on the DDM:

| param | basic-model prior | regression-model prior (before fix) | consequence |
|---|---|---|---|
| `t0` | softplus(N(-1.508, 1.0)) → ~0.20 s | softplus(N(0, 1.0)) → ~**0.69 s** | larger than most RTs → LOGP_LB dead zone (see §1) |
| `a`  | softplus(N(0.84, 1.0)) + 0.3 → ~1.5 | softplus(N(0, 1.0)) + 0.3 → ~0.99 | prior 3× wider, drift–boundary ridge un-anchored |
| `n1_evidence_sd` | softplus(N(-1, 0.5)) → ~0.31 | softplus(N(0, 1.0)) → ~0.69 | drift formula numerator/denominator collapse |

This explains why the regression-DDM convergence didn't respond to:
- longer warmup (the prior was just *wrong*, not just under-sampled)
- HSSM tightening of `a`/`t0` (only affects the basic-model code path)
- changing `chain_method` (every chain hit the same wrong prior)

After the fix, intercept-only regression should produce ≈ identical
posteriors to the basic model. Verified by graph inspection:
`n1_evidence_sd_mu` matches at mu=−1.0 between basic and reg now.

**Audit by claude-agent, 2026-05-23. Suggested defensive addition:**
smoke test in `tests/test_models.py` that asserts basic vs
intercept-only-regression model graphs are mathematically equivalent.

**Should re-fit any production regression-DDM (TMS, risk_experiment)
that was run before this commit** — the posterior *means* may be fine
(data dominates with hundreds of trials/subject), but credible intervals
and convergence diagnostics on those fits are not trustworthy.

## 3a. Why HSSM uses slice sampling for `blackbox` likelihoods (and what that tells us)

HSSM's `hssm/hssm.py:746` switches to `pm.Slice` when the likelihood
is `blackbox` (non-differentiable, e.g. LANs in HDDM). For
**analytical** and **approx_differentiable** likelihoods — what bauer
uses — HSSM defaults to NUTS (pymc or numpyro).

The reason matters for our case: slice sampling is **gradient-free**,
so the `LOGP_LB = -66.1` flat-floor pathology that traps NUTS chains
on $t_0 > \min(\text{rt})$ simply doesn't bite slice. It would just
bisect through the valid region. HSSM trusts the RT filter / sensible
priors to keep NUTS in the valid region; slice is the safety net for
the case where you can't compute gradients at all.

**Practical implication:** if NUTS keeps getting stuck on a bauer DDM
fit and tightening priors doesn't help, **slice via pmpc is the
gradient-free reference**. Pass `step=pm.Slice()` through bauer's
sample kwargs (currently CPU-only, several hours on n=64). Slow but
immune to gradient pathologies.

## 3b. Empirical sampler comparison (May 2026, Garcia n=64)

Same model, same data, varied sampler config:

| config | basic DDM r̂ / ESS_bulk | regression DDM r̂ / ESS |
|---|---|---|
| numpyro vec, tune=1000 (default) | 2.64 / 5 | 3.93 / 4 |
| numpyro vec, tune=2000, target=0.99 | **1.00 / 463** | 3.99 / 4 |
| numpyro vec, tune=2000, target=0.99 **(post-softplus-fix, 2026-05-26)** | 1.013 / 332 | **1.60 / 6** |
| numpyro parallel, tune=1000 | **1.00 / 1561** | (timeout — needs more time) |
| blackjax vec | crashed (jax bug, `IO effect not supported in vmap-of-cond`) |

> **RESOLVED — the lever is initialization, not `chain_method` (2026-05-26).**
> A controlled multi-seed experiment (see `experiments/ddm_sampler_experiments.md`)
> settled this. Convergence is **seed-dominated**: with the old generic-jitter
> init, the regression DDM converged in only **2/16 seeds (12%)** on vectorized —
> the *same config*, different seed, gave r̂ from 1.00 to 3.4. `parallel` merely
> decorrelates the chains' seeds so it loses the lottery less often; it is **not**
> the fix (and Alina pil02 failed on parallel). The fix is **bauer's
> starting-point finder** (`get_initial_points` / `recommended_init='mapjitter'`,
> now default-on for DDM/Race): MAP centre + per-parameter prior-scaled jitter,
> mirroring HSSM. On the exact config that failed 7/8 of the time it gives
> **8/8 (100%)** and ~3.7× faster. So: leave the finder on, `vectorized` is fine,
> check r̂. The rows above (incl. the `tune=4000` parallel "successes") were the
> lottery, not a real `chain_method` effect.

**Conclusions:**

- The basic DDM convergence issue is *seed luck*. Either longer warmup
  (`tune=2000`) or `chain_method='parallel'` (independent seeds per
  chain) fixes it cleanly. Parallel wins on ESS (~3× higher).
- The regression-DDM (`DDMMagnitudeComparisonRegressionModel`) on n=64
  with a single binary regressor is **harder than the basic DDM** —
  tune=2000 alone doesn't fix it, suggesting the Intercept + slope
  parameterisation has its own identifiability ridge beyond what HSSM
  priors on `a`/`t0` address. Either combine parallel + tune=2000, or
  go to slice (gradient-free).
- Blackjax on GPU + HSSM progress bar = JAX cond/vmap incompatibility.
  Would need `progressbar=False` to test; not retried.

## Decision guide — which sampler to use when

For a hierarchical bauer DDM (basic or regression) on a real-sized
dataset (~50+ subjects):

| Situation | Recommended | Why |
|---|---|---|
| Have GPU access | **numpyro vectorized on GPU L4** | ~25× faster than CPU; 45 min for n=64. Use this by default. |
| No GPU, multi-core CPU | **numpyro `chain_method='parallel'`** | Each chain in own process with independent seeds — robust to the "bad seed catches multiple vectorized chains" failure mode (§4). Slower than GPU but reliable. |
| Need a robustness check that NUTS isn't misleading | **pymc `Slice` step on CPU** | Gradient-free — immune to the LOGP_LB flat-floor pathology (§1) and identifiability ridges. Slow (hours), but the posterior it converges to is bulletproof. Use when NUTS posteriors look suspicious. |
| All you have is a CPU and one chain method to try | numpyro vectorized + `tune=2000, target_accept=0.99` | OK but slow (12-18 h for n=64); marginal cases of bad seed. |
| Diagnosing convergence problems | Fit twice with **different `chain_method`** and compare posteriors | If they agree, NUTS is fine. If they disagree, you have a parameterisation / prior bug — go hunt it (see §3c for one such bug). |

**Don't use:**
- `backend='blackjax'` (broken with HSSM's progress bar on GPU as of
  blackjax 1.x — `IO effect not supported in vmap-of-cond`. Would need
  `progressbar=False` to test, not retried).
- `backend='pymc'` with `init='auto'` on hierarchical DDM — gets stuck
  chains on the a↔t0↔drift ridge. Use `init='jitter+adapt_full'`
  (bauer auto-applies via `recommended_pymc_init`).

**Default for bauer DDM scripts:**
- `fit_for_lesson8.py`, `fit_garcia.py`, etc. use `backend='numpyro'`,
  default `chain_method='vectorized'`. For cluster GPU jobs that's
  optimal. Override to `--chain-method parallel` if you suspect chain
  coupling is biting you.

## 4. Sampler choice and chain coupling

`backend='numpyro'` with `chain_method='vectorized'` (bauer's default
for JAX backends) runs all chains in a single JAX `vmap` and shares
the RNG seed across them. **Unlucky seeds catch multiple chains at
once** — we saw a run where 2 of 4 chains were frozen in different
basins while the other 2 mixed cleanly.

If that happens:

- `chain_method='parallel'` runs each chain in a separate process with
  independent RNG → more robust to bad seeds, but uses 4× VRAM
  (won't fit n=64 on an L4 24 GB).
- `backend='blackjax'` is an alternative JAX-NUTS implementation with
  different adaptation logic; sometimes handles multi-modal posteriors
  more robustly than numpyro.
- Re-running with an explicit different `random_seed` is the cheapest
  fix.

## 5. Wall-time budget

Measured on cluster (Aug-May 2026, UZH sciencecluster):

| config | n=8 | n=64 |
|---|---|---|
| numpyro vectorized, GPU L4 | (not measured) | **~45 min** ✓ |
| numpyro vectorized, CPU 16-core EPYC 9554 | OK in minutes | **~12-18 hr** (tree depth grows during warmup; tight in 24 h slot) |
| numpyro vectorized, CPU 16-core EPYC 9654 | OK in minutes | (untested at n=64) |
| pymc multiprocess, CPU | ~15 min (lesson 9) | **~1:14 hr** but chains stuck (r̂ ≈ 4) |

The first "fast" pymc CPU n=64 run was misleading — chains were
stuck so tree depth stayed small, making it artificially cheap.
**Always check r̂ before celebrating wall time.**

The CPU run lottery is real: same code, same SLURM args, ~3×
wall-time variance depending on which EPYC generation the scheduler
hands you. `--constraint=9654` if you need reproducible timings.

## 6. What we did NOT do (for a real publication, you should)

- **Prior predictive checks.** Sample from the prior, check implied
  $P(\text{choose 2})$ and RT distributions are sensible.
- **Sensitivity analysis.** Refit at 2-3 prior strengths
  (`cauchy_sigma_intercept` ∈ {0.1, 0.2, 0.5}) and confirm the
  contrast of interest doesn't move materially.
- **Reference comparison.** Fit HDDM (the actual HDDM library) on
  the same data and check the posteriors agree.
- **Posterior predictive on RT distribution (not just psychometric).**
  Quantile-quantile plot of empirical vs PPC RT distributions per
  difficulty bin.

These should all be done before claiming the priors are right for a
new dataset.

## 7. Practical fit-or-load script

`bauer/scripts/fit_for_lesson8.py` is the canonical cluster-side
fitter for the tutorial. It:

- Drops `rt < 0.20 s`
- Adds median-split `isi_cat` for the regression demo
- Fits probit + DDM + DDM-with-ISI-regressor
- Caches each to `~/.bauer_tutorial_cache/garcia_n{n}_rtmin{ms}_{name}.nc`
- Skips any that are already cached (idempotent)
- Has a tail-able heartbeat thread (numpyro tqdm is silent in non-TTY)

Submit on cluster with:
```bash
sbatch --job-name=lesson8 --partition=lowprio --gres=gpu:L4:1 \
       --cpus-per-task=4 --mem=16G --time=03:00:00 \
       bauer/scripts/slurm_jobs/run_fit.sh bauer_cuda \
       bauer.scripts.fit_for_lesson8
```

## 8. Open questions

- Does the basic DDM convergence regression in May 2026 (good seed →
  r̂=1.0, bad seed → r̂=2.6) point to a deeper identifiability
  problem that needs a stronger prior on `a`, or just bad luck?
  A prior-sensitivity check would resolve.
- Does blackjax help? Trying as of 2026-05-22.
- Should bauer expose a `--chain-method` CLI on `fit_for_lesson8.py`?
  Currently hard-coded to whatever `model.sample()` defaults to.
