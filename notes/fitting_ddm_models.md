# How to fit DDM models in bauer on real data

**Read this before fitting any DDM/RDM.** It is the prescriptive recipe;
`ddm_convergence_lessons.md` is the longer retrospective on *why* each step
is here. If a fit won't converge, 90% of the time it's one of the four
pitfalls in §6 — check those before inventing a new theory.

Canonical runnable examples:
- `bauer/scripts/fit_for_lesson8.py` — magnitude comparison, hierarchical,
  with an ISI regressor (Garcia n=64).
- `examples/for_alina/fit_ddm.py` — risky choice, per-subject, gain/loss
  regression.

---

## 0. The one-paragraph recipe

Drop trials with `rt < 0.20 s`; make sure `choice` is boolean and `rt` is in
seconds. Build the DDM model with `fit_separate_evidence_sd=True` and
`fit_prior=True` (risk models: `prior_estimate='full'`). Sample with
**numpyro, vectorized, on a GPU**, `tune=2000, draws=1000, chains=4,
target_accept=0.99` for a real-sized dataset (~50+ subjects). Then **check
r̂ ≤ 1.01 and ESS ≥ 400 before believing any number**. That config is the
default for a reason — see below.

---

## 1. Prepare the data (do this first, every time)

```python
df = df[df['rt'] >= 0.20].copy()        # see §6.1 — the #1 cause of stuck chains
assert df['choice'].dtype == bool        # True = chose option 2
assert (df['rt'] > 0).all()              # seconds, not ms
```

- **RT filter is not optional.** bauer reuses HSSM's analytical WFPT, which
  *floors* the log-likelihood at `LOGP_LB = -66.1` (flat, zero-gradient)
  whenever the sampler proposes `t0 > min(rt)` for any subject. NUTS sees a
  zero gradient and the chain freezes. Dropping fast trials (0.20 s is a
  sane motor floor; tune per task) keeps the sampler out of that dead zone.
- **Hierarchical fits** need `subject` in the index or a column.
- Paradigm columns by family: magnitude `n1/n2`; risk `n1/n2/p1/p2`
  (+ `rt`, `choice`).

---

## 2. Pick the model class

| Task | Class |
|---|---|
| Magnitude comparison | `DDMMagnitudeComparisonModel` |
| …with a covariate on a parameter | `DDMMagnitudeComparisonRegressionModel` |
| Risky choice | `DDMRiskModel` |
| …with a covariate (group, condition, domain) | `DDMRiskRegressionModel` |
| Stimulus-dependent (B-spline) noise | the `*FlexibleNoise*` variants |

Standard front-end kwargs: `fit_separate_evidence_sd=True` (lets
`n1_evidence_sd ≠ n2_evidence_sd`, absorbing working-memory effects on the
first-presented option) and `fit_prior=True` / `prior_estimate='full'`
(estimate the Bayesian-observer prior).

**Regression**: pass `regressors={param: 'patsy_formula'}`, e.g.
`{'n1_evidence_sd': 'C(domain)', 'a': 'C(domain)'}`. Only regress a
parameter on a covariate you have a prior reason to expect it to move —
pre-register that, don't data-mine every parameter.

---

## 3. Sampler and settings — and why these defaults

```python
m.build_estimation_model(data=df, hierarchical=True)
idata = m.sample(
    draws=1000, tune=2000, chains=4,
    target_accept=0.99,
    backend='numpyro',          # JAX NUTS; chain_method='vectorized' is the bauer default
)
```

- **numpyro vectorized on GPU is the workhorse and the right default.** On
  an L4 a Garcia n=64 fit is ~45 min; on CPU it's many hours. It converges
  the basic *and* the regression DDM (the regression DDM only ever "needed"
  parallel/slice because of the softplus bug in §6.3 — that's fixed).
- **`tune=2000`, not 1000.** The default `tune=1000` gives r̂≈2.6 on n=64 —
  non-convergence, not a result. Warmup is where the
  `a ↔ t0 ↔ drift` identifiability ridge gets adapted.
- **`target_accept=0.99`** for hierarchical DDMs (smaller step, deeper
  trees, fewer divergences on the ridge). 0.95 is fine for the simpler
  static-choice models.
- **Do NOT use `backend='blackjax'`** — broken on GPU with HSSM's
  progressbar (JAX "IO effect not supported in vmap-of-cond").
- **`chain_method='parallel'`** = one process per chain, robust to a bad
  shared RNG seed, but 4× VRAM (won't fit n=64 on a 24 GB L4). Use only on
  CPU or small models.
- **`pm.Slice` (pymc, CPU)** = gradient-free, immune to the LOGP_LB
  pathology. Slow (hours). Keep it as a *robustness reference* when you
  suspect NUTS is misleading, not as a default.

---

## 4. Check convergence (before interpreting anything)

```python
import arviz as az
s = az.summary(idata, var_names=['n1_evidence_sd_mu', 'a_mu', 't0_mu'])  # + your params
print('max r_hat', s['r_hat'].max(), '| min ess', s['ess_bulk'].min())
print('divergences', int(idata.sample_stats['diverging'].sum()))
```

Ship only if **max r̂ ≤ 1.01**, **min ESS_bulk ≥ 400** (≈100/chain), and
divergences are a small fraction of post-warmup draws. A PPC that
mispredicts wildly (e.g. simulated RT 2× the data) is almost always a
*non-converged fit*, not a model-misfit story — check r̂ first.

---

## 5. PPC notes specific to DDMs

- `model.ppc(df, idata, n_posterior_samples=60)` returns `simulated_choice`
  (bool) and `simulated_rt`. Regression models work too (the 2D-parameter
  ppc path was fixed).
- Simulated RT has a **heavy right tail** (WFPT) that real data doesn't,
  because the experiment truncates slow trials. Summarize simulated RT with
  the **median or a matched-support mean**, not the raw mean, or the PPC
  will look biased even when the fit is fine.

---

## 6. When it won't converge — decision tree

Work top to bottom; stop at the first that applies.

1. **Did you filter `rt < 0.20 s`?** If not, do it (§1). This is the most
   common cause, full stop.
2. **Is r̂ bad on a *regression* model fit before 2026-05-23?** The
   `RegressionModel` softplus-prior bug silently zeroed Intercept priors
   (e.g. drove `t0`'s prior mean to 0.69 s, into the LOGP_LB zone). Fixed
   in `bauer/core.py` (commit `34f8777`). **Re-fit it** on current code.
3. **Bad seed catching vectorized chains?** Re-run with a different
   `random_seed`, or `chain_method='parallel'` (CPU / small model), or just
   bump `tune` to 2000–4000.
4. **Genuinely weak data (a parameter is unidentified)?** Symptom:
   `P(choice) ≈ 0.5` in some condition → flat drift signal → intrinsically
   broad WFPT posterior. No sampler fixes this. Options, in order:
   a. Drop the regression on parameters the covariate shouldn't move
      (often `a`, `t0`) — get to the minimal model that answers your
      question.
   b. Tighten the `t0` prior (subclass, set `sigma_intercept ≈ 0.3` on
      `t0`) so it can't wander toward the dead zone.
   c. For a pilot/small-N subject: accept the wide posterior, label it as
      pilot, and re-fit when more subjects arrive (the hierarchical fit
      usually behaves once the group constrains it).

---

## 7. Cluster submission (sciencecluster)

```bash
# one GPU fit via the generic runner (logs to ~/logs/<jobname>_<id>.txt)
sbatch --job-name=myfit --gres=gpu:L4:1 --partition=lowprio --time=04:00:00 \
    bauer/scripts/slurm_jobs/run_fit.sh bauer_cuda \
    bauer.scripts.fit_for_lesson8 --tune 2000 --target-accept 0.99
```

`bauer_cuda` is the GPU conda env. Always `git pull` on the cluster first;
verify `import bauer` works before burning a GPU slot (a broken import
fails in ~10 s but wastes the queue wait).
