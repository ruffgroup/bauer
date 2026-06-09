# Sampling robustness: lapses, funnels, init, and target_accept

A practical guide to getting bauer's hierarchical models to **converge by
default**, distilled from debugging the numloss risky-choice and dyscalculic
DDM fits. Read this when a fit shows divergences, r̂ > 1.01, low ESS, or a
"seed-lottery" where convergence depends on the random seed.

The headline: most bauer non-convergence we've seen traces to one of four
**model/prior misspecifications**, not the sampler — and each has a one-line
fix that is now built in.

---

## 1. Unmodeled lapses → rough likelihood → stuck chains

**Symptom.** A static choice model (or DDM) won't mix: r̂ ≈ 1.5–2, min ESS ≈ 6,
0–few divergences, 1–2 frozen chains; a *seed lottery* (some seeds "work").
Synthetic data simulated *from the model* converges fine; **real data doesn't**.

**Cause.** Real subjects lapse (guess / attention slips) on a fraction of
trials. A model with no lapse term can only explain EV-defying easy-trial
choices by inflating the encoding noise, which makes the likelihood rough /
multimodal — hence the stuck chains. The "synthetic converges, real doesn't"
asymmetry is the tell (synthetic has no lapses).

**Fix.** Use the lapse variant:
- static choice: `RiskLapseModel` / `RiskLapseRegressionModel`,
  `MagnitudeComparisonLapseModel`, `PsychophysicalLapseModel`, …
  (`p = p·(1−λ) + 0.5·λ`).
- DDM/RDM: the `*LapseModel` variants add an HSSM-style RT contaminant
  (`p_outlier`); see §2.

**Worked example (numloss risk).** `RiskRegressionModel` → r̂ 1.9, ESS 6.
Switch to `RiskLapseRegressionModel` → **r̂ 1.01, ESS ~1000**, same data, same
sampler. The lapse term doubles as a **data-quality diagnostic**: per-subject
lapse correlated −0.71 with EV-tracking accuracy, cleanly flagging ~10/46
disengaged subjects.

**Lapse is also a screening tool.** A small group-mean lapse can hide a
heavy-tailed per-subject distribution (a cluster of disengaged subjects at
λ→0.9). Inspect the per-subject lapse posterior; consider a pre-specified
performance screen (e.g. above-chance in all conditions).

---

## 2. The DDM/RDM RT contaminant (`p_outlier`)

The static choice-lapse (`p·(1−λ)+0.5·λ`) is **wrong** for a likelihood over
`(rt, choice)`. The DDM/RDM lapse is an HSSM-style mixture:

```
logp = log( (1−p_outlier)·exp(ll_WFPT) + p_outlier·exp(lapse_logp) + 1e-29 )
```

with the contaminant a **uniform density over RT** on `[0, lapse_upper]`
(default 20 s, HSSM convention) and 50/50 over choice, so
`lapse_logp = log(0.5) − log(lapse_upper)` (constant). Configure via
`lapse_upper` / `lapse_choice_5050`. Regularize `p_outlier` **downward** (small
`beta_mu_mean`, ~0.02–0.03) so the contaminant can't soak up real structure.

---

## 3. The two funnels (prior misspecification at the group level)

Hierarchical SDs / bounded rates with the wrong group prior **funnel** — a
weakly-identified group parameter wanders to extreme values, the per-subject
offsets blow up, and NUTS diverges. Two we hit and fixed:

### 3a. Lapse rate: logit-Normal → Beta "1/x"  (`lapse_group`)
A logit-Normal hierarchy on the lapse rate funnels when subjects' true lapse
≈ 0 (logit → −∞, group SD → ∞; observed group SD = 35, 368 divergences). The
**hierarchical Beta** group prior — density `∝ p^(α−1)`, a spike at 0 with a
heavy upper tail ("1/x") — keeps the rate in (0,1) and removes the funnel
(368 → 2 divergences). **Now the default** (`lapse_group='beta'`);
`lapse_group='logit_normal'` reproduces old fits.

### 3b. Group SD: HalfCauchy → HalfNormal  (`group_sd_dist`)
`group_sd ~ HalfCauchy(0.25)` has an infinite-variance heavy tail; a
weakly-identified group SD can run to huge values → funnel + divergences
(seen on DDM noise components and near-0 lapse). `group_sd_dist='halfnormal'`
(opt-in) tames the tail. Trade-off: mild over-shrinkage if true between-subject
heterogeneity is genuinely large — so it's opt-in, not yet default.

---

## 4. Initialization: mapjitter, and MAP-seeded Pathfinder

bauer's starting-point finder (`recommended_init='mapjitter'`, default for
DDM/RDM) seeds each chain from the MAP centre + prior-scaled jitter. This alone
turns a regression-DDM seed-lottery (~1/7 seeds) into reliable convergence.

For genuinely nasty geometries, **Pathfinder** (`find_init='pathfinder'`,
opt-in, needs `pymc_extras`) seeds chains from a multipath variational
approximation of the typical set instead of the MAP basin.

**Critical detail — seed Pathfinder from a plausible point.** On hard DDM
geometries, Pathfinder's own optimization can wander into the WFPT flat/−inf
zone and hand NUTS a starting point from which *every* transition diverges
(observed: independent-noise DDM → **4000/4000 divergences, within-chain SD 0**,
frozen). bauer now seeds Pathfinder's optimization from `pm.find_MAP` (falls
back to Pathfinder's default if MAP fails). With seeding, that same fit went to
**r̂ 1.03, 0 divergences**.

**Pathfinder is expensive** (a `find_MAP` + multipath VI before sampling). It is
**not** a free default — at small/easy scales mapjitter converges nearly as well
and faster. Treat Pathfinder as the **escalation** lever (§6).

---

## 5. target_accept is a band-aid for geometry — lower it once geometry is fixed

A high `target_accept` (0.99) forces a tiny step size so NUTS dodges
divergences on bad geometry — but tiny steps mean deep trees and slow sampling.
Once the geometry is fixed (lapse + good init), much of that is wasted.

**Measured** (plain DDM, 8 subj, MAP-seeded Pathfinder):

| target_accept | r̂ | divergences | min ESS | elapsed |
|---|---|---|---|---|
| 0.99 | 1.00 | 0 | 2664 | ~600 s |
| 0.90 | 1.00 | 10 | 2386 | 285 s |
| 0.80 | 1.00 | 12 | 1916 | 260 s |

→ ~2.3× faster at ta=0.9, r̂/ESS basically unchanged, but ~10 divergences
return (geometry is *mostly*, not perfectly, fixed). So:
- **ta ≈ 0.9** for fast iteration/debugging (eyeball the few divergences).
- **ta ≈ 0.95–0.99** for the final, divergence-free fit.

---

## 6. The tiered recipe (cheapest first; escalate only if it diverges)

1. **Right model first.** Add the lapse / `p_outlier` term (§1–2). This fixes
   more "non-convergence" than any sampler knob.
2. **Defaults**: `mapjitter` init, `lapse_group='beta'` (default),
   `target_accept=0.9`, numpyro + `chain_method='vectorized'`, `tune≈2000`.
   Check r̂ / ESS / divergences.
3. **If divergences concentrate at the `*_sd` level** → `group_sd_dist='halfnormal'`.
4. **If still a seed-lottery / stuck chains** → `find_init='pathfinder'`
   (MAP-seeded; expensive — escalation only).
5. **For the final clean fit**, raise `target_accept` to 0.95–0.99 to drive
   divergences to 0.
6. **Always** validate with parameter recovery (simulate from the fitted
   posterior, refit, check r̂ across an ensemble + recoverability) — convergence
   on one real dataset isn't enough.

DDM guardrails (see `fitting_ddm_models.md`): keep `v_scale` fixed
(v_scale↔evidence_sd degeneracy); filter `rt < 0.20 s`; debug the plain DDM
before the sv-DDM.

---

## Production recipes

**numloss risky choice (static) — SETTLED:**
`RiskLapseRegressionModel`, objective prior, `lapse_group='beta'` (default),
dense_mass + mapjitter, `tune=1500–2000`, ta=0.95 → r̂ 1.01, ESS ~1000.

**dyscalculic DDM — (pending full-scale validation):**
`memory_model='shared_perceptual_noise'`, `fit_v_scale=False`, MAP-seeded
Pathfinder, ta≈0.9 (final 0.95). 8-subject: r̂ 1.00, 0 div, ESS 2664.
_66-subject + lapse + HalfNormal results: TBD (fits in flight)._
