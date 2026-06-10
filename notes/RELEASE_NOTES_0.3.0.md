# bauer 0.3.0 — release notes

`bauer.__version__ == '0.3.0'`.

This is the "make the accumulator models converge on real data" release. The
core change is a fixed-rate RT-outlier contaminant baked into the base DDM/RDM
models; combined with a lighter-tailed group-SD prior it turns a 66-subject
magnitude-comparison DDM from "never mixes" (r̂ = 2.42) into "converges cleanly"
(r̂ = 1.00). That improvement came at the cost of three breaking changes — read
those first, because one of them silently changes the numbers of every existing
hierarchical fit.

---

## ⚠️ BREAKING CHANGES

Three things change behaviour. The first is the dangerous one: it alters results
without erroring or warning.

### 1. `group_sd_dist` default changed: `halfcauchy` → `halfnormal`

**What changed.** The group-level SD prior on every hierarchical regression /
random-effect node now defaults to `HalfNormal(sigma=scale)` instead of
`HalfCauchy(beta=scale)`.

**Why it matters.** This is the prior on the between-subject spread of *every*
hierarchical parameter, so it touches *every* hierarchical fit in the library.
HalfCauchy has an infinite-variance heavy tail: when a group SD is weakly
identified (DDM noise components, near-zero lapse rates, …) it can wander to huge
values, producing a funnel and divergences. HalfNormal's light tail tames the SD
and removes that pathology, at the cost of mild over-shrinkage when true
between-subject heterogeneity is large.

**The trap.** Re-running a pre-0.3.0 analysis under 0.3.0 silently gives
different posteriors — no error, no warning. They are usually better-mixing and
slightly more shrunken, but they are *not* your old numbers.

**Migration — to reproduce pre-0.3.0 results exactly:**

```python
model = DDMMagnitudeComparisonRegressionModel(df, fixed_regressors=regressors)
model.group_sd_dist = 'halfcauchy'      # restore the pre-0.3.0 heavy-tail prior
```

**Migration — to adopt the new default (recommended):**

```python
model = DDMMagnitudeComparisonRegressionModel(df, fixed_regressors=regressors)
# model.group_sd_dist == 'halfnormal'   # already the default in 0.3.0
```

Only `'halfnormal'` and `'halfcauchy'` are valid; any other value raises
`ValueError("group_sd_dist must be 'halfcauchy' or 'halfnormal', got ...")`.

### 2. DDM / RDM `*Lapse*` model classes removed

**What changed.** The RT-aware contaminant moved into the *base* DDM/RDM models,
so the dedicated lapse subclasses are gone. Removed:

- `DDMMagnitudeComparisonLapseModel`
- `DDMMagnitudeComparisonLapseRegressionModel`
- `DDMRiskLapseModel`, `DDMRiskLapseRegressionModel`
- every `RaceDiffusion*LapseModel` / `RaceDiffusion*LapseRegressionModel`

These names no longer exist — importing one raises `ImportError` /
`AttributeError`.

**Migration.** Drop `Lapse` from the class name and control the contaminant with
the `p_outlier` attribute (default `0.05`, fixed):

```python
# before 0.3.0:
model = DDMMagnitudeComparisonLapseModel(df)

# 0.3.0:
model = DDMMagnitudeComparisonModel(df)     # contaminant built into the base class
model.p_outlier = 0.0          # pure WFPT, no contaminant (old non-lapse class)
model.p_outlier = 0.05         # fixed 5% contaminant rate (the new default)
model.p_outlier = 'hierarchical'   # per-subject estimated rate (weakly id'd; avoid)
```

**Not affected.** The **static-choice** `*Lapse*` models are UNCHANGED, because
there the lapse is an optional *choice*-only mixture, not an RT contaminant:
`RiskLapseModel`, `RiskLapseRegressionModel`, `MagnitudeComparisonLapseModel`,
`MagnitudeComparisonLapseRegressionModel`, `PsychophysicalLapseModel`,
`PsychophysicalLapseRegressionModel` all remain their own classes.

### 3. `regressors=` keyword deprecated

**What changed.** Regression models now split the design into a fixed
(population-mean) part and a random (per-subject) part via `fixed_regressors=`
and `random_regressors=`. The legacy `regressors=` keyword still works
**bit-for-bit** (it maps to `fixed_regressors = random_regressors = regressors`)
but now emits a `DeprecationWarning`.

**Why it matters.** `regressors=` put a per-subject random effect on *every*
term — a spurious random slope even on a between-subjects contrast like `group`,
which is non-identified for off-group subjects and makes the between-subject
variance heteroscedastic.

**Migration.**

```python
# before 0.3.0 (now emits DeprecationWarning):
model = DDMMagnitudeComparisonRegressionModel(
    df, regressors={'a': 'group', 'n1_evidence_sd': 'group'})

# 0.3.0 — population means only (correct for a between-subjects 'group'):
model = DDMMagnitudeComparisonRegressionModel(
    df,
    fixed_regressors={'a': 'group', 'n1_evidence_sd': 'group'})
    # random_regressors omitted -> per-subject random INTERCEPT only

# 0.3.0 — random slope on a SUBSET of columns:
model = DDMMagnitudeComparisonRegressionModel(
    df,
    fixed_regressors={'a': 'C(session)'},
    random_regressors={'a': 'C(session)'})   # this term ALSO varies by subject
```

Rules: `random_regressors` defaults to intercept-only; its terms must be a
subset of the fixed design's columns (else `ValueError`); passing both
`regressors` and the new kwargs raises `ValueError`; putting a random effect on a
within-subject-constant regressor emits a `UserWarning`.

The exact deprecation text emitted:

> `regressors` is deprecated: it puts a per-subject random effect on EVERY term
> (a random slope even on between-subjects contrasts like group). Use
> `fixed_regressors` (population means) + `random_regressors` (per-subject
> effects; default intercept-only). …

---

## New features

### 1. Fixed-rate RT-outlier contaminant in the base DDM/RDM models

The base sequential-sampling models now carry an HSSM-style outlier mixture. On a
fraction `p_outlier` of trials the `(rt, choice)` is treated as a draw from a
flat density over RT rather than the diffusion process:

```
logp = log( (1 - p_outlier)·exp(WFPT) + p_outlier·exp(lapse) + 1e-29 )
lapse = Uniform(0, lapse_upper = 20 s) × 0.5 choice
```

The default `model.p_outlier = 0.05` is a **fixed** model attribute, *not* a
sampled parameter — so it cannot funnel or diverge. `0.0` recovers the pure
WFPT; `'hierarchical'` estimates a per-subject rate under a Beta "1/x" group
prior (weakly identified for the DDM — avoid). `simulate` / `ppc` generate the
contaminant by default (`simulate_contaminant=True`, matching HSSM's posterior
predictive).

```python
model = DDMMagnitudeComparisonModel(df)   # p_outlier = 0.05 by default
```

*Why it exists:* an unmodelled slow-RT tail is the single cause behind three
symptoms that look unrelated — RT-tail PPC misfit, exploding Pareto-k, and
non-convergence. The fixed contaminant fixes all three at once.

### 2. Configurable `group_sd_dist`

The group-SD prior family is now a per-instance / per-subclass attribute,
`'halfnormal'` (default) or `'halfcauchy'`.

```python
model.group_sd_dist = 'halfnormal'   # light tail, no funnel (default)
model.group_sd_dist = 'halfcauchy'   # heavy tail (pre-0.3.0 behaviour)
```

*Why it exists:* it is the single knob that decides whether a weakly-identified
group SD funnels. Exposing it lets you trade the HalfNormal default's mild
over-shrinkage for HalfCauchy's heavy tail when between-subject heterogeneity is
genuinely large.

### 3. `fixed_regressors` / `random_regressors` API

Regression models split the design into population-mean (`fixed_regressors`) and
per-subject random (`random_regressors`) parts. Random terms default to
intercept-only and must be a subset of the fixed design.

```python
model = DDMRiskRegressionModel(
    df,
    fixed_regressors={'a': 'C(domain)', 'n1_evidence_sd': 'C(domain)'},
    random_regressors={'a': '1'})        # random intercept only
```

*Why it exists:* the old `regressors=` forced a random slope on every column —
wrong for between-subjects contrasts. You can now put a random effect on exactly
the columns where it belongs.

### 4. `memory_as_sv` DDM variant

Routes the frozen memory noise into Ratcliff across-trial drift variability `sv`
(perceptual noise stays within-trial), so memory vs perceptual noise become
separately identifiable. Requires `memory_model='shared_perceptual_noise'`;
composes with the contaminant (the lapse mixin honours it) and emits an `sv`
node.

```python
model = DDMMagnitudeComparisonModel(
    df, memory_as_sv=True, memory_model='shared_perceptual_noise')
```

*Why it exists:* under `shared_perceptual_noise` the two noise sources enter the
drift SNR symmetrically and are collinear — only their sum is constrained.
Giving the memory error a distinct temporal role (frozen → across-trial `sv`,
producing slow errors) breaks the degeneracy. Full derivation in
`notes/memory_as_sv.md`.

### 5. Race `fit_w_d` / `fit_w_s` toggles

`fit_w_d=False` fixes the discriminative gain `w_d ≡ 1` (the race analogue of
pinning the DDM's `v_scale = 1`); `fit_w_s=False` ablates the overall-magnitude
(sum) drift term, `w_s ≡ 0`.

```python
model = RaceDiffusionMagnitudeComparisonModel(df, fit_w_d=False, fit_w_s=True)
```

*Why it exists:* `w_d` and the encoding noise `σ` are ~collinear (r ≈ 0.8).
Fixing `w_d` removes that degeneracy, lets the encoding noise / boundary set the
scale, and makes `σ` directly comparable across DDM and RDM fits.

### 6. `consistent_choice_noise` for static choice models

Normalizes the static (cumulative-normal) choice likelihood by the SD of the
noisy posterior mean (`w·σ_e`, via `posterior_mean_sd`) rather than the raw
evidence SD. Default `False` for back-compat.

```python
model = MagnitudeComparisonModel(df)
model.consistent_choice_noise = True
```

*Why it exists:* it makes the static choice rule use the same decision-variable
SD that the accumulators (DDM/RDM) already use internally — a KLW/DDM-consistent
normalization. Only active when `flat_observer_prior` is False (a flat prior has
no shrinkage gain `w`).

---

## Convergence / robustness

The fixed `p_outlier` contaminant is the change that cracks a 66-subject
magnitude-comparison DDM on real dyscalculia data. The three pathologies you'd
otherwise chase separately — RT-tail PPC misfit, exploding Pareto-k, and
non-convergence — are a single cause: unmodelled slow-tail / lapse RTs. The
fixed contaminant fixes all three.

| Config | r̂ | min ESS | Note |
|---|---|---|---|
| Plain DDM, no contaminant, HalfCauchy SD | 2.42 | 5 | Funnel, never mixes |
| Plain DDM, no contaminant, HalfNormal SD | 1.53 | 33 | Better, still bad |
| **+ fixed p_outlier=0.05, mapjitter** | **1.00** | **3211–4264** | Converges |
| RDM + contaminant, HalfCauchy SD | 1.12–1.20 | ~20 | RDM funnel returns |
| RDM + contaminant, **HalfNormal SD** | ~1.0 | hundreds | Needs HalfNormal |

The recipe that works (every item matters):

- Backend `numpyro` + `chain_method='vectorized'` on a GPU (e.g. an L4) — NOT
  CPU pymc NUTS.
- Pass `draws` / `tune` by **keyword** (a classic bug swaps them positionally).
- `find_init='mapjitter'` (the DDM default) — dispersed, data-informed starts.
  `find_init='pathfinder'` exists but is expensive and unnecessary once the
  contaminant is in.
- Drop `rt < 0.20 s`: this gives the WFPT gradient (avoids the t0
  likelihood-floor trap) and killed ~80 RDM divergences.
- `fit_v_scale=False` always — freeing `v_scale` reintroduces a perfect
  `v_scale ↔ evidence_sd` degeneracy.
- `target_accept=0.99` to converge; relax toward 0.9 for ~2.3× speed once
  stable.

The full DDM-fitting report — with the divergence post-mortem, PPC diagnostics,
and the complete sampler call — is in
`notes/fitting_ddm_without_divergences.md`.
