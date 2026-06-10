# Changelog

All notable changes to **bauer** are documented here. This project loosely
follows [Keep a Changelog](https://keepachangelog.com/) and semantic versioning.

## 0.3.0

This release makes bauer's sequential-sampling (DDM / RDM) models converge on
real, messy data. The headline is a fixed-rate RT-outlier contaminant baked
into the base accumulator models, which — together with a tamer group-SD prior
— cracks a 66-subject magnitude-comparison DDM that previously never mixed
(r̂ 2.42 → 1.00). Getting there required three breaking changes; read them first.

### ⚠️ BREAKING CHANGES

#### 1. `group_sd_dist` default changed: `halfcauchy` → `halfnormal` (silently changes every hierarchical fit)

The group-level SD prior on hierarchical regression / random-effect nodes now
defaults to `HalfNormal` instead of `HalfCauchy`. **This changes the prior on
EVERY hierarchical fit in the library**, so re-running an old analysis under
0.3.0 will give different (usually better-mixing, slightly more shrunken)
posteriors *without any error or warning*. HalfCauchy's infinite-variance tail
let a poorly-identified group SD wander to huge values (funnel + divergences);
HalfNormal's light tail tames it.

**To reproduce pre-0.3.0 results, set the old prior per-instance:**

```python
# before 0.3.0 (implicit):
model = DDMMagnitudeComparisonRegressionModel(df, regressors)

# 0.3.0, to reproduce the OLD numbers exactly:
model = DDMMagnitudeComparisonRegressionModel(df, regressors)
model.group_sd_dist = 'halfcauchy'      # restore the pre-0.3.0 prior

# 0.3.0 default (new behaviour, recommended):
model.group_sd_dist = 'halfnormal'      # this is now the default
```

Valid values are `'halfnormal'` (default) and `'halfcauchy'`; anything else
raises `ValueError`.

#### 2. DDM / RDM `*Lapse*` model classes removed (lapse is now in the base classes)

The RT-aware contaminant now lives in the *base* DDM/RDM models, so the
separate lapse subclasses are gone. Removed:

`DDMMagnitudeComparisonLapseModel`, `DDMMagnitudeComparisonLapseRegressionModel`,
`DDMRiskLapseModel`, `DDMRiskLapseRegressionModel`, and every
`RaceDiffusion*LapseModel` / `RaceDiffusion*LapseRegressionModel`.

Migrate by dropping `Lapse` from the class name and controlling the contaminant
via the `p_outlier` attribute:

```python
# before 0.3.0:
model = DDMMagnitudeComparisonLapseModel(df)        # lapse was a separate class

# 0.3.0:
model = DDMMagnitudeComparisonModel(df)             # contaminant is built in
# model.p_outlier defaults to 0.05 (fixed); override if needed:
model.p_outlier = 0.0                               # pure WFPT, no contaminant
model.p_outlier = 0.05                              # fixed 5% rate (default)
model.p_outlier = 'hierarchical'                    # per-subject estimated rate
```

Note: the **static-choice** `*Lapse*` models — `RiskLapseModel`,
`RiskLapseRegressionModel`, `MagnitudeComparisonLapseModel`,
`MagnitudeComparisonLapseRegressionModel`, `PsychophysicalLapseModel`, … — are
**UNCHANGED**. There the lapse is an optional *choice*-only mixture and remains
its own class.

#### 3. `regressors=` keyword deprecated in favour of `fixed_regressors` / `random_regressors`

Regression models now split the design into a fixed (population-mean) part and
a random (per-subject) part. The old `regressors=` keyword still works
bit-for-bit but emits a `DeprecationWarning`: it put a per-subject random effect
on *every* term — a spurious random slope even on between-subjects contrasts like
`group`.

```python
# before 0.3.0 (now warns):
model = DDMMagnitudeComparisonRegressionModel(
    df, regressors={'a': 'group', 'n1_evidence_sd': 'group'})

# 0.3.0 — population means only (correct for a between-subjects 'group'):
model = DDMMagnitudeComparisonRegressionModel(
    df,
    fixed_regressors={'a': 'group', 'n1_evidence_sd': 'group'},
    # random_regressors omitted -> per-subject random INTERCEPT only
)

# 0.3.0 — put a random slope on a SUBSET of columns:
model = DDMMagnitudeComparisonRegressionModel(
    df,
    fixed_regressors={'a': 'C(session)'},
    random_regressors={'a': 'C(session)'},   # this term also varies by subject
)
```

`random_regressors` defaults to intercept-only and its terms must be a subset of
the fixed design's columns (otherwise `ValueError`). Passing both `regressors`
and the new kwargs raises `ValueError`. A `UserWarning` fires if you put a random
effect on a regressor that is constant within subject (the classic footgun).

### New features

- **Fixed-rate RT-outlier contaminant in the base DDM/RDM models.** An
  HSSM-style mixture: on a fraction `p_outlier` of trials the response is a flat
  contaminant over RT,
  `logp = log((1-p)·exp(WFPT) + p·exp(lapse) + 1e-29)`, with
  `lapse = Uniform(0, lapse_upper=20 s) × 0.5` choice. The default is a **fixed**
  `model.p_outlier = 0.05` (a model attribute, not a sampled parameter, so it
  cannot funnel). `model.p_outlier = 0.0` recovers the pure WFPT;
  `'hierarchical'` estimates a per-subject rate under a Beta "1/x" group prior
  (weakly identified for the DDM — avoid). `simulate` / `ppc` generate the
  contaminant by default (`simulate_contaminant=True`, matching HSSM). *Why:* an
  unmodelled slow-RT tail is the single cause behind RT-tail PPC misfit,
  exploding Pareto-k, and non-convergence; modelling it fixes all three.

- **`group_sd_dist` is now configurable** (`'halfnormal'` | `'halfcauchy'`) per
  instance / subclass, governing the group-SD prior on every hierarchical
  regression node. *Why:* it is the single knob that decides whether a
  weakly-identified group SD funnels — exposing it lets you trade the new
  HalfNormal default's mild over-shrinkage for HalfCauchy's heavy tail when
  between-subject heterogeneity really is large.

  ```python
  model.group_sd_dist = 'halfnormal'   # default: light tail, no funnel
  ```

- **`fixed_regressors` / `random_regressors` API** on regression models.
  `fixed_regressors={param: formula}` is a population-mean (pure fixed-effect)
  design; `random_regressors={param: formula}` selects which terms *also* carry
  a per-subject random effect (default: intercept-only). *Why:* the old
  `regressors=` put a random slope on every column, which is wrong for
  between-subjects contrasts — you can now put a random effect on a subset of
  columns.

  ```python
  model = DDMRiskRegressionModel(
      df,
      fixed_regressors={'a': 'C(domain)'},
      random_regressors={'a': '1'})       # random intercept only
  ```

- **`memory_as_sv` DDM variant.** Routes the frozen memory noise into Ratcliff
  across-trial drift variability `sv` (perceptual noise stays within-trial),
  making memory vs perceptual noise identifiable. Requires
  `memory_model='shared_perceptual_noise'`; composes with the contaminant and
  emits an `sv` node. *Why:* under `shared_perceptual_noise` the two noise
  sources are otherwise collinear in the drift SNR (only their sum is
  constrained). See `notes/memory_as_sv.md`.

  ```python
  model = DDMMagnitudeComparisonModel(
      df, memory_as_sv=True, memory_model='shared_perceptual_noise')
  ```

- **Race `fit_w_d` / `fit_w_s` toggles.** `fit_w_d=False` fixes the
  discriminative gain `w_d ≡ 1` (the race analogue of pinning the DDM's
  `v_scale = 1`); `fit_w_s=False` ablates the overall-magnitude (sum) drift term
  (`w_s ≡ 0`). *Why:* `w_d` and the encoding noise `σ` are ~collinear (r ≈ 0.8),
  so fixing `w_d` removes a degeneracy and makes `σ` comparable across DDM/RDM.

  ```python
  model = RaceDiffusionPowerLawNoiseComparisonModel(
      df, fit_w_d=False, fit_w_s=True)
  ```

- **`consistent_choice_noise` for static choice models.** Normalizes the choice
  likelihood by the SD of the noisy posterior mean (`w·σ_e`, KLW/DDM-consistent)
  rather than the raw evidence SD. Default `False` (back-compat). *Why:* it makes
  the static cumulative-normal choice rule consistent with what the accumulators
  already do internally (`posterior_mean_sd`).

  ```python
  model = MagnitudeComparisonModel(df)
  model.consistent_choice_noise = True
  ```

### Convergence / robustness

The fixed `p_outlier` contaminant is what cracks the 66-subject
magnitude-comparison DDM on real dyscalculia data. The three pathologies —
RT-tail PPC misfit, exploding Pareto-k, and non-convergence — are one cause
(unmodelled slow-tail / lapse RTs), and the fixed contaminant fixes all three:

| Config | r̂ | min ESS | Note |
|---|---|---|---|
| Plain DDM, no contaminant, HalfCauchy SD | 2.42 | 5 | Funnel, never mixes |
| Plain DDM, no contaminant, HalfNormal SD | 1.53 | 33 | Better, still bad |
| **+ fixed p_outlier=0.05, mapjitter** | **1.00** | **3211–4264** | Converges |
| RDM + contaminant, HalfCauchy SD | 1.12–1.20 | ~20 | RDM funnel returns |
| RDM + contaminant, **HalfNormal SD** | ~1.0 | hundreds | Needs HalfNormal |

The recipe that works: backend `numpyro` + `chain_method='vectorized'` on a GPU
(NOT CPU pymc NUTS); pass `draws` / `tune` by **keyword** (a classic bug swaps
them); `find_init='mapjitter'` (the DDM default); drop `rt < 0.20 s` (gives the
WFPT gradient — killed ~80 RDM divergences); `fit_v_scale=False` always (freeing
`v_scale` reintroduces a perfect `v_scale ↔ evidence_sd` degeneracy);
`target_accept=0.99` to converge, relax toward 0.9 for ~2.3× speed once stable.

See `notes/fitting_ddm_without_divergences.md` for the full DDM-fitting report.
