# Changelog

All notable changes to **bauer** are documented here. This project loosely
follows [Keep a Changelog](https://keepachangelog.com/) and semantic versioning.

## 0.5.0 (2026-10-18)

The headline of this release is a much-extended `bauer.efficient_coding`
module: the efficient-coding models of Bedi et al. (2026) for orientation ->
value judgements (BDM bids), now hierarchical, JAX/numpyro-compilable, and
carrying the model extensions used for the paper fits. It also fixes a serious
bug that made every *hierarchical* fit of the grid-based estimation models
invalid (see "Fixed").

> **Version note.** A `v0.4.0` tag exists on the side branch
> `feature/rdm-fix-wd-default` (RDM `fit_w_d=False` default + cell-means
> random-effect ergonomics). It was never released and is **not** part of
> `main` or of this release; 0.5.0 skips the number to avoid colliding with
> that tag. In 0.5.0 the race models still default to `fit_w_d=True`.

### Fixed

- **Hierarchical estimation fits used subject 0's parameters for everyone.**
  `EstimationBaseModel` grid models (all efficient-coding models and the three
  `bauer.numerosity` models) were handed trialwise-expanded parameters and then
  gathered `table[subject_ix]` from the first `n_subjects` *trials*, all of
  which belong to subject 0. The data contributed nothing for any other
  subject and the group SD had zero gradient. Single-subject fits were
  unaffected. New `EstimationBaseModel.subjectwise(name)` returns the
  subject-level RV. **Re-run any hierarchical fit of these models made with
  bauer <= 0.3.0.**
- Efficient-coding group-SD priors are now scaled to each parameter
  (1.5x its own prior SD) instead of bauer's generic 0.25, which under
  HalfNormal forced complete pooling of `kappa_r` (all subjects identical).
- `predict`/`simulate` of efficient-coding models no longer fail with a
  subject-count shape mismatch when drawing PPCs from fitted-prior fits.
- `EstimationBaseModel.simulate` now mixes in the lapse, so PPC draws come
  from the same model the likelihood fits.
- JAX-compilability: index gathers (`pt.arange` over symbolic lengths) in the
  parabolic theta-hat refinement and in `bin_probability` replaced by one-hot
  contractions; fitted-prior tensors get static shapes. Hierarchical fits now
  JIT under numpyro (previously failed within minutes on multi-subject data).
- Memory: rank-5 broadcast-and-sums replaced by `pt.matmul` (cut O(N^3)
  intermediates that OOM'd at grid 101); drops deprecated `pt.batched_dot`.
- The q_per loss argmin is refined to sub-grid resolution (parabolic vertex),
  making the likelihood smooth in `kappa_r`; the plain grid argmin produced a
  step function that NUTS could not integrate.

### Added — `bauer.efficient_coding` (Bedi et al. 2026)

Models: `EfficientPerceptionModel` (efficient coding + Bayesian decoding in
orientation space, `kappa_r`), `EfficientValuationModel` (veridical
perception, efficient coding in value space, `sigma_rep`),
`SequentialEfficientCodingModel` (both stages), and
`CategoricalSequentialModel` — previously a stub, now the paper's Fig. 6 hard
category gate at 90 deg / 22 CHF (no free parameters).

Options (all default off / backward-compatible unless noted):

- `fit_prior_weight` — fitted prior peakedness `p(phi) ~ 1 - w|sin phi|`,
  nesting uniform (w=0) and the paper's long-term prior (w=0.5).
- `prior_fourier_order=K` — orientation prior as a K-harmonic circular Fourier
  series (k=1 horizontal-vs-vertical, k=2 cardinal-vs-oblique, k>=3 under a
  `0.5/(k-1)^2` roughness prior).
- `cardinal_peak_width` — narrow von Mises peaks at 0/90 deg mixed into the
  perceptual prior with a free per-subject weight.
- `cardinal_truncation` — truncate orientation perception at 0/90/180 deg.
- `no_seam_crossing` — no 0/180 deg seam crossing (measurement-referenced
  mask, smooth one-cell window).
- `fit_motor_noise` — per-subject Gaussian motor noise on the response axis,
  applied after the category gate (gate now acts on the (S,C,K,V) table).
- `value_kernel='cell'` (**new default**) — cell-integrated (erf) value-noise
  kernel; stays well-defined as `sigma_rep` drops below the grid step.
  `value_kernel='point'` restores the old behaviour for re-evaluating older
  traces.
- Response grid extends down to 0 CHF (latent value domain stays [2, 42]);
  the valuation model is padded onto the response grid too.
- `fit_mapping_shrinkage` — per-subject lambda pulling the internal
  orientation->value mapping toward the straight line.
- `kappa_r` prior capped at what the orientation grid can resolve.

### Added — other

- `bauer.utils.diagnostics.trim_stuck_chains` — conservative removal of a
  chain isolated from an agreeing majority on group-level `*_mu` parameters
  (never gates on log-density).
- Tutorial lesson 7 (numerosity estimation) re-run against the hierarchical
  subject-axis fix.
- `notes/choice_vs_ddm_decision_noise_inconsistency.md` — design note on the
  static-choice vs DDM/RDM noise normalisation (addressed by the opt-in
  `consistent_choice_noise` flag).

### Maintenance

- Version strings made consistent (`pyproject.toml`, `bauer.__version__`,
  `CITATION.cff`); bumpversion config now targets `pyproject.toml` and
  `CITATION.cff` instead of the removed `setup.py`.

## 0.3.0 (2026-06-10)

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
