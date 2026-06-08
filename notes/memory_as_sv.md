# Memory noise as across-trial drift variability (`memory_as_sv`)

Branch: `sv-memory-noise`. Flag: `memory_as_sv=True` on the DDM magnitude
models (defaults OFF). Requires `memory_model='shared_perceptual_noise'`.

## Motivation

bauer's Bayesian-observer DDM normalizes the drift by the *combined* encoding
noise of both stimuli:

```
v = ((μ̂₂ - μ̂₁) + threshold) / sqrt(sd1² + sd2²)
```

Under `shared_perceptual_noise` the two noise components enter this SNR
symmetrically (`n1_evidence_sd = σ_perc + σ_mem`, `n2_evidence_sd = σ_perc`),
so perceptual and memory noise are collinear in the drift and essentially
**unidentifiable** — the data can only constrain their sum.

The scientific fix is to give them **different temporal roles**, justified by
the task structure (n1 shown first, then n2 while n1 is held in memory):

- **Perceptual noise** on the still-visible n2 → the accumulator can draw fresh
  samples over time → *reducible by accumulation* → **within-trial** diffusion
  noise. Costs RT, not asymptotic accuracy.
- **Memory noise** on the held n1 → a single frozen degraded trace that cannot
  be re-sampled → *irreducible* → makes the per-trial drift rate vary from
  trial to trial → **across-trial drift variability `sv`** (Ratcliff). Produces
  **slow errors**.

## Derivation

Prior `V ~ N(μ_p, σ_p²)`. Posterior-mean estimate of accumulator k under
linear-Gaussian shrinkage:

```
μ̂_k = w_k·x_k + (1-w_k)·μ_p,   w_k = σ_p² / (σ_p² + σ_e,k²)
SD[μ̂_k | true V] = w_k · σ_e,k        (see utils.bayes.posterior_mean_sd)
```

Recover the two components from the model inputs (exactly how
`MagnitudeComparisonModel.get_model_inputs` builds them):

```
σ_perc = n2_evidence_sd
σ_mem  = n1_evidence_sd - n2_evidence_sd   (≥ 0 by construction)
```

n1 carries both noise streams through the **same** shrinkage gain `w₁`
(computed from the combined `σ_e,1`). We attribute its posterior-mean SD to the
two physical sources in proportion to their SD:

```
within-trial part of n1 : w₁·σ_perc   (reducible)
frozen part of n1        : w₁·σ_mem    (frozen offset → sv)
```

**Within-trial diffusion noise** (what the accumulator integrates against) gets
only the reducible, perceptual parts of *both* stimuli:

```
sd_DV = sqrt( (w₂·σ_perc)² + (w₁·σ_perc)² )
```

**Drift mean** is the SNR against this perceptual-only noise:

```
v = v_scale · ((μ̂₂ - μ̂₁) + threshold) / sd_DV
```

**Across-trial drift variability.** The frozen memory error shifts the
numerator ΔV by a trial-constant Gaussian amount with `SD = w₁·σ_mem`. A
constant numerator shift scales the drift by `1/sd_DV`, so across trials the
drift rate `v` is Gaussian with

```
sv = v_scale · (w₁·σ_mem) / sd_DV
```

which is exactly the HSSM/Ratcliff `sv` parameter.

### Modelling choice / approximation

`w₁` is the exact linear-Gaussian posterior-mean gain for the *combined*
`σ_e,1`. We then split the resulting posterior-mean output variance between the
perceptual and memory streams in proportion to their input SDs (i.e. each
stream is pushed through the same gain `w₁`). This is the documented
simplification: a fully separate two-stage filter (perceptual filtering, then a
memory-corruption stage) would give slightly different gains, but the
proportional split is exact in the limit where the two streams are independent
Gaussian contributions to `x₁` and `w₁` is treated as the (data-fixed) gain.
The qualitative claim — perceptual is reducible, memory is frozen — is captured
faithfully.

## Implementation choice: which likelihood

HSSM 0.3.0 ships `hssm.likelihoods.logp_ddm_sdv`, an **analytic** Navarro-Fuss
WFPT that integrates over the Gaussian drift distribution (closed form, no
numerical integration). Signature `logp_ddm_sdv(data, v, a, z, t, sv)`, same
`a→2a` (half-boundary) convention as `logp_ddm`, with `sv = SD of drift`.
`ssm-simulators` exposes a matching `ddm_sdv` simulator (`[v,a,z,t,sv]`) used by
`simulate`/`ppc`. So the route is fully analytic and faithful — no
approximation in the likelihood itself.

Code: `bauer/models/ddm.py`
- `_drift_and_sv_from_snr(model_inputs, v_scale)` — the derivation above.
- `DDMMixin.memory_as_sv` flag + `_get_drift_and_sv` (guards
  `shared_perceptual_noise` and rejects `flat_observer_prior`).
- `build_likelihood`, `build_prediction_model`, `build_loglik_model`,
  `simulate` all branch on `memory_as_sv` and use the sdv likelihood/simulator.
- Wired into `DDMMagnitudeComparisonModel` and
  `DDMMagnitudeComparisonRegressionModel` (`memory_as_sv=` kwarg).

## Validation results

### 1. Builds + samples
8 subjects (both groups), `shared_perceptual_noise`, `memory_as_sv=True`,
`tune=300, draws=300, numpyro`. Ran clean, max r̂ = 1.1. Perceptual noise
well-identified (ESS ≈ 460); memory/sv has lower ESS (≈ 46) as expected — the
sv parameter is the harder one. Converges-ish; production runs should use
≥1000/1000 and target_accept 0.9–0.95.

### 2. Slow errors
Matched-drift comparison (sv off via σ_mem=0 + inflated perceptual, vs sv on):

| | error−correct median Δ | error−correct q90 Δ |
|---|---|---|
| no sv (perceptual only) | +0.27 s | +1.09 s |
| with sv (memory→sv)     | +0.48 s | +1.29 s |

sv nearly doubles the error-slowness relative to corrects — the Ratcliff
slow-error signature, as designed.

### 3. Parameter recovery (the key test)
*(filled in by validate_sv.py runs — see RECOVERY RESULTS below)*

## Limitations / honest status

- `sv` is intrinsically weakly identified at modest trial counts (low ESS);
  the perceptual/memory separation relies on the RT *distribution shape*, so it
  needs decent per-subject trial counts and clean fast-RT filtering
  (`rt >= 0.20`).
- The proportional variance-split (single `w₁`) is an approximation to a true
  two-stage filter; see modelling-choice note above.
- Only wired into the magnitude-comparison DDM classes (plain + regression).
  Risk / flexible-spline / power-law DDM variants inherit the mixin machinery
  but have not been validated with `memory_as_sv`.
- `flat_observer_prior` is explicitly rejected (the sv derivation needs the
  shrinkage gains).
