# Briefing: `RegressionModel` softplus-prior bug (fixed 2026-05-23)

A handoff document for any agent / future-Gilles picking up bauer.
Read this before touching `bauer/core.py:RegressionModel` or any
`*RegressionModel` subclass.

## TL;DR

`bauer/core.py:RegressionModel.build_hierarchical_nodes` (and
`build_prior`) had a missing `softplus` branch in its transform
dispatch. For every softplus-transformed parameter — which is **most
of bauer's regression-model parameters** (`n1/n2_evidence_sd`, `a`,
`t0`, `v_scale`, all flexible-noise spline coefficients, prospect
theory α/β, lapse rate, etc.) — the Intercept prior was silently
defaulting to `Normal(0, sigma_regressors)` on the untransformed
scale instead of the model-declared `Normal(mu_intercept,
sigma_intercept)`.

**Fixed in `bauer/core.py`** — see commit message and the diff at
`build_prior` (~line 736) and `build_hierarchical_nodes` (~line 768).

## Why this matters (and where to recheck things)

Every `*RegressionModel` fit ever produced from this codebase had the
wrong intercept prior. For models with strong data signal (hundreds
of trials per subject) the posterior **mean** is probably ~OK because
the data dominates the prior. But:

- **Convergence diagnostics** (r̂, ESS, divergences) on those fits are
  not trustworthy. NUTS was navigating a posterior with a misspecified
  prior; some chains got stuck in regions the prior shouldn't have
  allowed.
- **Credible intervals** are likely too wide or biased.
- **The direction of effects** is probably preserved (data dominates
  means), but **the magnitude and uncertainty are not**.

Audit candidates (regression DDM/probit/RDM fits known to exist):
- `tms_risk` — TMS condition regressions on `a` / `n*_evidence_sd`.
  → Anything fit before 2026-05-23 should be re-run.
- `risk_experiment` — regression models for stimulation_condition,
  session interactions.
- `bauer/scripts/fit_dehollander_tms.py` outputs.
- Any user code using `MagnitudeComparisonRegressionModel`,
  `DDMMagnitudeComparisonRegressionModel`,
  `RaceDiffusionFlexibleNoiseRiskRegressionModel`, etc.

## The bug, concretely

`bauer/core.py` (BEFORE fix):

```python
# RegressionModel.build_hierarchical_nodes (around line 773)
if self.design_matrices[name].design_info.column_names[0] == 'Intercept':
    if name in ['n1_evidence_mu', 'n2_evidence_mu']:
        warnings.warn(...)
    if transform == 'identity':
        mu[0] = mu_intercept
        sigma[0] = sigma_intercept
    elif transform == 'logistic':
        mu[0] = mu_intercept
        sigma[0] = sigma_intercept
    # Possibly use inverse of softplus     ← MISSING BRANCH
```

The `softplus` branch was never written. Same omission in
`RegressionModel.build_prior` a few lines earlier. Both are now:

```python
# AFTER fix:
if self.design_matrices[name].design_info.column_names[0] == 'Intercept':
    if name in ['n1_evidence_mu', 'n2_evidence_mu']:
        warnings.warn(...)
    # Regression operates on the *untransformed* scale (softplus /
    # logistic / identity is applied later in get_trialwise_variable),
    # so the Intercept prior is Normal(mu_intercept, sigma_intercept)
    # regardless of which transform the parameter uses.
    mu[0] = mu_intercept
    sigma[0] = sigma_intercept
```

## How the bug manifested

Concrete impact on `DDMMagnitudeComparisonRegressionModel` on Garcia
2022 (n=64) with `regressors={'n1_evidence_sd': 'isi_cat'}`:

| param | basic-model prior on `_mu_untransformed` | regression-model prior on `_mu` (before fix) | resulting issue |
|---|---|---|---|
| `t0` | `Normal(-1.508, 1.0)` → softplus → ~0.20 s | `Normal(0, 1.0)` → softplus → **~0.69 s** | Larger than most RTs → triggers the LOGP_LB=−66.1 flat-floor pathology in HSSM's `logp_ddm` → NUTS chains stuck with zero gradient |
| `a` | `Normal(0.84, 1.0)` → softplus + 0.3 → ~1.5 | `Normal(0, 1.0)` → softplus + 0.3 → ~0.99 | Prior 3× wider than HSSM-tuned; a↔ν identifiability ridge un-anchored |
| `n1_evidence_sd` | `Normal(-1, 0.5)` → softplus → ~0.31 | `Normal(0, 1.0)` → softplus → ~0.69 | Drift formula `(post_n2-post_n1)/sqrt(σ1²+σ2²)` numerator/denominator collapse, likelihood near-flat in (v_scale × σ) |

The `t0` row is the killer: it directly triggers the unrelated-but-
also-real LOGP_LB pathology that we'd already fixed for the basic DDM
via RT-filtering and HSSM-style priors. Because the bug zeroed out
those careful tunings whenever a regression was added, the regression
model fell straight back into the dead-zone trap.

**Symptoms** (every regression DDM fit, including intercept-only):
- r̂ ≈ 2-4 on every group-level mean (vs ≈ 1.0 for the basic DDM)
- ESS bulk ≈ 4-7 per parameter (vs 1000+)
- Per-chain inspection: 1-2 chains frozen with within-chain SD ≈ 0,
  others in different basins
- Bumping `tune=2000` / `target_accept=0.99` did not help (prior was
  wrong, not under-sampled)
- Switching `chain_method='parallel'` did not help (every chain hit
  the same misspecified prior)

## How to verify the fix

```python
from bauer.models import DDMMagnitudeComparisonModel, DDMMagnitudeComparisonRegressionModel
from bauer.utils.data import load_garcia2022

df = load_garcia2022('magnitude'); df = df[df.rt >= 0.20]
subs = df.index.get_level_values('subject').unique()[:8]
df = df[df.index.get_level_values('subject').isin(subs)].copy()

m_basic = DDMMagnitudeComparisonModel(
    paradigm=df, fit_separate_evidence_sd=True, fit_prior=True)
m_basic.build_estimation_model(data=df, hierarchical=True)

m_reg = DDMMagnitudeComparisonRegressionModel(
    paradigm=df, fit_separate_evidence_sd=True, fit_prior=True,
    regressors={'n1_evidence_sd': '1'})   # intercept-only
m_reg.build_estimation_model(data=df, hierarchical=True)

# Extract the Normal-prior mu for each model-class
def normal_params(rv):
    ins = list(rv.owner.inputs)
    return float(ins[-2].eval()), float(ins[-1].eval())

for p in ['n1_evidence_sd', 't0', 'a']:
    mu_b, sd_b = normal_params(m_basic.estimation_model.named_vars[f'{p}_mu_untransformed'])
    mu_r, sd_r = (m_reg.estimation_model.named_vars[f'{p}_mu']
                  .owner.inputs[-2].eval()[0],
                  m_reg.estimation_model.named_vars[f'{p}_mu']
                  .owner.inputs[-1].eval()[0])
    assert abs(mu_b - mu_r) < 1e-6, f'{p}: {mu_b} != {mu_r}'
    print(f'{p}: mu={mu_b:+.3f} matches ✓')
```

Expected post-fix output:
```
n1_evidence_sd: mu=-1.000 matches ✓
t0:             mu=-1.508 matches ✓
a:              mu=+0.842 matches ✓
```

(Note: `sigma_intercept` defaults differ between
`BaseModel.build_hierarchical_nodes` (0.5) and
`RegressionModel.build_hierarchical_nodes` (1.0), so `sd_b ≠ sd_r`
for parameters that don't set `sigma_intercept` explicitly. This is
a secondary asymmetry; for DDM-relevant parameters with explicit
`sigma_intercept=1.0` in `DDMMixin._default_free_pars`, both match.)

## Related fixes / context

This bug-find followed days of trying every other lever:
- RT filter at 200 ms (`bauer/models/ddm.py:_get_paradigm` —
  necessary independent of this bug; HSSM logp pathology)
- HSSM-style priors on `a`, `t0` (`bauer/models/ddm.py:DDMMixin` —
  necessary for the basic DDM but doesn't reach the regression model's
  code path)
- `chain_method='parallel'` (correct fix for the basic DDM's
  vectorized-seed coupling, but doesn't help with a misspecified
  prior)
- `tune=2000, target_accept=0.99` (doesn't help when the prior is
  pinned wrong)
- `backend='blackjax'` (crashed with `IO effect not supported in
  vmap-of-cond` — separate JAX/blackjax integration bug)

Detailed sampler-comparison table and other lessons are in
`notes/ddm_convergence_lessons.md`.

## What's still TODO

1. **Run the basic vs intercept-only regression equivalence as an
   actual unit test** (`tests/test_models.py`). The verification
   above is one-shot.
2. **Re-fit production regression DDMs** in dependent codebases (tms_risk,
   risk_experiment). Use the same SLURM scripts they were originally
   submitted with; just delete the cache and rerun.
3. **Audit `BaseModel.build_hierarchical_nodes`** for similar
   transform-conditional asymmetries. None known, but the same author
   wrote both functions, and the bug pattern is mechanical (missing
   branch in if/elif).
4. **The `sigma_intercept` default mismatch** (0.5 in BaseModel vs
   1.0 in RegressionModel) is a separate asymmetry. Probably worth
   unifying at 0.5 (the tighter, sensible default), but verify it
   doesn't break the bauer-style identity-transform priors first.
5. The audit (saved test scripts at `/tmp/audit_bauer*.py` on the
   author's laptop, not committed) noted that `_get_paradigm` MRO
   `try/except` in `RegressionModel._get_paradigm` papers over a
   real signature mismatch with `MagnitudeComparisonModel._get_paradigm`
   that doesn't accept `subject_mapping`. Worth fixing properly.

## Acknowledgements

Bug found by an audit agent (Claude general-purpose) on 2026-05-23
after a multi-day fitting saga on Garcia 2022 n=64 DDM + ISI
regression. The agent's audit report is what unlocked the fix; the
empirical iteration before that was thoroughly misleading because
the symptoms (stuck chains, basin-hopping) looked like a sampler
problem when they were a prior problem.
