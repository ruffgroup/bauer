# PowerLawNoise DDM/RDM models in bauer

For your collaborator's Claude. Drop this whole file (or its contents) into the chat and you'll have everything needed.

## What's new

8 new classes added to bauer (commit `bbc6d4e+`) — DDM and Race-Diffusion variants of the existing `PowerLawNoise*` classes:

```python
from bauer.models import (
    # DDM (Wiener WFPT) × PowerLawNoise
    DDMPowerLawNoiseComparisonModel,
    DDMPowerLawNoiseComparisonRegressionModel,
    DDMPowerLawNoiseRiskModel,
    DDMPowerLawNoiseRiskRegressionModel,

    # Race-Diffusion (Wald-race) × PowerLawNoise
    RaceDiffusionPowerLawNoiseComparisonModel,
    RaceDiffusionPowerLawNoiseComparisonRegressionModel,
    RaceDiffusionPowerLawNoiseRiskModel,
    RaceDiffusionPowerLawNoiseRiskRegressionModel,
)
```

All inherit from the cognitive front-end (`PowerLawNoiseComparisonModel` / `PowerLawNoiseRiskModel`) and add either `DDMMixin` or `RaceMixin` for the joint choice + RT likelihood. The cognitive part is unchanged: noise SD follows a power law in magnitude,

```
σ_k(n) = exp(log_sd_intercept_k) · n^noise_exponent
```

with `noise_exponent` shared across n1/n2 (it characterises the representational geometry — `0` = linear, `1` = Weber's law / log scale).

## What gets fit

| class | new parameters (relative to choice-only) |
|---|---|
| `DDMPowerLawNoiseComparisonModel` | `a`, `t0`, optionally `v_scale` |
| `RaceDiffusionPowerLawNoiseComparisonModel` | `a`, `t0`, `w_0`, `w_d`, `w_s` (with `advantage=True`) |

Plus the cognitive `noise_exponent`, `n1_log_sd_intercept`, `n2_log_sd_intercept` and (if `fit_prior=True`) `prior_mu`, `prior_sd`.

## Quickstart

```python
from bauer.utils.data import load_garcia2022
from bauer.models import RaceDiffusionPowerLawNoiseComparisonModel

df = load_garcia2022(task='magnitude')   # 64 subjects, ~13k trials

m = RaceDiffusionPowerLawNoiseComparisonModel(
    paradigm=df, fit_prior=True, fit_separate_evidence_sd=True,
    advantage=True,    # default; van Ravenzwaaij decomposition
)
m.build_estimation_model(data=df, hierarchical=True)
idata = m.sample(draws=1000, tune=2000, chains=4, target_accept=0.95)

# Posterior predictive check (long-format; index = paradigm.index + ppc_sample)
ppc = m.ppc(df, idata, n_posterior_samples=200)
```

Use `tune=2000` rather than the default 1000 — flex/regression variants of these models often need more warmup for the mass matrix to stabilise. `target_accept=0.95` is the bauer default; if you see r̂ > 1.05, bump to 0.97 or 0.99.

## Regression versions

Same pattern as `FlexibleNoiseRiskRegressionModel`. Pass a `regressors` dict that maps parameter names to patsy formulas. The DataFrame needs the regression columns accessible (move them from index to columns first):

```python
from bauer.utils.data import load_dehollander_tms_risk
from bauer.models import RaceDiffusionPowerLawNoiseRiskRegressionModel

df = load_dehollander_tms_risk()
df_for_fit = df.reset_index().set_index('subject')   # promote stim cond etc.

m = RaceDiffusionPowerLawNoiseRiskRegressionModel(
    paradigm=df_for_fit,
    regressors={
        'noise_exponent': 'stimulation_condition',
        'a': 'stimulation_condition',     # also let boundary depend on TMS
    },
    prior_estimate='full', fit_separate_evidence_sd=True,
)
m.build_estimation_model(data=df_for_fit, hierarchical=True)
idata = m.sample(draws=1000, tune=2000, chains=4)
```

## Backend choice

bauer's CLI scripts support `--backend {pymc,numpyro,blackjax}`. For these power-law models, **JAX (numpyro) is much faster on GPU**:

- pymc CPU: baseline (slowest)
- numpyro CPU: 1.5–3× faster
- numpyro GPU L4 with `chain_method='vectorized'`: 5–30× faster

If running on a single GPU (most cluster setups), **`chain_method='vectorized'` is critical** — `'parallel'` (the default) falls back to running chains sequentially on a 1-device host. The bauer fit scripts pass this for you, but if you call `pm.sampling.jax.sample_numpyro_nuts` directly, set it explicitly.

```python
from pymc.sampling.jax import sample_numpyro_nuts
with m.estimation_model:
    idata = sample_numpyro_nuts(
        draws=1000, tune=2000, chains=4,
        target_accept=0.95,
        chain_method='vectorized',   # ← critical on 1 GPU
    )
```

## What can `noise_exponent` regression buy you?

The headline use case is "does manipulation X compress or stretch the internal magnitude representation?". With `regressors={'noise_exponent': 'condition'}`, you get a posterior over `noise_exponent` per condition level. This lets you ask:

- Does TMS over IPS push the exponent toward Weber's law (1.0) or linear (0.0)?
- Does stake size change the exponent (log-perception only emerges at high stakes)?
- Does training shift the exponent toward 0 (more linear / Stevens-like)?

Compared to the flex (B-spline) variants, the power-law parameterisation is more constrained — it can't capture arbitrary noise functions, but it gives a single interpretable exponent per condition.

## Diagnostics quick-check

After fitting:

```python
import arviz as az
az.summary(idata, kind='diagnostics')         # max r̂, ess
idata.sample_stats['diverging'].sum()         # divergence count
```

Targets: r̂ ≤ 1.01, min ESS ≥ 100/chain, divergences < 1% of draws. If any are off, try `tune=2000`, `target_accept=0.97`. Power-law model usually behaves better than the flex/spline variant (fewer free parameters), so 1000 warmup is often enough — but 2000 is safer.

## Pointers in bauer

- Source: `bauer/models/ddm.py` (DDM × PowerLaw, 4 classes) and `bauer/models/race.py` (RDM × PowerLaw, 4 classes).
- Cognitive base: `bauer/models/magnitude.py:PowerLawNoiseComparisonModel`, `bauer/models/risky_choice.py:PowerLawNoiseRiskModel`.
- Existing FlexibleNoise versions follow the exact same pattern — see `DDMFlexibleNoiseRiskRegressionModel` for analogy.
- Project-level CLAUDE.md in the repo root covers the full repo layout and the `--backend` story.

## Testing your fit

bauer's smoke test for these models lives in `tests/test_models.py`. To verify your install:

```bash
python -m pytest tests/test_models.py -k PowerLaw -x
```
