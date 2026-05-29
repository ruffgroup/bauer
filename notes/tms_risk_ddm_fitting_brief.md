# Brief: fitting risky-choice DDMs with bauer (for the tms_risk agent)

A self-contained guide for fitting drift-diffusion / cognitive models to the
TMS risky-choice data using the **bauer** library. Hand this to the tms_risk
fitting agent.

## Before anything: get a current bauer

bauer recently gained a **starting-point finder** that makes DDM/regression
fits converge reliably. Make sure your bauer is up to date and installed
editable in the fitting env:

```bash
cd ~/git/bauer && git pull            # (or merge main into your branch)
~/data/conda/envs/tms_risk_cuda/bin/pip install -e .   # if the repo was refreshed
```

If your fits were run before this, **re-fit** — earlier DDM/regression fits
were a seed lottery (often non-converged even when they looked fine).

## The recipe

```python
import pandas as pd
from bauer.utils.data import load_dehollander_tms_risk
from bauer.models import DDMRiskRegressionModel

df = load_dehollander_tms_risk()          # 35 subjects, sessions 2/3
df = df[df['rt'] >= 0.20].copy()          # 1. drop fast RTs (essential, see below)

m = DDMRiskRegressionModel(               # 2. risky-choice DDM with a covariate
        paradigm=df,
        fit_separate_evidence_sd=True,    # n1 vs n2 encoding noise can differ
        prior_estimate='full',            # estimate the Bayesian-observer prior
        regressors={'n1_evidence_sd': 'C(stimulation_condition)',
                    'n2_evidence_sd': 'C(stimulation_condition)'})
m.build_estimation_model(data=df, hierarchical=True)     # 3. build (35 subjects)
idata = m.sample(draws=1000, tune=2000, chains=4,        # 4. sample
                 target_accept=0.99, backend='numpyro')
# 5. CHECK before interpreting: max r̂ ≤ 1.01, min ESS ≥ 400, few divergences
```

Data conventions: `choice` boolean (`True` = option 2 / second alternative),
`rt` in seconds, columns `n1/n2/p1/p2`, `subject` in the index for
hierarchical fits. Only regress a parameter on `stimulation_condition` if you
have a prior reason it should move under TMS — don't data-mine all of them.

## What matters for convergence

1. **Filter `rt < 0.20 s`.** bauer's WFPT likelihood has a flat, zero-gradient
   region when `t0` exceeds the fastest RT; the sampler gets stuck there. This
   is the #1 cause of stuck chains. Tune the cutoff to the task's motor floor.
2. **Leave the starting-point finder on (it is, by default).** bauer starts
   each chain at a data-informed plausible point (`find_MAP`) dispersed by a
   fraction of each parameter's prior SD. This is what makes these models
   converge — with generic init they're a seed lottery (the *same* config can
   give r̂ from 1.0 to 3+ across seeds). It is on automatically for DDM/Race;
   `find_init=False` disables it. It handles **every** parameter — including
   B-spline noise coefficients (below) — with no per-parameter tuning.
3. **`tune=2000`, `target_accept=0.99`.** `numpyro` `chain_method='vectorized'`
   on a GPU is the fast, correct default — you do **not** need
   `chain_method='parallel'`. Always check r̂/ESS afterwards.

## TMS-specific: different noise functions per stimulation condition

If the hypothesis is that TMS changes the *shape* of encoding noise across
magnitudes (not just a level), use the flexible (B-spline) noise variant and
regress its coefficients on the condition:

```python
from bauer.models import DDMFlexibleNoiseRiskRegressionModel
m = DDMFlexibleNoiseRiskRegressionModel(
        paradigm=df, prior_estimate='full', fit_separate_evidence_sd=True,
        spline_order=5,
        regressors={'n1_evidence_sd': 'C(stimulation_condition)',
                    'n2_evidence_sd': 'C(stimulation_condition)'})
m.build_estimation_model(data=df, hierarchical=True)
```

bauer auto-expands the formula across all spline coefficients, and the finder
initialises them automatically. **Note:** the spline basis is fixed at model
construction from the paradigm's stimulus range (not the fit data), so build
the model with a paradigm spanning the full range of `n1`/`n2` you'll fit.

**Known gotcha:** the *non-hierarchical* flexible build path currently errors
(`cauchy_sigma_intercept`). Fit flexible models **hierarchically**
(`hierarchical=True`) — which is what you want for 35 subjects anyway.

## If a fit won't converge

1. Confirm `rt < 0.20 s` was filtered.
2. **Weak data** (`P(choice) ≈ 0.5` in a condition → flat drift → broad
   posterior): no sampler fixes it. Drop the regression on parameters the
   covariate shouldn't move (often `a`, `t0`); rely on the hierarchical group
   to constrain weak subjects.

Avoid `backend='blackjax'` (incompatible with HSSM's progress bar on GPU).

## Cluster (sciencecluster)

```bash
sbatch --job-name=tms_ddm --gres=gpu:L4:1 --partition=lowprio --time=04:00:00 \
    bauer/scripts/slurm_jobs/run_fit.sh bauer_cuda \
    bauer.scripts.fit_dehollander_tms <args>
```

`git pull` on the cluster first; GPU env is `bauer_cuda` (or `tms_risk_cuda`).
Verify `import bauer` works before submitting a long job.

---

More detail (optional): `notes/fitting_ddm_models.md` (general recipe),
`notes/experiments/ddm_sampler_experiments.md` (the experiment showing the
finder takes the failing config from ~12 % to 100 % convergence).
