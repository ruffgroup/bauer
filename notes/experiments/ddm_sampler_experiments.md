# Computational experiment: what makes a bauer regression-DDM converge?

**Status: RUNNING (results pending).** This document records a *controlled*
sampler experiment, not anecdote. It exists because the convergence guidance
("regression DDMs need `chain_method='parallel'`") was inferred from
uncontrolled runs where `tune` and `chain_method` changed together — so we
couldn't actually attribute the fix. This experiment isolates each factor.

## Question

A *basic* DDM converges on numpyro `vectorized`; a *regression* DDM (Garcia
n=64 `ddm_isi`, and the Alina per-subject gain/loss fits) does not (r̂≈1.6).
**Why?** Candidate explanations, to be separated:

1. **Model difficulty** — the regression adds an intercept↔slope ridge on top
   of the `a↔t0↔drift` ridge, so it's just harder and needs more warmup.
2. **`chain_method`** — `vectorized` shares one `vmap`; maybe it couples chains.
3. **Backend** — does `blackjax` (untested to date — it only ever *crashed* on
   the HSSM progressbar) do better or worse than `numpyro`?
4. **Seed luck** — is a single "r̂=1.6" just one unlucky realization?

## Design

Testbed: **`pil03`** (Alina pilot, near-chance in losses → a genuinely hard,
weakly-identified regression case), ~314 trials, per-subject — small enough to
fit in minutes on an L4 so we can run a real grid. Same data fit two ways:

- **basic** = `DDMRiskModel` (no domain regression)
- **regression** = `DDMRiskRegressionModel`, `{n1_evidence_sd, n2_evidence_sd, a} ~ C(domain)`, no `t0` regression

crossed with **backend** {numpyro, blackjax} × **chain_method**
{vectorized, parallel} × **tune** {2000, 4000} × **seed** {0,1,2} (fractional —
see `run_ddm_sampler_experiment.py` for the exact cell list). All:
`target_accept=0.99`, `draws=tune/2`, 4 chains, `fit_seperate_evidence_sd`,
`prior_estimate='full'`, `hierarchical=False`. Convergence judged on the
cognitive parameters (`n*_evidence_sd`, `a`, `t0`, prior μ/σ): **max r̂** and
**min ESS_bulk**, plus divergences and wall-time.

Reproduce:

```bash
sbatch notes/experiments/run_ddm_sampler_experiment.sh    # cluster GPU
# or: python notes/experiments/run_ddm_sampler_experiment.py --out <tsv>
```

Raw results land in `ddm_sampler_results.tsv` (one row per cell, written
incrementally).

## Results

_(Pending the run — table + plot inserted here, then the conclusion. The
guidance in `notes/fitting_ddm_models.md` will be reconciled to whatever this
shows: if `vectorized` at tune=4000 converges the regression model, the
recommendation becomes "more warmup", not "parallel".)_

## Confirmation on the production case

The fast pil03 testbed is cross-checked against the real target — Garcia n=64
hierarchical `ddm_isi` — via the lesson-8 fits (`fit_for_lesson8.py`): vectorized
tune=2000 gave r̂=1.60; the parallel tune=4000 re-fit result will be recorded
here once it lands.
