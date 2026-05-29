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
`target_accept=0.99`, `draws=tune/2`, 4 chains, `fit_separate_evidence_sd`,
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

**Convergence is seed-dominated, and the lever is initialization — not
`chain_method` and not `tune`.** pil03 regression DDM, `tune=2000`,
`target_accept=0.99`; convergence = `max r̂ ≤ 1.01` on the cognitive params:

| init scheme | chain_method | seeds | converged | rate | median r̂ | median walltime |
|---|---|---|---|---|---|---|
| default (bauer's old generic jitter) | vectorized | 16 | 2 | **12 %** | 2.02 | 647 s |
| default | parallel | 2 | 2 | (100 %, n=2) | 1.004 | 639 s |
| **finder** (MAP centre + prior-scaled jitter) | vectorized | 8 | 8 | **100 %** | 1.005 | **174 s** |

Reading this:

- **Same config, different seed → r̂ from 1.00 to 3.4.** With default init,
  vectorized converged 2/16 (12 %). It was a lottery, which is why earlier
  single runs looked like "vectorized is broken."
- **`parallel` is not the fix.** It looked better only because independent
  per-chain seeds make it less likely *all* chains lose the lottery at once —
  but it's the same underlying fragility (and Alina pil02 had failed on
  parallel earlier). Chasing `chain_method` was chasing the symptom.
- **The starting-point finder is the fix: 8/8 (100 %) on the exact config that
  failed 7/8 of the time** — and ~3.7× faster (174 s vs 647 s), because
  well-initialised chains don't fall into the max-tree-depth stall. This is
  bauer's `get_initial_points` / `recommended_init='mapjitter'` (MAP centre +
  per-parameter prior-SD jitter), the same idea HSSM uses.

Independent confirmation on real data: the **Alina** gain/loss DDMs — which
had lost the seed lottery (pil02 failed on parallel) — both converge with the
finder on vectorized (pil02 r̂=1.002, pil03 r̂=1.003).

**Conclusion → guidance:** leave bauer's finder on (default for DDM/Race);
`chain_method='vectorized'` is fine; check r̂. This is reconciled into
`notes/fitting_ddm_models.md`, `CLAUDE.md`, `ddm_convergence_lessons.md §3b`,
and the `ddm-sampler-choice` memory.

## n=64 arm (production headline)

The pil03 arm is a **fast stress-test on a degenerate, near-chance subject** —
useful for cheap iteration, but not representative of a normal fit. The real
target is **Garcia n=64, hierarchical**, basic DDM vs ISI-regression DDM
(`run_ddm_sampler_experiment_n64.py`). Each n=64 regression fit is slow (the hierarchical funnel drives NUTS to max
tree depth with default init), so once the pil03 arm pinned the cause on
initialization, the full old-init n=64 crossing was no longer worth the GPU-hours
(those cells were cancelled). The n=64 evidence we keep:

| n=64 `ddm_isi` (regression), tune=2000 | r̂ |
|---|---|
| vectorized, **default init** | **1.60** (the lottery, on the production case) |
| vectorized, basic DDM (contrast) | 1.013 (basic is easy; regression is the hard one) |
| vectorized, **finder** | _from job 3432017 (lesson-8 re-fit), recorded on landing_ |

The pattern matches pil03: default init makes the n=64 regression DDM a lottery
(r̂=1.60), while the basic DDM is fine — and the finder re-fit (well-behaved so
far: steady ~1.5 s/it, no max-depth stalls) is expected to converge it on
vectorized, confirming init is the lever at production scale too.
