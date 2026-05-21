# Briefing: Lesson 6 — Choice + RT models (DDM and Race-Diffusion)

This is for a Claude agent picking up the task of writing a new tutorial on bauer's DDM and Race-Diffusion (RDM) models. Read this first, then `notes/race_diffusion_math.md` for the math, then the docstrings of `DDMMixin` and `RaceMixin`.

## Where everything lives

- **Tutorial format**: `docs/tutorial/make_notebooks.py` is the source. Existing lessons 1-5 are in there as Python functions producing `lessonN.ipynb`. Use the same `code()`/`md()` helpers and `write_if_changed()` pattern. New entry slots in after `# Lesson 5` (~line 2588) as `# Lesson 6`. The function should produce `lesson6.ipynb`.
- **Run after editing**: `python docs/tutorial/make_notebooks.py 6` regenerates only lesson 6.
- **Index**: `docs/tutorial/index.rst` may need a new entry pointing at lesson 6.
- **Reference math**: `notes/race_diffusion_math.md` has the analytical Wald-race derivation, identifiability discussion, and tight-prior limit.

## What models to teach

The lesson should introduce three ideas in sequence:

### 1. Why care about RT?
Choice-only models throw away wall-clock information. Choice + RT models can:
- Distinguish "biased noisy decision" from "unbiased clear decision";
- Falsify cognitive models that fit choice but predict wrong RTs;
- Give better identifiability for some parameters (e.g. evidence_sd vs prior_sd ridges that appear in choice-only fits).

The Garcia 2022 magnitude task is a good demo because there's a clean **size effect**: RT decreases with stake size at fixed difficulty. Choice models can't capture this; RT models can.

### 2. DDM (single accumulator, signed evidence)

bauer's `DDMMagnitudeComparisonModel(DDMMixin, MagnitudeComparisonModel)`:
- **Cognitive front-end** (inherited from MagnitudeComparisonModel): Bayesian observer with prior on log(n), per-stimulus encoding noise σ_1, σ_2.
- **Drift** = SNR of perceived log-magnitude difference: `v = v_scale * (post_n2_mu - post_n1_mu) / sqrt(sd1^2 + sd2^2)`. This is `_drift_from_snr` in `bauer/models/ddm.py`.
- **Likelihood**: HSSM Wiener WFPT (`hssm.likelihoods.logp_ddm`). Three new params: half-boundary `a`, non-decision `t0`, optional `v_scale`.
- **HSSM convention**: `choice = True` → upper boundary (positive drift drives upper). `t0` is in seconds. `a` is half the boundary separation.
- **`fix_z=True` default**: starting point z=0.5 (unbiased). Don't free unless you suspect a starting-point bias.
- The `DDMRiskModel` is the same drift formula but with `log(EU_k) = post_n_k + log(p_k)` carried via `RiskModel.get_model_inputs`. Probabilities enter as a deterministic shift.

### 3. Race-diffusion (separate accumulators per option)

bauer's `RaceDiffusionMagnitudeComparisonModel(RaceMixin, MagnitudeComparisonModel)`:
- **Two parallel Wiener accumulators**, one per stimulus, racing to a common boundary `a`.
- **First-passage time analytically inverse-Gaussian** — no LANs needed. The race likelihood is `logp_race_diffusion_2(data, v1, v2, sigma1, sigma2, a, t)` in `race.py`.
- **Drift formulation matters!** Two options:

```
advantage=True (DEFAULT):
  μ_i = w_0 + w_d·(tilde_i − tilde_j) + w_s·(tilde_i + tilde_j)
  
advantage=False (ablation):
  μ_i = w_0 + tilde_μ_post,i
```

where `tilde = post_n_mu - prior_mu` and `i,j` are the two accumulators. The advantage form (van Ravenzwaaij 2020) is the *only* one that fits choice properly — the no-advantage version produces a flat psychometric. **Demonstrate this in the tutorial!** It's a great pedagogical showcase: you can fit both and show the broken psychometric of `advantage=False`.

- **Sequential evidence stream interpretation** (lock this in): the within-trial Wiener noise σ *is* the per-unit-time sensory noise; the accumulator state is the running posterior estimate. So σ=1 (`pt.ones_like(v1)` — fixed in code, not free). NO across-trial drift variability `s_v`. See Bogacz et al. 2006, Drugowitsch et al. 2012. We had this wrong earlier (β·ν per-accumulator noise) and the user pushed back on it; the corrected story is in `notes/race_diffusion_math.md`.

## Concrete code patterns

### Loading data
```python
from bauer.utils.data import load_garcia2022
df = load_garcia2022(task='magnitude')   # 64 subjects, ~13k trials
# Use a subset for the tutorial — keeps fit times short:
subs = df.index.get_level_values('subject').unique()[:8]
df = df.loc[df.index.get_level_values('subject').isin(subs)].copy()
```

The loader already converts rt to seconds, drops non-responses, and casts choice to bool. RT < 150 ms dropped by default.

### Fitting
For the tutorial, **use pymc backend** (default) so it works without JAX. Mention the JAX backend as an aside for power users:

```python
m = RaceDiffusionMagnitudeComparisonModel(
    paradigm=df, fit_seperate_evidence_sd=True, fit_prior=True,
    advantage=True,
)
m.build_estimation_model(data=df, hierarchical=True)
idata = m.sample(draws=1000, tune=1000, chains=4, target_accept=0.95)
```

8-subj DDM/RDM fits take ~5-15 min on a laptop. That's acceptable for a tutorial. The user *should* run them.

### Predictions (out-of-sample)
- `m.predict(paradigm, idata)` — choice probabilities (uses cognitive front-end only).
- `m.simulate(paradigm, parameters)` — draws (rt, choice) from a parameter set.
- `m.ppc(paradigm, idata, n_posterior_samples=80)` — full posterior predictive. Returns long-format DataFrame with index `(paradigm.index..., ppc_sample)` and columns `simulated_choice` (bool) + `simulated_rt` (float, seconds).

For RT plots, the standard pattern (see existing analyze_garcia.py for full example):
```python
ppc = m.ppc(df, idata, n_posterior_samples=80, progressbar=False)
# Mean RT per (n1, n2, ppc_sample), then aggregate across samples:
per_draw = ppc.groupby(['n1', 'n2', 'ppc_sample'])['simulated_rt'].mean()
band = per_draw.groupby(['n1', 'n2']).agg(['mean',
    lambda s: s.quantile(0.10), lambda s: s.quantile(0.90)])
```

### Posterior summaries
```python
group_pars = m.get_groupwise_parameter_estimates(idata)
subj_pars = m.get_subjectwise_parameter_estimates(idata)
import arviz as az
az.summary(idata, var_names=['a_mu', 't0_mu', 'w_0_mu', 'prior_mu_mu'])
```

## Concrete demonstrations to include

1. **Load Garcia, fit `MagnitudeComparisonModel` (choice only) and `DDMMagnitudeComparisonModel`. Compare posteriors of `n1_evidence_sd`, `n2_evidence_sd`.** The DDM tightens or shifts these because RT info constrains noise scale.

2. **Show the size effect**:
   - Plot mean RT vs n1 (small range stake) for the data — modest decrease.
   - Then PPC from race-A: same direction, often slightly stronger.
   - The choice-only model can't reproduce this (no RT predictions).

3. **Show the advantage decomposition matters**:
   - Fit `RaceDiffusionMagnitudeComparisonModel(advantage=False)` (small fit, 8 subj is fine).
   - Plot psychometric (P(chose 2) vs log(n2/n1)) — flat for `advantage=False`, sigmoidal for `advantage=True`.
   - This is the "look how this default isn't arbitrary" moment.

4. **Diagnostics block**:
   - `az.summary(idata, kind='diagnostics')` → max r̂, ess.
   - `idata.sample_stats['diverging'].sum()` → divergence count.
   - Walk through what you'd do if r̂ > 1.01 (bump warmup, raise target_accept).

5. **Implied σ_k(n) curve** for the flex DDM/RDM:
   - Use `model.make_dm(x=n_grid, variable='n1_evidence_sd')` to get spline basis.
   - Multiply by per-subject coefficient posteriors → softplus → σ.
   - Plot per-subject curves with shaded HDIs, plus the mean.

## Things to NOT do

- **Don't introduce s_v (across-trial drift variability)** as a default. Sequential-evidence-stream interp says no.
- **Don't use the no-advantage default** for race models. Always `advantage=True` unless explicitly demonstrating the failure.
- **Don't suggest target_accept=0.99**. We tried that, it's overkill. 0.95 is the bauer default.
- **Don't use the old `polynomial_order=` kwarg** — it's now `spline_order=` (no backwards-compat alias).
- **Don't fit on the full N=64**. 8-subj is enough for pedagogy and fits in tutorial-acceptable time.

## Style guide (from existing lessons)

- **Markdown cells**: explain the math first, then show code. Mathematical notation in LaTeX.
- **Code cells**: small, self-contained, no skipped intermediate steps.
- **Use seaborn** for plots (`sns.set_theme(context='notebook', style='whitegrid', palette='deep')` at the top). FacetGrid for per-subject comparisons.
- **End each section with a 1-2 sentence takeaway**.
- **Reference papers in markdown**: van Ravenzwaaij 2020 (advantage RDM), Bogacz et al. 2006 (DDM as Bayes-optimal), Tillman, Van Zandt & Logan 2020 (Wald race). PDFs in `notes/papers/`.

## Sanity-test before committing

```bash
python docs/tutorial/make_notebooks.py 6     # regenerate
~/mambaforge/envs/bauer/bin/jupyter nbconvert --to notebook \
    --execute --inplace docs/tutorial/lesson6.ipynb
```

If it runs end-to-end on 8-subj data within ~30 min on a laptop, ship it.

## What's already done that you can reuse

- `notes/race_diffusion_math.md` — math derivations, identifiability, tight-prior limit
- `notebooks/lib/plotting.py` — `diagnostics_summary`, `group_param_forest`, `per_subject_dots`, `implied_noise` helpers
- `notebooks/garcia_report.py` — production report style; copy-paste plotting patterns from there
- The CLAUDE.md (project-level) has a one-page summary of the model class hierarchy
