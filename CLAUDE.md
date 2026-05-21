# CLAUDE.md

Project guidance for Claude Code working in this repo.

## Project

**bauer** — Bayesian Estimation of Perceptual, Numerical and Risky Choice. PyMC-based hierarchical Bayesian cognitive models for behavioural decision-making (psychophysics, magnitude comparison, risky choice). Core models include static-choice (cumulative-normal Bernoulli), DDM (Wiener-WFPT), and Race-Diffusion (analytical Wald) likelihoods, all sharing the same Bayesian observer cognitive front-end.

## Setup

```bash
pip install -e .
```

The DDM/race likelihoods need extras: `pip install hssm jax numpyro blackjax`. The `environment.yml` (CPU) and `environment_cuda.yml` (GPU, with `jax[cuda12]`) capture the full env.

## Repo layout

```
bauer/
├── bauer/                          # the package
│   ├── core.py                     # BaseModel, RegressionModel, LapseModel mixins
│   ├── models/
│   │   ├── psychophysics.py        # PsychometricModel + variants
│   │   ├── magnitude.py            # MagnitudeComparisonModel,
│   │   │                           # FlexibleNoiseComparisonModel (B-spline noise),
│   │   │                           # PowerLawNoiseComparisonModel
│   │   ├── risky_choice.py         # RiskModel, ProspectTheoryModel,
│   │   │                           # FlexibleNoiseRiskModel + Regression
│   │   │                           # ExpectedUtilityRiskModel,
│   │   │                           # PowerLawNoiseRiskModel
│   │   ├── ddm.py                  # DDMMixin + DDM*ComparisonModel,
│   │   │                           # DDM*RiskModel (incl. Flex + Regression)
│   │   └── race.py                 # RaceMixin + RaceDiffusion*ComparisonModel,
│   │                               # RaceDiffusion*RiskModel (incl. Flex + Regression)
│   ├── data/                       # bundled CSVs (Garcia 2022, dehollander 2024, TMS)
│   ├── utils/                      # data loaders, bayes utils, math, plotting
│   └── scripts/
│       ├── fit_garcia.py           # CLI: magnitude comparison fits
│       ├── fit_dehollander2024.py  # CLI: dotcloud (N=30) + symbolic (N=58)
│       ├── fit_dehollander_tms.py  # CLI: TMS risky choice (N=35 sessions 2/3)
│       └── slurm_jobs/             # cluster sbatch wrappers
│           ├── run_fit.sh          # generic fit runner (4 cores; logs ~/logs/)
│           ├── build_cuda_env.sh   # creates bauer_cuda env on a GPU node
│           ├── install_hssm.sh     # adds hssm/jax/numpyro to existing bauer env
│           ├── submit_jax_experiment.sh    # backend benchmark (pymc/numpyro × CPU/GPU)
│           └── submit_all_production.sh    # full per-dataset fit sweep
├── notebooks/
│   ├── lib/plotting.py             # shared seaborn-based plotting helpers
│   ├── analyze_{garcia,dehollander,tms}.py # reproducible analysis pipelines
│   │                               # used by the slurm post-fit step
│   └── {garcia,dehollander_dotcloud,dehollander_symbolic,tms}_report.py
│                                   # per-dataset reports — diagnostics, group +
│                                   # per-subject posteriors, PPCs, σ_k(n)
├── notes/                          # design docs (race_diffusion_math.md, papers/)
├── tests/test_models.py            # smoke + parametric tests for risky/DDM/RDM
└── docs/                           # Sphinx
```

## Commands

```bash
make lint          # flake8 bauer tests
make test          # unittest discovery
make coverage      # coverage report
make docs          # Sphinx HTML
```

## Architecture

### Inheritance

`BaseModel` (in `core.py`) is abstract. Two mixins extend it:
- **`LapseModel`** — adds `p_lapse` for random-choice trials.
- **`RegressionModel`** — adds patsy-formula regression on free parameters; for flex models, formulas targeting `n1_evidence_sd` etc. are auto-expanded across spline coefficients.

Models are built by multiple inheritance, e.g.

```python
class DDMFlexibleNoiseRiskRegressionModel(
    DDMMixin, FlexibleNoiseRiskRegressionModel,
): ...
```

DDM/Race **mixins** (`DDMMixin`, `RaceMixin`) swap the static cumulative-normal likelihood for a Wiener WFPT (DDM) or analytical Wald-race (RDM). The cognitive front-end (Bayesian observer with priors, asymmetric noise, memory model) is reused unchanged.

### Race model: `advantage` decomposition

`RaceMixin` defaults to `advantage=True`:

```
μ_i = w_0 + w_d·(tilde_i − tilde_j) + w_s·(tilde_i + tilde_j)
```

(van Ravenzwaaij 2020). The ablation `advantage=False` (`μ_i = w_0 + tilde_μ_i`) is broken on choice — both accumulators race nearly neck-and-neck, giving a flat psychometric. Don't use it as the default. Diffusion noise σ = 1, no across-trial drift variability (sequential-evidence-stream interpretation; see `notes/race_diffusion_math.md`).

### BaseModel key methods

- `build_estimation_model(data, hierarchical=True)` — constructs the PyMC model; required before sampling.
- `sample(draws, tune, target_accept, ...)` — pymc NUTS (default).
- `predict(paradigm, idata)` — out-of-sample prediction (probability of choice).
- `simulate(paradigm, parameters)` — generate synthetic choices/RTs from posterior samples.
- `ppc(paradigm, idata, n_posterior_samples=200)` — posterior predictive checks. Returns DataFrame with `(trial_keys, ppc_sample)` index and `simulated_choice` (bool) column. DDM/Race versions also have `simulated_rt`.

### Data conventions

- `choice` column: **boolean** (`True` = chose option 2 / second alternative).
- DDM/RDM also need `rt` (seconds, > 0).
- Hierarchical fits need `subject` in the DataFrame index or column.
- Paradigm columns by family:
  - Psychometric: `x1`/`x2`
  - Magnitude: `n1`/`n2`
  - Risk: `n1`/`n2`/`p1`/`p2`

### Bundled datasets

```python
from bauer.utils.data import (
    load_garcia2022,           # Barreto-Garcia 2022 magnitude comparison (N=64)
    load_dehollander2024_risk, # de Hollander dotcloud risky choice (N=30)
    load_dehollander2024_symbolic,  # de Hollander symbolic risky choice (N=58)
    load_dehollander_tms_risk, # TMS risky choice (N=35 in sessions 2/3)
)
```

### Parameter transforms

`BaseModel` keeps params in valid ranges via per-parameter transforms (`identity`, `softplus`, `logistic`). Set in each subclass; applied/inverted automatically.

### Regression on flex models

`FlexibleNoiseRiskRegressionModel.__init__` auto-expands a formula on a base noise name to all spline coefficients:

```python
m = DDMFlexibleNoiseRiskRegressionModel(
    paradigm=df_with_subject_as_column,
    regressors={'n1_evidence_sd': 'stimulation_condition',
                'n2_evidence_sd': 'stimulation_condition',
                'a': 'stimulation_condition'},
    spline_order=5,
)
# becomes 5 entries each for n1/n2_evidence_sd_spline{1..5} + 1 for a
```

For TMS analyses (different noise functions per stim condition), pass `stimulation_condition` as the formula RHS on the noise-base names.

## Fitting workflow

### Local (small/medium fits)

```bash
python -m bauer.scripts.fit_garcia rdm --flex --n-subjects 8 \
    --backend pymc --no-ppc --out-dir results/garcia
```

### Cluster (production, GPU-accelerated)

Submit single fits via `slurm_jobs/run_fit.sh`:

```bash
sbatch --job-name=garcia_rdm --gres=gpu:L4:1 --time=01:00:00 \
    bauer/scripts/slurm_jobs/run_fit.sh bauer_cuda \
    bauer.scripts.fit_garcia rdm --n-subjects all --backend numpyro \
    --out-dir /shares/zne.uzh/gdehol/bauer_results/garcia
```

Or run the whole sweep in one shot:

```bash
bash bauer/scripts/slurm_jobs/submit_all_production.sh
```

### Backend choice

`--backend {pymc,numpyro,blackjax}`:
- **pymc** — PyMC's native NUTS. Most reliable, slowest.
- **numpyro** — JAX-backed NUTS via `pm.sampling.jax.sample_numpyro_nuts`. ~1.5–3× faster on CPU; ~5–30× faster on GPU. **Always pass `chain_method='vectorized'` (already set in fit_*.py) so chains run in parallel on a single GPU.**
- **blackjax** — alternative JAX-NUTS implementation. Roughly equivalent to numpyro in wall time; useful as a methods-paper robustness check.

### Hyperparams (current defaults)

- `target_accept=0.95` — 0.99 was overkill on cluster; 0.95 is well-behaved here.
- `tune=1000, draws=1000, chains=4` — solid for these sample sizes. **Bump warmup to 1500-2000 if any fit shows `r̂ > 1.01` or `min ESS < 100/chain` post-hoc.**

### Output filenames

```
<out-dir>/<n>subj/<model>{_flex}{_reg}{_freescale|_fixedscale|_noadvantage}{_<prior_estimate>}.nc
```

PPCs (when not skipped) land alongside as `*_ppc.parquet`.

## Reports / analysis

Per-dataset report scripts in `notebooks/*_report.py` produce, for each fit landed in `/shares/zne.uzh/gdehol/bauer_results/`:

1. Diagnostics table (max r̂, divergences, min ESS).
2. Group-level forest plot (`az.plot_forest`).
3. Per-subject FacetGrid: x = subject sorted by mean, y = posterior mean ± 95% CI. Built with seaborn.
4. Implied σ_k(n) curves (flex models) with 94% HDI bands.
5. **TMS only**: per-condition σ_k(n) for the regression fits (TODO until reg fits land).
6. PPCs: psychometric + chronometric, points = data, lines + shaded HDI = PPC.

Conversion to `.ipynb` once content stabilises:

```bash
jupytext --to ipynb notebooks/garcia_report.py
```

## Wall-time tracking

Cluster `run_fit.sh` appends one row per job to `~/logs/bauer_runtimes.tsv`:

```
job_id  job_name  host  env  elapsed_sec  exit  start  end  cmd
```

Inspect with:

```bash
column -t -s$'\t' ~/logs/bauer_runtimes.tsv | sort -k5n
```

Used for the GPU-vs-CPU benchmark and any future bauer methods paper. Wall time includes JAX JIT compile (~30-120 s for the first chain on numpyro), which inflates timings on small fits but amortises for big ones.
