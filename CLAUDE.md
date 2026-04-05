# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**bauer** — Bayesian Estimation of Perceptual, Numerical and Risky Choice. A PyMC-based framework for fitting hierarchical Bayesian cognitive models to behavioral decision-making data (psychophysics, magnitude comparison, risky choice).

## Setup

```bash
pip install -e .
```

Runtime dependencies (not declared in setup.py — install manually):
`pymc`, `pytensor`, `pandas`, `numpy`, `patsy`, `arviz`, `scipy`, `seaborn`, `matplotlib`

## Commands

```bash
make lint          # flake8 bauer tests
make test          # unittest discovery
make coverage      # coverage report
make docs          # Sphinx HTML
```

There are currently no meaningful automated tests — `tests/test_bauer.py` is a stub.

## Architecture

### Core class hierarchy

`BaseModel` (`core.py`) is the abstract base for all models. Two mixins extend it:
- **`LapseModel`** — adds a `p_lapse` parameter for random-choice trials
- **`RegressionModel`** — adds patsy formula support for trial-level covariate modulation of parameters

Models are built by multiple inheritance, e.g. `PsychometricLapseRegressionModel(PsychometricModel, LapseModel, RegressionModel)`.

`models.py` (~1750 lines) implements ~21 model classes across three task families:

| Family | Key classes |
|--------|-------------|
| Psychometric | `PsychometricModel` and variants |
| Magnitude comparison | `MagnitudeComparisonModel`, `FlexibleNoiseComparisonModel` |
| Risky choice | `RiskModel`, `ProspectTheoryModel`, `LossAversionModel`, `RNPModel`, `FlexibleNoiseRiskModel`, `ExpectedUtilityRiskModel` |

Each family has `*LapseModel`, `*RegressionModel`, and `*LapseRegressionModel` variants.

### BaseModel key methods

- `build_estimation_model(data, hierarchical=True)` — constructs the PyMC model; must be called before sampling
- `sample(draws, tune, ...)` — MCMC via PyMC; returns `arviz.InferenceData`
- `fit_map()` — MAP point estimate
- `predict(paradigm, idata)` — out-of-sample predictions
- `simulate(paradigm, parameters)` — generate synthetic choices
- `ppc(paradigm, idata)` — posterior predictive checks
- `get_subjectwise_parameter_estimates(idata)` / `get_groupwise_parameter_estimates(idata)` — extract posteriors

### Data conventions

- `choice` column must be **boolean** (`True` = chose option 2 / second alternative)
- `subject` must be in the DataFrame index or as a column for hierarchical models
- Paradigm columns vary by model family: `x1`/`x2` for psychometric; `n1`/`n2` for magnitude; `n1`/`n2`/`p1`/`p2` for risk

### Included example data

```python
from bauer.utils.data import load_garcia2022
df = load_garcia2022(task='magnitude')  # or task='risk'
```

### Parameter transforms

`BaseModel` applies transforms to keep parameters in valid ranges. The transform per parameter is set in each subclass and applied/inverted automatically. Available transforms: `identity`, `softplus` (for positive params like σ), `logistic` (for [0,1] params like lapse rate).

### Regression models

`RegressionModel` accepts a `formula` dict mapping parameter names to patsy formulas. Design matrices are built once via `build_design_matrix()` and can be updated for new data via `rebuild_design_matrix()`. Use `get_conditionwise_parameters()` to extract posterior means at specific covariate values.
