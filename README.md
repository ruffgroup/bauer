# bauer

**Bayesian Estimation of Perceptual, Numerical and Risky Choice**

[![Documentation Status](https://readthedocs.org/projects/bauer/badge/?version=latest)](https://bauer.readthedocs.io/en/latest/?version=latest)

*bauer* is a PyMC-based Python library for fitting hierarchical Bayesian cognitive
models to behavioural decision-making data. It covers three families of
forced-choice tasks:

- **Magnitude comparison** — "which set of dots is more numerous?"
- **Psychometric functions** — general two-alternative sensitivity models
- **Risky choice** — "take the safe option or gamble?"

All models are fitted hierarchically by default: each participant gets their own
parameters, regularised by a shared group-level distribution. This is essential
at the trial counts typical of psychophysics experiments (100–250 per condition).

## Installation

```bash
pip install -e .
```

Runtime dependencies (install manually if not already present):
`pymc`, `pytensor`, `pandas`, `numpy`, `patsy`, `arviz`, `scipy`, `seaborn`, `matplotlib`.

## Quick start

```python
from bauer.models import MagnitudeComparisonModel
from bauer.utils.data import load_garcia2022

data = load_garcia2022(task='magnitude')

model = MagnitudeComparisonModel(paradigm=data, fit_seperate_evidence_sd=True)
model.build_estimation_model(data=data, hierarchical=True)
idata = model.sample(draws=1000, tune=1000)

# Posterior predictive check
ppc = model.ppc(data, idata)

# Subject-level parameter estimates
subj_pars = model.get_subjectwise_parameter_estimates(idata)
```

## Model families

| Family | Key classes |
|--------|-------------|
| Psychometric | `PsychometricModel`, `*LapseModel`, `*RegressionModel` |
| Magnitude comparison | `MagnitudeComparisonModel`, `FlexibleNoiseComparisonModel`, `AffineNoiseComparisonModel` |
| Risky choice | `RiskModel`, `ProspectTheoryModel`, `FlexibleNoiseRiskModel`, `ExpectedUtilityRiskModel` |

Each family has `*LapseModel`, `*RegressionModel`, and `*LapseRegressionModel`
variants. Regression models accept patsy formulas to let any parameter vary by
experimental condition.

## Tutorials

The `docs/tutorial/` directory contains five Jupyter notebooks that build up
from first principles:

1. **Psychophysical modelling** — NLC model, Weber's law, asymmetric noise, hierarchical fitting
2. **Risky choice** — KLW model, noise-driven risk aversion, format effects
3. **Stake effects** — presentation-order interactions, EU vs KLW vs PMCM
4. **Flexible noise curves** — `FlexibleNoiseComparisonModel`, `AffineNoiseComparisonModel`, ELPD model comparison
5. **Why hierarchical modelling beats MLE** — split-half reliability demonstration

Generate the notebooks with `python docs/tutorial/make_notebooks.py`, then
execute them with Jupyter.

## Documentation

Full documentation: https://bauer.readthedocs.io

## License

MIT
