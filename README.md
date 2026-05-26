# bauer

**Bayesian Estimation of Perceptual, Numerical and Risky Choice**

[![Documentation Status](https://readthedocs.org/projects/bauer/badge/?version=latest)](https://bauer.readthedocs.io/en/latest/?version=latest)

*bauer* is a PyMC-based Python library for fitting hierarchical Bayesian cognitive models to behavioural decision-making data. It covers three task families — **psychometric**, **magnitude comparison**, **risky choice** — with three likelihoods on top of the same Bayesian observer cognitive front-end:

- **Static** (cumulative-normal Bernoulli) — choice only
- **DDM** (Wiener WFPT, via `hssm.likelihoods.logp_ddm`) — choice + RT
- **Race-diffusion** (analytical Wald-race) — choice + RT, with optional advantage decomposition

All models hierarchical by default: each participant gets their own parameters, regularised by a shared group-level distribution.

## Quick start (5 minutes)

```bash
pip install git+https://github.com/ruffgroup/bauer.git
pip install hssm jax numpyro       # optional: DDM + JAX-NUTS support
```

```python
from bauer.utils.data import load_garcia2022
from bauer.models import RaceDiffusionMagnitudeComparisonModel

# Bundled data (Barreto-Garcia et al. 2022, N=64, magnitude comparison)
df = load_garcia2022(task='magnitude')

# Race-diffusion with advantage decomposition (default)
m = RaceDiffusionMagnitudeComparisonModel(
    paradigm=df, fit_prior=True, fit_seperate_evidence_sd=True,
)
m.build_estimation_model(data=df, hierarchical=True)
idata = m.sample(draws=1000, tune=1000, chains=4)

# Posterior predictive check
ppc = m.ppc(df, idata, n_posterior_samples=200)
```

Want it 10× faster on a GPU? Use `--backend numpyro`:

```python
from pymc.sampling.jax import sample_numpyro_nuts
with m.estimation_model:
    idata = sample_numpyro_nuts(draws=1000, tune=1000, chains=4,
                                 chain_method='vectorized')
```

## Model families

| Family | Static (choice) | DDM | Race-diffusion |
|---|---|---|---|
| Psychometric | `PsychometricModel` | – | – |
| Magnitude comparison | `MagnitudeComparisonModel`, `FlexibleNoiseComparisonModel` | `DDMMagnitudeComparisonModel`, `DDMFlexibleNoiseComparisonModel` | `RaceDiffusionMagnitudeComparisonModel`, `RaceDiffusionFlexibleNoiseComparisonModel` |
| Risky choice | `RiskModel`, `FlexibleNoiseRiskModel`, `ProspectTheoryModel`, `ExpectedUtilityRiskModel` | `DDMRiskModel`, `DDMFlexibleNoiseRiskModel` | `RaceDiffusionRiskModel`, `RaceDiffusionFlexibleNoiseRiskModel` |

Every family has `*LapseModel` (random-choice lapse rate) and `*RegressionModel` (patsy-formula regression on parameters) variants. Combine them via multiple inheritance. Flex models replace scalar `n_evidence_sd` with B-spline `σ_k(n)` for stimulus-dependent noise.

## Bundled experimental datasets

```python
from bauer.utils.data import (
    load_garcia2022,                # N=64, magnitude (dots) + risky choice
    load_dehollander2024_risk,      # N=30, dotcloud risky choice (3T+7T fMRI)
    load_dehollander2024_symbolic,  # N=58, Arabic-numeral risky choice
    load_dehollander_tms_risk,      # N=35, TMS over IPS / vertex / baseline
    load_bedi2026,                  # N=13, abstract orientation→value pilot
)
```

All loaders apply consistent RT/choice cleaning (drop non-responses, RT > 150 ms by default).

## CLI fits

For reproducibility, every dataset has a unified fit script:

```bash
# magnitude / risky choice (Barreto-Garcia 2022)
python -m bauer.scripts.fit_garcia rdm --flex --n-subjects all \
    --backend numpyro --out-dir results/garcia

# de Hollander 2024 (dotcloud or symbolic)
python -m bauer.scripts.fit_dehollander2024 ddm --task dotcloud \
    --prior-estimate full --backend numpyro

# de Hollander TMS — with regression on stimulation_condition
python -m bauer.scripts.fit_dehollander_tms ddm --flex --regression \
    --reg-on n1_evidence_sd,n2_evidence_sd
```

For cluster-scale GPU fitting, see `bauer/scripts/slurm_jobs/` (sbatch wrappers, CUDA env build, full production sweep).

## Tutorials

`docs/tutorial/` contains five Jupyter notebooks building up from first principles:

1. **Psychophysical modelling** — NLC model, Weber's law, asymmetric noise, hierarchical fitting
2. **Risky choice** — KLW model, noise-driven risk aversion, format effects
3. **Stake effects** — presentation-order interactions, EU vs KLW vs PMCM
4. **Flexible noise curves** — `FlexibleNoiseComparisonModel`, ELPD model comparison
5. **Hierarchical vs MLE** — split-half reliability

Generate via `python docs/tutorial/make_notebooks.py`.

## Citing the included datasets

If you use a bundled dataset, please cite the original source:

- **Garcia 2022** (`load_garcia2022`): Barreto-Garcia, Pelin, et al. "Cognitive imprecision and small-stakes risk aversion." *Journal of Economic Behaviour & Organization* (2023).
- **de Hollander 2024 dotcloud / symbolic** (`load_dehollander2024_*`): de Hollander, G., et al. "Bayesian inference on parietal magnitude representations shapes risk attitudes." *bioRxiv* preprint (2024).
- **de Hollander TMS** (`load_dehollander_tms_risk`): de Hollander et al., *in prep*.
- **Bedi 2026** (`load_bedi2026`): Bedi et al., *in prep* (orientation→value pilot).

## Citing bauer

See `CITATION.cff`. Brief:

```
de Hollander, G., Renkert, M., Davydova, A., Ruff, C. (2024).
bauer: Bayesian Estimation of Perceptual, Numerical and Risky Choice.
https://github.com/ruffgroup/bauer
```

## Documentation

Full API + concepts: https://ruffgroup.github.io/bauer/

## License

MIT
