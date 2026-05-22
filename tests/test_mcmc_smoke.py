"""End-to-end MCMC smoke tests.

These tests run a *real* (but very short — draws=50, tune=50, chains=2)
NUTS sample for one representative model per family. They guard against
regressions in the integration between:
    - the cognitive front-end (prior, posterior combination)
    - PyMC's NUTS sampler
    - the predict() / simulate() / ppc() helpers

They are not statistical correctness tests — they cannot verify posterior
recovery — but they catch wiring/shape/return-type breakage that the
build-only smoke tests miss.

Marked ``slow`` for opt-out via ``pytest -m 'not slow'`` if the user wants
the fast feedback loop. CI runs the whole suite.
"""
import pytest
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Common short-sample kwargs
# ---------------------------------------------------------------------------

SAMPLE_KWARGS = dict(
    draws=50, tune=50, chains=2, cores=1,
    progressbar=False, random_seed=42,
    compute_convergence_checks=False,
)


# ---------------------------------------------------------------------------
# Fixtures — small but well-behaved synthetic paradigms
# ---------------------------------------------------------------------------

@pytest.fixture
def paradigm_magnitude():
    """4 subjects × 24 trials of magnitude comparison.

    True choice probability is monotone in n2/n1, so the noise parameter
    has a well-defined posterior even with tiny n.
    """
    rng = np.random.default_rng(0)
    n_subj, n_trials_per_subj = 4, 24
    n1_set = np.array([5, 7, 10, 14, 20, 28], dtype=float)
    n2_set = np.array([4, 6, 9, 13, 22, 30], dtype=float)

    rows = []
    for s in range(1, n_subj + 1):
        for t in range(n_trials_per_subj):
            n1 = float(rng.choice(n1_set))
            n2 = float(rng.choice(n2_set))
            # Choose option 2 (= True) more often when n2 > n1
            p = 1 / (1 + np.exp(-(np.log(n2) - np.log(n1)) / 0.3))
            rows.append({
                'subject': s, 'run': 1, 'trial_nr': t,
                'n1': n1, 'n2': n2,
                'choice': bool(rng.random() < p),
                'rt': 0.4 + rng.gamma(2.0, 0.15),
            })
    return pd.DataFrame(rows).set_index(['subject', 'run', 'trial_nr'])


@pytest.fixture
def paradigm_risk():
    """4 subjects × 24 trials of risky-vs-safe gambles."""
    rng = np.random.default_rng(1)
    n_subj, n_trials_per_subj = 4, 24
    rows = []
    for s in range(1, n_subj + 1):
        for t in range(n_trials_per_subj):
            risky_first = bool(t % 2)
            n_risky = float(rng.choice([10, 15, 20, 25, 30]))
            n_safe = float(rng.choice([10, 12, 15, 18, 20]))
            p_risky = float(rng.choice([0.35, 0.45, 0.55, 0.65]))
            if risky_first:
                n1, p1, n2, p2 = n_risky, p_risky, n_safe, 1.0
            else:
                n1, p1, n2, p2 = n_safe, 1.0, n_risky, p_risky
            rows.append({
                'subject': s, 'run': 1, 'trial_nr': t,
                'n1': n1, 'n2': n2, 'p1': p1, 'p2': p2,
                'choice': bool(rng.random() > 0.5),
                'rt': 0.4 + rng.gamma(2.0, 0.15),
            })
    return pd.DataFrame(rows).set_index(['subject', 'run', 'trial_nr'])


# ---------------------------------------------------------------------------
# MagnitudeComparisonModel: sample, predict, ppc
# ---------------------------------------------------------------------------

def test_magnitude_model_sample_predict_ppc(paradigm_magnitude):
    from bauer.models import MagnitudeComparisonModel
    m = MagnitudeComparisonModel(paradigm=paradigm_magnitude,
                                  fit_separate_evidence_sd=True, fit_prior=True)
    m.build_estimation_model(data=paradigm_magnitude, hierarchical=True)
    idata = m.sample(**SAMPLE_KWARGS)

    # idata shape
    assert idata.posterior.sizes['chain'] == 2
    assert idata.posterior.sizes['draw'] == 50
    # Every free parameter has subject-level draws
    for p in m.free_parameters:
        assert p in idata.posterior, f"Posterior is missing {p}"
        assert 'subject' in idata.posterior[p].dims

    # predict() returns one row per trial with a probability in (0,1)
    pars = m.get_subjectwise_parameter_estimates(idata=idata).groupby('subject').mean()
    pred = m.predict(paradigm_magnitude, pars)
    assert len(pred) == len(paradigm_magnitude)
    assert 'p_choice' in pred.columns
    p_choice = pred['p_choice'].values
    assert np.all((p_choice > 0) & (p_choice < 1))

    # ppc() returns simulated choices with the documented index
    ppc = m.ppc(paradigm_magnitude, idata, n_posterior_samples=10,
                random_seed=0, progressbar=False)
    assert 'simulated_choice' in ppc.columns
    assert ppc['simulated_choice'].dtype == bool
    assert 'ppc_sample' in ppc.index.names


# ---------------------------------------------------------------------------
# RiskModel: sample only — confirms risky/safe + prior wiring runs end-to-end
# ---------------------------------------------------------------------------

def test_risk_model_sample(paradigm_risk):
    from bauer.models import RiskModel
    # RiskModel still accepts only the historical-spelling kwarg
    m = RiskModel(paradigm=paradigm_risk, prior_estimate='shared',
                   fit_seperate_evidence_sd=True)
    m.build_estimation_model(data=paradigm_risk, hierarchical=True)
    idata = m.sample(**SAMPLE_KWARGS)
    assert {'prior_mu', 'prior_sd'}.issubset(idata.posterior.data_vars)


# ---------------------------------------------------------------------------
# DDM model: sample, then exercise ppc (requires hssm)
# ---------------------------------------------------------------------------

def test_ddm_magnitude_model_sample_and_ppc(paradigm_magnitude):
    pytest.importorskip('hssm')
    from bauer.models import DDMMagnitudeComparisonModel

    m = DDMMagnitudeComparisonModel(paradigm=paradigm_magnitude,
                                      fit_separate_evidence_sd=True,
                                      fit_prior=True, fit_v_scale=True)
    try:
        m.build_estimation_model(data=paradigm_magnitude, hierarchical=True)
    except TypeError:
        m.build_estimation_model(paradigm=paradigm_magnitude, hierarchical=True)

    # Reduce target_accept slightly for the very short warmup; we don't care
    # about diagnostic-quality output here, only that the pipeline runs.
    idata = m.sample(target_accept=0.8, **SAMPLE_KWARGS)
    for required in ('a', 't0'):
        assert required in idata.posterior.data_vars

    # DDM ppc returns choice AND rt
    ppc = m.ppc(paradigm_magnitude, idata, n_posterior_samples=4,
                random_seed=0, progressbar=False)
    assert 'simulated_choice' in ppc.columns
    assert 'simulated_rt' in ppc.columns


# ---------------------------------------------------------------------------
# Race-diffusion model: sample (no rt-PPC, that exercises Wald sampling)
# ---------------------------------------------------------------------------

def test_race_diffusion_magnitude_model_sample(paradigm_magnitude):
    from bauer.models import RaceDiffusionMagnitudeComparisonModel
    m = RaceDiffusionMagnitudeComparisonModel(
        paradigm=paradigm_magnitude, fit_seperate_evidence_sd=True,
        fit_prior=True,
    )
    try:
        m.build_estimation_model(data=paradigm_magnitude, hierarchical=True)
    except TypeError:
        m.build_estimation_model(paradigm=paradigm_magnitude, hierarchical=True)
    idata = m.sample(target_accept=0.8, **SAMPLE_KWARGS)
    assert 'a' in idata.posterior.data_vars
    assert 't0' in idata.posterior.data_vars
