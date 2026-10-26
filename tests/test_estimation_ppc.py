"""Posterior predictive checks for the continuous-response estimation models.

``EstimationBaseModel.ppc`` must simulate replicate DATASETS (response noise,
lapses, discretisation), one per posterior draw, so that a band on any data
statistic is the distribution of that statistic over datasets like the
observed one.  ``posterior_predicted_moments`` gives per-trial expected
responses and is NOT a PPC; test (c) shows how much too narrow a band built
from it is.

All tests use fake posteriors built from known parameters (no MCMC), a small
grid and a few hundred trials, so they run in seconds.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings('ignore')

STIMULI = np.array([10, 14, 18, 22, 26, 30, 34, 38], dtype=float)
TRUE = {1: dict(nu=0.20, sigma_motor=2.0),
        2: dict(nu=0.25, sigma_motor=1.5),
        3: dict(nu=0.15, sigma_motor=2.5)}


def make_paradigm(n_reps=10, subjects=(1, 2, 3)):
    rows = [dict(subject=s, n=n, range='wide', response=n)
            for s in subjects for _ in range(n_reps) for n in STIMULI]
    df = pd.DataFrame(rows)
    trial = df.groupby('subject').cumcount().rename('trial')
    return df.set_index(['subject', trial])


def make_model(paradigm, lapse_rate=0.01):
    from bauer.numerosity import LogEncodingEstimationModel
    return LogEncodingEstimationModel(paradigm, grid_resolution=41,
                                      lapse_rate=lapse_rate)


def fake_idata(true=TRUE, n_chain=2, n_draw=100, spread=0.0, seed=0,
               subject_order=None):
    """Posterior with the given per-subject values (times lognormal jitter of
    s.d. ``spread``), subject coordinate in ``subject_order``."""
    import arviz as az
    import xarray as xr
    rng = np.random.default_rng(seed)
    subjects = list(subject_order or true)
    pars = list(next(iter(true.values())))
    data = {}
    for par in pars:
        centre = np.array([true[s][par] for s in subjects])
        jitter = np.exp(rng.normal(0, spread, size=(n_chain, n_draw, len(subjects))))
        data[par] = (('chain', 'draw', 'subject'), centre * jitter)
    post = xr.Dataset(data, coords=dict(chain=np.arange(n_chain),
                                        draw=np.arange(n_draw),
                                        subject=subjects))
    return az.InferenceData(posterior=post)


def group_curve(df, col):
    """Group mean per stimulus: subject means first, then across subjects."""
    keys = [k for k in ('ppc_draw',) if k in df.index.names]
    m = df.groupby(keys + ['subject', 'n'])[col].mean()
    return m.groupby(keys + ['n']).mean()


# ---------------------------------------------------------------------------
# (a) shape and index
# ---------------------------------------------------------------------------

def test_ppc_shape_and_index():
    df = make_paradigm(n_reps=2)
    m = make_model(df)
    idata = fake_idata(n_chain=2, n_draw=5)
    sim = m.ppc(df, idata, n_draws=7, seed=3)

    assert sim.index.names == ['ppc_draw', 'subject', 'trial']
    assert len(sim) == 7 * len(df)
    assert sorted(sim.index.unique('ppc_draw')) == list(range(7))
    for _, rep in sim.groupby(level='ppc_draw'):
        assert rep.droplevel('ppc_draw').index.equals(df.index)
    assert {'simulated_response', 'chain', 'draw', 'n', 'range'} <= set(sim.columns)
    assert 'response' not in sim.columns          # observed data stays out
    assert sim['simulated_response'].notna().all()
    # (chain, draw) picked jointly, without replacement
    assert not sim.groupby(level='ppc_draw')[['chain', 'draw']].first().duplicated().any()
    # caller's paradigm untouched, and deterministic under the seed
    assert 'response' in df.columns and df.index.names == ['subject', 'trial']
    pd.testing.assert_frame_equal(sim, m.ppc(df, idata, n_draws=7, seed=3))
    # n_draws capped at chains x draws
    assert m.ppc(df, idata, n_draws=50, seed=0).index.unique('ppc_draw').size == 10


def test_subject_alignment_by_coordinate():
    from bauer.estimation import posterior_subject_draws
    idata = fake_idata(subject_order=[3, 1, 2], n_draw=4)
    for _, _, pars in posterior_subject_draws(idata, [1, 2, 3],
                                              ['nu', 'sigma_motor'], 3):
        for s in (1, 2, 3):
            assert pars.loc[s, 'nu'] == pytest.approx(TRUE[s]['nu'])
            assert pars.loc[s, 'sigma_motor'] == pytest.approx(TRUE[s]['sigma_motor'])

    # A paradigm subject the posterior has no parameters for must raise.
    df = make_paradigm(n_reps=1, subjects=(1, 2, 4))
    with pytest.raises(ValueError, match='no parameters'):
        make_model(df).ppc(df, idata, n_draws=2)


def test_response_step_rounds():
    df = make_paradigm(n_reps=2)
    sim = make_model(df).ppc(df, fake_idata(n_draw=5), n_draws=3, response_step=1)
    r = sim['simulated_response'].to_numpy()
    assert np.allclose(r, np.round(r))
    assert r.min() >= 10 and r.max() <= 40


# ---------------------------------------------------------------------------
# (b) calibration: replicate-vs-replicate coverage of the binned-mean band
# ---------------------------------------------------------------------------

def test_ppc_band_is_calibrated():
    df = make_paradigm(n_reps=10)
    m = make_model(df)
    idata = fake_idata(spread=0.05, n_draw=150)
    sim = m.ppc(df, idata, n_draws=300, seed=1)
    curves = group_curve(sim, 'simulated_response').unstack('n').to_numpy()

    # Leave-one-replicate-out: is replicate j inside the 95% band of the rest?
    inside = []
    for j in range(len(curves)):
        rest = np.delete(curves, j, axis=0)
        lo, hi = np.quantile(rest, [.025, .975], axis=0)
        inside.append((curves[j] >= lo) & (curves[j] <= hi))
    coverage = np.mean(inside)
    assert coverage >= .85, coverage

    # A dataset simulated from the true parameters (the "observed" data).
    true_pars = pd.DataFrame(TRUE).T.rename_axis('subject')
    np.random.seed(123)
    obs = m.simulate(df.copy(), true_pars).droplevel('sample')
    obs_curve = group_curve(obs, 'simulated_response').to_numpy()
    lo, hi = np.quantile(curves, [.025, .975], axis=0)
    assert np.mean((obs_curve >= lo) & (obs_curve <= hi)) >= .75


# ---------------------------------------------------------------------------
# (c) the moment band is far too narrow -- the bug the old ppc invited
# ---------------------------------------------------------------------------

def test_ppc_band_wider_than_moment_band():
    df = make_paradigm(n_reps=10)
    m = make_model(df)
    idata = fake_idata(spread=0.05, n_draw=100)
    sim = m.ppc(df, idata, n_draws=200, seed=2)
    mom = m.posterior_predicted_moments(df, idata, n_draws=200, seed=2)

    def width(curve):
        lo, hi = curve.unstack('n').quantile([.025, .975]).to_numpy()
        return hi - lo

    w_ppc = width(group_curve(sim, 'simulated_response'))
    w_mom = width(group_curve(mom, 'predicted_mean'))
    assert np.all(w_ppc > w_mom)
    assert np.median(w_ppc / w_mom) > 2, np.median(w_ppc / w_mom)

    # Replicates from the true model fall outside the moment band far more
    # often than 5% -- it cannot serve as a check of data statistics.
    curves = group_curve(sim, 'simulated_response').unstack('n').to_numpy()
    lo, hi = np.quantile(group_curve(mom, 'predicted_mean').unstack('n').to_numpy(),
                         [.025, .975], axis=0)
    assert np.mean((curves >= lo) & (curves <= hi)) < .6


# ---------------------------------------------------------------------------
# (d) lapses are in the simulated data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('lapse_rate', [0.0, 0.3])
def test_ppc_includes_lapses(lapse_rate):
    df = make_paradigm(n_reps=10)
    m = make_model(df, lapse_rate=lapse_rate)
    sharp = {s: dict(nu=0.03, sigma_motor=0.5) for s in TRUE}
    sim = m.ppc(df, fake_idata(true=sharp, n_draw=20), n_draws=20, seed=4)

    far = (sim['simulated_response'] - sim['n']).abs() > 6
    # Expected share of uniform-[10, 40] lapses landing > 6 from the stimulus.
    n = sim['n'].to_numpy()
    near_len = np.minimum(n + 6, 40) - np.maximum(n - 6, 10)
    expected = lapse_rate * np.mean(1 - near_len / 30)
    assert far.mean() == pytest.approx(expected, abs=.03)
    if lapse_rate == 0:
        assert far.mean() < .005


# ---------------------------------------------------------------------------
# Efficient-coding subclasses
# ---------------------------------------------------------------------------

@pytest.fixture
def paradigm_orientation():
    rng = np.random.default_rng(1)
    rows = [dict(subject=s, orientation=float(o), mapping=mp,
                 response=float(np.clip(2 + o / 180 * 40 + rng.normal(0, 1), 2, 42)))
            for s in (1, 2) for mp in ('linear', 'cdf')
            for o in np.linspace(0, 180, 9, endpoint=False)]
    df = pd.DataFrame(rows)
    return df.set_index(['subject', df.groupby('subject').cumcount().rename('trial')])


@pytest.mark.parametrize('cls, true', [
    ('EfficientPerceptionModel', {1: dict(kappa_r=5.), 2: dict(kappa_r=8.)}),
    ('SequentialEfficientCodingModel', {1: dict(kappa_r=5., sigma_rep=.3),
                                        2: dict(kappa_r=8., sigma_rep=.2)}),
])
def test_ppc_efficient_coding(paradigm_orientation, cls, true):
    import bauer.efficient_coding as ec
    m = getattr(ec, cls)(paradigm=paradigm_orientation, grid_resolution=25)
    assert set(m.free_parameters) == set(true[1])
    sim = m.ppc(paradigm_orientation, fake_idata(true=true, n_draw=5),
                n_draws=4, seed=0)
    assert len(sim) == 4 * len(paradigm_orientation)
    assert np.isfinite(sim['simulated_response']).all()
    mom = m.posterior_predicted_moments(paradigm_orientation,
                                        fake_idata(true=true, n_draw=5), n_draws=2)
    # The compiled-once fast path matches ``predict`` exactly.
    pars = pd.DataFrame(true).T.rename_axis('subject')
    pred = m.predict(paradigm_orientation.copy(), pars)
    np.testing.assert_allclose(mom.xs(0, level='ppc_draw')['predicted_mean'],
                               pred['predicted_mean'], rtol=1e-6)
