"""Convergence triage utilities for hierarchical DDM/RDM fits.

The common failure on a well-specified hierarchical fit is a single chain stuck
in a pathological region (funnel neck / minor mode) while the rest mix — see
``notes/ddm_convergence_lessons.md`` section 4b. ``trim_stuck_chains``
operationalises the *safe* version of dropping it: a chain is removed only if

  (1) it is isolated from the *agreeing majority* of chains on the group-level
      ``*_mu`` parameters, AND
  (2) that majority is more than half the chains (a tight modal cluster). If
      chains split roughly evenly we refuse to auto-trim — that may be genuine
      multimodality and you must investigate.

NB: do NOT gate on log-density. A stuck chain often sits in a *sharp* minor
mode at the funnel neck and has *higher* ``lp`` than chains correctly sampling
the broad typical set, so "drop the low-lp chain" keeps the wrong one. ``lp`` is
reported as context only. The signal is **disagreement with the majority**
(ideally cross-checked against a re-seed / 8-chain run / a converged superset
model). Selecting chains by ``lp`` or any quality measure biases the posterior —
never do that.

Over-provision chains (``chains=8``+) so a stray chain is a small fraction, then
trim. Always report what was dropped (this returns a report dict).
"""
import numpy as np


def _per_chain_feature_matrix(post):
    """(n_chains, n_features): per-chain posterior mean of each group-level
    ``*_mu`` parameter (Intercept regressor if it's a regression model)."""
    mu_vars = [v for v in post.data_vars
               if v.endswith("_mu") and not v.endswith("_mu_mu")]
    cols = []
    for v in mu_vars:
        da = post[v]
        rd = [d for d in da.dims if d.endswith("_regressors")]
        if rd:
            levels = list(da.coords[rd[0]].values)
            da = da.sel({rd[0]: "Intercept"}) if "Intercept" in levels \
                else da.isel({rd[0]: 0})
        cols.append(da.mean(dim=[d for d in da.dims if d != "chain"]).values)
    return np.asarray(cols).T, mu_vars


def trim_stuck_chains(idata, isolation_z=3.0, verbose=True):
    """Return ``(trimmed_idata, report)`` dropping chains isolated from the
    agreeing majority on the group-level ``*_mu`` parameters.

    A chain is dropped only if its robust z-distance (over standardized
    group-level means) from the chain-median exceeds ``isolation_z`` AND the
    surviving (non-isolated) chains are a strict majority. Nothing is dropped
    otherwise. ``lp`` is reported for context but not used as a criterion.
    """
    post, sstats = idata.posterior, idata.sample_stats
    n = post.sizes["chain"]
    X, _ = _per_chain_feature_matrix(post)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) + 1e-12
    Z = (X - med) / (1.4826 * mad)
    dist = np.sqrt((Z ** 2).mean(axis=1))
    lp = sstats["lp"].mean(dim=[d for d in sstats["lp"].dims
                                if d != "chain"]).values

    isolated = dist > isolation_z
    majority_ok = (~isolated).sum() > n / 2.0
    drop = sorted(int(c) for c in np.where(isolated)[0]) if majority_ok else []
    keep = [c for c in range(n) if c not in drop]

    if verbose:
        print(f"trim_stuck_chains: {n} chains")
        print(f"  feature-distance : {np.round(dist, 2)}")
        print(f"  lp (context only): {np.round(lp, 1)}")
        print(f"  isolated         : {list(np.where(isolated)[0])}  "
              f"(majority agrees: {majority_ok})")
        if drop:
            print(f"  DROPPED {drop} -> kept {keep}")
        else:
            print("  dropped nothing"
                  + ("" if majority_ok else " (no clear majority — investigate, "
                     "may be genuine multimodality)"))

    trimmed = idata.sel(chain=keep) if drop else idata
    return trimmed, {"dropped": drop, "kept": keep, "distance": dist, "lp": lp,
                     "majority_ok": bool(majority_ok)}


def convergence_report(idata, var_names=None, rhat_max=1.01, ess_min=400, max_divergences=0,
                       verbose=True):
    """r̂, bulk ESS and divergences, with a one-line verdict.

    By default checks every group-level variable (``*_mu``, ``*_mu_untransformed``
    and ``*_sd``): the ones that get interpreted and the group SDs, which are
    often the worst-mixing. Pass ``var_names`` to check others (e.g. the
    subject-level parameters). A NaN r̂ (a stuck or constant chain) counts as a
    failure.

    Returns the arviz diagnostics table; ``table.attrs['ok']`` is the verdict.
    """
    import arviz as az
    if var_names is None:
        # Match on the name AND require no subject dimension: subject-level
        # parameters can themselves end in '_sd' or '_mu' (n1_evidence_sd,
        # prior_sd, prior_mu), and would otherwise be swept in.
        var_names = [v for v in idata.posterior.data_vars
                     if v.endswith(('_mu', '_mu_untransformed', '_sd'))
                     and 'subject' not in idata.posterior[v].dims]
        if not var_names:
            raise ValueError('No group-level (*_mu / *_sd) variables found (non-hierarchical '
                             'fit?). Pass var_names=[...].')
    summ = az.summary(idata, var_names=var_names, kind='diagnostics')
    ss = getattr(idata, 'sample_stats', None)
    n_div = int(ss['diverging'].sum()) if ss is not None and 'diverging' in ss else 0
    rhat_nan = summ['r_hat'].isna()
    worst = summ['r_hat'].fillna(np.inf).idxmax()
    ok = (not rhat_nan.any() and summ['r_hat'].max() <= rhat_max
          and summ['ess_bulk'].min() >= ess_min and n_div <= max_divergences)
    summ.attrs['ok'] = bool(ok)
    if verbose:
        msg = (f"Max r̂ = {summ['r_hat'].max():.3f} (want ≤ {rhat_max}; worst: {worst}), "
               f"min bulk ESS = {summ['ess_bulk'].min():.0f} (want ≥ {ess_min}), "
               f"divergences = {n_div} (want ≤ {max_divergences})")
        if rhat_nan.any():
            msg += f", r̂ is NaN for {int(rhat_nan.sum())} variable(s) (stuck chain?)"
        print(msg + f"  →  {'OK' if ok else 'CHECK BEFORE USING'}")
    return summ
