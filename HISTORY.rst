=======
History
=======

0.3.0 (2026-06-10)
------------------

.. warning::

   **Breaking changes since 0.2.0** — read before re-running old analyses:

   * **Group-level SD prior default changed** ``halfcauchy`` → ``halfnormal``
     (``BaseModel.group_sd_dist``). This silently changes the posterior of
     **every hierarchical regression fit**. HalfCauchy's infinite-variance
     tail let poorly-identified group SDs run away (funnels, divergences);
     HalfNormal's light tail tames them. To reproduce pre-0.3.0 numbers,
     set ``model.group_sd_dist = 'halfcauchy'`` per instance.
   * **The** ``regressors=`` **keyword is deprecated** in favour of the
     explicit ``fixed_regressors=`` (population-mean design, no per-subject
     offset) and ``random_regressors=`` (per-subject random effect) split.
     ``regressors=`` still works bit-for-bit (it maps to a random slope on
     every column) but now emits a ``DeprecationWarning``.
   * **Default lapse group prior changed** to Beta "1/x"
     (``p_lapse`` / ``p_outlier`` transform ``'beta'``); the old logit-Normal
     prior is opt-in via ``model.lapse_group = 'logit_normal'``.

**Sampling robustness and convergence** (what makes the DDM/RDM models above
actually mix on real, multi-subject data):

* **RT outlier contaminant** (``p_outlier``) baked into every DDM and
  race-diffusion model (HSSM-style mixture). On a fraction ``p_outlier`` of
  trials the response is a flat-over-RT lapse rather than the diffusion
  process. **Fixed** ``p_outlier = 0.05`` by default (set ``0.0`` for pure
  WFPT, ``'hierarchical'`` to estimate it). This is what cracks the
  66-subject DDM: the slow-RT tail otherwise breaks convergence, inflates
  Pareto-k (unreliable LOO), and shows up as a PPC tail misfit — one cause,
  three symptoms, all fixed (r̂ 1.59 → 1.00, min ESS 7 → 3200+).
* **Configurable group-SD prior** ``group_sd_dist`` (``'halfnormal'`` |
  ``'halfcauchy'``); see the breaking note above.
* **Opt-in multipath Pathfinder init** (``find_init='pathfinder'``,
  MAP-seeded) for genuinely multimodal posteriors; ``'mapjitter'`` remains
  the default and is usually sufficient once the contaminant is in.
* New guide ``notes/fitting_ddm_without_divergences.md`` in the source tree —
  the full recipe and a symptom → diagnosis → fix catalogue for DDM/RDM
  convergence.

**Fixed vs random effects API**

* ``fixed_regressors`` / ``random_regressors`` on the regression models
  (static choice and the DDM magnitude-comparison regression model), letting
  you put a population-mean (fixed) effect on a between-subjects covariate
  while keeping a per-subject random intercept — the correct parameterisation
  for group contrasts (a random slope on a between-subjects covariate is
  non-identified; bauer now warns). New tutorial: *Fixed vs random effects*.

**Other modelling additions**

* ``memory_as_sv`` DDM variant: routes the frozen memory noise into Ratcliff
  across-trial drift variability ``sv`` (perceptual noise stays within-trial),
  making the two noise sources separately identifiable. Composes with the
  contaminant.
* Race-diffusion ``fit_w_d`` / ``fit_w_s`` toggles to fix the discriminative
  gain (w_d ≡ 1) or ablate the overall-magnitude (sum) drift term.
* ``consistent_choice_noise`` for static choice models: normalise the choice
  likelihood by the SD of the noisy posterior mean (KLW/DDM-consistent),
  matching how the accumulator models treat decision noise.

The original 0.3.0 feature set (unreleased until now):

* **DDM and race-diffusion models** for choice + RT data, sharing the same
  Bayesian observer cognitive front-end as the static-choice models:

  * ``DDMMagnitudeComparisonModel``, ``DDMFlexibleNoiseComparisonModel``,
    ``DDMRiskModel``, ``DDMFlexibleNoiseRiskModel`` (+ regression variant)
    — Wiener WFPT likelihood via ``hssm.likelihoods.logp_ddm``.
  * ``RaceDiffusionMagnitudeComparisonModel``,
    ``RaceDiffusionFlexibleNoiseComparisonModel``,
    ``RaceDiffusionRiskModel``, ``RaceDiffusionFlexibleNoiseRiskModel``
    (+ regression variant) — analytical Wald-race likelihood with
    ``advantage=True`` decomposition by default
    (van Ravenzwaaij 2020 style).
* **JAX backend** support via ``--backend {numpyro,blackjax}`` in the
  CLI fit scripts. JAX-NUTS is 1.5–3× faster on CPU and 5–30× faster on
  GPU (NVIDIA L4, with ``chain_method='vectorized'`` so chains run in
  parallel on a single device).
* **CLI fit scripts** under ``bauer/scripts/``:
  ``fit_garcia.py``, ``fit_dehollander2024.py``, ``fit_dehollander_tms.py``.
* **SLURM job templates** under ``bauer/scripts/slurm_jobs/`` for cluster
  fitting, including a CUDA-env build job, a generic ``run_fit.sh``
  wrapper, a JAX backend benchmark, and a full per-dataset production
  submit script.
* **Bundled datasets** added (loaders in ``bauer.utils.data``):
  ``load_dehollander2024_risk`` (dotcloud, N=30),
  ``load_dehollander2024_symbolic`` (Arabic-numeral risk, N=58),
  ``load_dehollander_tms_risk`` (TMS risky choice, N=35 sessions 2/3),
  ``load_bedi2026`` (Bedi 2026 abstract-value estimation pilot, N=13).
* **Unified PPC API**: ``BaseModel.ppc`` now returns the same long-format
  DataFrame as the DDM/RDM PPCs — index = paradigm levels + ``ppc_sample``,
  single ``simulated_choice`` column.
* Renamed ``polynomial_order`` → ``spline_order`` throughout (no
  backwards-compat alias).
* Cleanup: removed ``SafeVsRisky*`` family, ``RNPModel``, several
  deprecated ``prior_estimate`` options in ``RiskModel``, and the
  ``incorporate_probability`` parameter. 26 stale dev notebooks deleted;
  bundled CSVs trimmed of derived columns.

0.2.0 (2026-04-03)
------------------

* Refactored model classes into a dedicated ``bauer/models/`` package
  (``psychophysics``, ``magnitude``, ``risky_choice`` submodules).
* Added ``pyproject.toml`` with fully declared runtime and optional dependencies.
* Added ``environment.yml`` for reproducible conda environment setup.
* Added tutorial notebooks covering psychophysics / magnitude comparison,
  risky choice with the KLW model, and de Hollander et al. (2024) stake effects.
* ``load_dehollander2024()`` data loader added to ``bauer.utils.data``.
* Documentation expanded with API reference, concepts page, and tutorial.

0.1.0 (2022-11-17)
------------------

* First release on PyPI.
