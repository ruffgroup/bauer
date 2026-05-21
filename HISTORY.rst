=======
History
=======

0.3.0 (2026-05-06)
------------------

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
