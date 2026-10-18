# Inconsistency: static-choice vs DDM/RDM decision-noise normalization

**Status (2026-10): addressed behind an opt-in flag** --
`BaseModel.consistent_choice_noise = True` normalises the static-choice
likelihood by `posterior_mean_sd`; the default `False` keeps the historical
behaviour. Whether to flip the default is still open.

**Original status (2026-06): needs a decision from the model owner.** Flagged 2026-06 while fitting
the dyscalculia magnitude-comparison data, where the *static choice* model
attributed the group deficit to **memory** noise but the **DDM/RDM** attributed
it to **perceptual** noise. The split turns out to be (at least partly) an
artifact of how the two likelihoods normalize the decision variable, not a
genuine choice-vs-RT information difference.

## The two formulas

Both families build the decision **signal** from the *shrunken* Bayesian
posterior means, `μ̂₂ − μ̂₁`. They differ in the **noise** they divide by:

- **Static choice** — `BaseModel._get_choice_predictions` (`core.py:148`):
  ```python
  diff_mu, diff_sd = get_diff_dist(post_n2_mu, n2_evidence_sd,
                                   post_n1_mu, n1_evidence_sd)
  # get_diff_dist -> diff_sd = sqrt(n1_evidence_sd**2 + n2_evidence_sd**2)
  ```
  i.e. **raw, un-shrunken evidence SD**  `√(σ_e1² + σ_e2²)`.

- **DDM / RDM** — `_drift_from_snr` (`ddm.py:534`), `_drifts_from_post_and_prior`:
  ```python
  sigma_i = posterior_mean_sd(prior_sd, evidence_sd)   # = w_i · σ_e,  w = σp²/(σp²+σe²)
  diff_sd = sqrt(sigma1**2 + sigma2**2)                # = sqrt(w1²σ1² + w2²σ2²)
  ```
  i.e. **shrunken posterior-mean SD** (code comment: "the principled DDM
  diffusion noise").

## What KLW (Khaw, Li & Woodford 2021) says is correct

KLW eq (2.1): `r ~ N(log V, ν²)`; decision compares posterior means
`E[V|r] = exp(α + β r)`, `β = σ²/(σ²+ν²)`; choice prob eq (2.5):

    P = Φ( (log(X/C) − β⁻¹ log p⁻¹) / (√2 ν) ).

The decision variable is the **noisy posterior-mean point estimate** `μ̂ = β·r`
(we "sample posterior means, not the full posterior"). Its across-trial SD is
`β·ν`. The same choice prob can be written equivalently as
**(true signal)/(raw ν)** [eq 2.5] OR **(posterior-mean signal)/(β·ν)** — `β`
scales signal and noise together. Both correct.

- bauer **DDM**: (posterior-mean signal)/(posterior-mean SD `=w·σ_e=β·ν`) →
  **matches KLW**.
- bauer **static**: (posterior-mean signal)/(**raw** `σ_e`) → shrinks the signal
  by `β` but not the noise ⇒ psychometric a factor `β` too steep, and since `β`
  depends on the noise level, the distortion is noise-dependent → warps the
  compression and the perceptual-vs-memory attribution. **This is the bug.**

## Why the static version is the inconsistent one

The static model **shrinks the signal** (uses `post_*_mu`) but uses
**un-shrunken noise** (`*_evidence_sd`). For the generative choice probability
`P(μ̂₂ > μ̂₁)`, the across-trial variance of the decision variable is
`Var(μ̂₂ − μ̂₁) = w₂²σ₂² + w₁²σ₁²` (because `μ̂_i = w_i x_i + (1−w_i)μ_p`, so
`SD[μ̂_i | V] = w_i σ_e,i`). That is exactly what the DDM uses. So the DDM
normalization is correct and the **static model's `evidence_sd` normalization is
a bug** (signal and noise treated with inconsistent shrinkage).

## Consequence for inference

Because the common (perceptual) noise term enters the two normalizers
differently, the *same* group change in perceptual vs memory noise produces
**different compression / choice-proportion behaviour** in the two models —
which is why a perceptual-only contrast reproduces the n1-compression group gap
in the DDM (~60%) but not in the static model (~0%). Any cross-family
comparison of *where* a group effect sits (perceptual vs memory) is therefore
confounded by this until the normalizers are made consistent.

## Fix + test

Make `_get_choice_predictions` normalize by `posterior_mean_sd` too:
```python
sd1 = posterior_mean_sd(model_inputs['n1_prior_sd'], model_inputs['n1_evidence_sd'])
sd2 = posterior_mean_sd(model_inputs['n2_prior_sd'], model_inputs['n2_evidence_sd'])
diff_mu, diff_sd = post_n2_mu - post_n1_mu, pt.sqrt(sd1**2 + sd2**2)
```
(Under a flat prior `w→1`, this reduces to the current evidence-SD formula, so
flat-prior fits are unaffected.) Then refit a choice perceptual-only vs
memory-only pair and confirm the perceptual-vs-memory discrepancy with the DDM
**collapses**. If it does, the historical "choice model says memory" results
were driven by this normalization, not the data — re-run before interpreting.

**Do not silently change this** — it alters every non-flat-prior static-choice
fit in past analyses. Decide with the model owner whether to fix-and-refit or
keep for backward-compat behind a flag.
