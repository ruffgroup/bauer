# Race-diffusion model math (and why it's tricky)

This note spells out the generative math for the racing diffusion (RDM) and
related models in `bauer`, and why the principled β² noise formulation creates
identifiability problems that the static cumnorm and DDM don't have.

## 1. Common cognitive front-end (Khaw, Li & Woodford 2020)

For each stimulus $n_k$ on a trial, the agent has a noisy internal
representation:

$$r_k \mid n_k \sim \mathcal{N}(\log n_k,\ \nu_k^2)$$

with a log-normal prior:

$$\log n \sim \mathcal{N}(\mu_p,\ \sigma_p^2)$$

The Bayesian posterior over $\log n_k$ given $r_k$ is Gaussian:

$$\log n_k \mid r_k \sim \mathcal{N}\!\big(\beta_k r_k + (1-\beta_k)\mu_p,\ \beta_k \nu_k^2\big),
\quad
\beta_k = \frac{\sigma_p^2}{\sigma_p^2 + \nu_k^2}$$

Three quantities are useful downstream:

| quantity | formula | meaning |
|---|---|---|
| posterior mean | $\mu_{\text{post},k}(r_k) = \beta_k r_k + (1-\beta_k)\mu_p$ | best estimate of $\log n_k$ given one $r_k$ |
| posterior width | $\sigma_{\text{post},k}^2 = \beta_k \nu_k^2$ | agent's uncertainty about $\log n_k$ given $r_k$ |
| variance of posterior mean given $n$ | $\mathrm{Var}[\mu_{\text{post},k} \mid n_k] = \beta_k^2 \nu_k^2$ | trial-to-trial spread of $\mu_{\text{post},k}$ across hypothetical replications |

Note that the three differ — encoding noise $\nu_k$, posterior width
$\sigma_{\text{post}}$, and variance-of-posterior-mean $\beta\nu$ are
algebraically related but conceptually distinct.

## 2. Static psychometric (`MagnitudeComparisonModel`)

The decision rule is "choose 2 if $\mu_{\text{post},2} > \mu_{\text{post},1}$",
applied once at the end of the trial. The choice probability over many
replications of the same $(n_1, n_2)$ is

$$P(\text{choose 2}) = \Phi\!\left(\frac{\mathrm{E}[\mu_{\text{post},2}] - \mathrm{E}[\mu_{\text{post},1}]}{\sqrt{\mathrm{Var}[\mu_{\text{post},1}] + \mathrm{Var}[\mu_{\text{post},2}]}}\right)$$

Substituting:

$$P(\text{choose 2}) = \Phi\!\left(\frac{\beta_2 \log n_2 - \beta_1 \log n_1 + (\beta_1 - \beta_2)\mu_p}{\sqrt{\beta_1^2\nu_1^2 + \beta_2^2\nu_2^2}}\right)$$

In `bauer`'s implementation, the variance is computed from $\nu_k^2$ directly
(not $\beta_k^2\nu_k^2$); both work and just shift the SNR scale, but
strictly only the latter matches "variance of the posterior mean".

**Limits:**

- $\sigma_p \to 0$ (tight prior): $\beta_k \to 0$, both $\mathrm{Var}[\mu_{\text{post},k}]\to 0$, ratio is $0/0$. Posterior means collapse to $\mu_p$, denominator collapses too — limit is well-defined and gives $P \to 1/2$ (the agent is overruled by the prior).
- $\sigma_p \to \infty$ (flat prior): $\beta_k \to 1$, recovers the Khaw-Woodford eq (1.2): $P = \Phi(\log(n_2/n_1)/\sqrt{\nu_1^2+\nu_2^2})$.
- Symmetric noise ($\nu_1 = \nu_2 = \nu$): $\beta_1 = \beta_2$, prior cancels — $P$ depends only on $\log(n_2/n_1)$, scale-invariant.

The order effect ($\nu_1 \ne \nu_2$ from memory degradation) breaks the
prior-cancellation, so the prior matters precisely *because* the noise is
asymmetric.

## 3. Drift-diffusion model (DDM)

Same cognitive front-end. The agent integrates a single noisy difference
signal over time; the first hitting boundary determines choice and RT.
Drift is the SNR of the perceived difference:

$$v = \frac{\mu_{\text{post},2} - \mu_{\text{post},1}}{\sqrt{\nu_1^2 + \nu_2^2}}$$

Diffusion noise is fixed at $\sigma^2 = 1$ (HSSM convention, full boundary $2a$).

Choice probability and RT distribution come from the Wiener first-passage-time
likelihood (analytical via Navarro & Fuss 2009).

**Where σ_p enters:** through the *drift* ($\beta_k$ mixing in the numerator).
Diffusion noise $\sigma$ is fixed and contains no σ_p.

## 4. Racing-diffusion model (RDM)

Each stimulus drives its own Wiener accumulator. First to hit threshold $a$
wins. Per-accumulator first-passage time is inverse-Gaussian:

$$T_k \sim \mathrm{IG}\!\left(\mu = \frac{a}{v_k},\ \lambda = \frac{a^2}{\sigma_k^2}\right)$$

Likelihood for "$k$ wins at time $t$" is

$$\mathcal{L} = f_{\mathrm{IG}}(t;\mu_w,\lambda_w) \prod_{j \ne w} S_{\mathrm{IG}}(t;\mu_j,\lambda_j)$$

where $f_{\mathrm{IG}}$ is the IG pdf and $S_{\mathrm{IG}} = 1 - F_{\mathrm{IG}}$
the survival.

### 4a. Drift

$$v_k = \mu_{\text{post},k} = \beta_k \log n_k + (1-\beta_k)\mu_p$$

(uncontroversial — same as in the static model).

### 4b. Diffusion noise — the principled β² formulation

We want σ_k to reflect the trial-to-trial variability of the agent's drift
estimate at fixed $n_k$:

$$\sigma_k^2 = \mathrm{Var}[\mu_{\text{post},k} \mid n_k] = \beta_k^2 \nu_k^2$$

i.e. $\sigma_k = \beta_k\nu_k$.

**Limits:**

- $\sigma_p \to 0$ (tight prior): $\beta_k \to 0$, so $v_k \to \mu_p$ for both accumulators **and** $\sigma_k \to 0$. The race is deterministic at rate $\mu_p$; both accumulators hit boundary at exactly the same moment. Choices are 50/50 by symmetry, RT is fixed at $t_0 + a/\mu_p$. **Consistent with static cumnorm in this limit.**
- $\sigma_p \to \infty$ (flat prior): $\beta_k \to 1$, $v_k \to \log n_k$, $\sigma_k \to \nu_k$. Pure Stevens-style race, drift = log magnitude, noise = encoding SD.
- Symmetric noise: $\beta_1 = \beta_2$, both accumulators have identical $\sigma$ but drifts still differ via $\log n_k$. Larger numbers → faster RTs (size effect) regardless of which one wins.

### 4c. Diffusion noise — the simple alternative

Drop the β² and use $\sigma_k = \nu_k$ directly. Interpretable as
"per-unit-time variance of the accumulator's continuous evidence stream".

**Limit pathology:** $\sigma_p \to 0$ now gives $v_k \to \mu_p$ (drifts
identical) but $\sigma_k = \nu_k$ unchanged. Asymmetric $\nu_1 \ne \nu_2$
biases the race toward the noisier accumulator (fatter left tail of IG),
**so choices are not 50/50** even when the prior dominates. This is
counter to the Bayesian-observer story, where a sharp prior should make
the agent's choices uninformative by symmetry.

So the β² formulation is principled; the simple one isn't quite.

## 5. Why the RDM is harder to fit than the DDM / cumnorm

In the cumnorm and DDM, σ_p enters only through the drift mean (via the
β_k weighting). The diffusion / SNR denominator is independent of σ_p.
**Two parameters, two roles.**

In the RDM with β² noise, σ_p enters drift **and** noise simultaneously:

- drift: $v_k = \beta_k \log n_k + (1-\beta_k)\mu_p$ — σ_p in $\beta_k$
- noise: $\sigma_k = \beta_k \nu_k$ — σ_p **multiplies** ν_k via β_k

This creates a multiplicative coupling between σ_p and ν_k. Two posterior
ridges:

- (large σ_p, small ν_k) → β ≈ 1, σ_k ≈ ν_k (small), drift ≈ log n_k
- (small σ_p, large ν_k) → β ≈ 0, σ_k ≈ 0, drift ≈ μ_p

Both regions partially fit the data; NUTS sees a curved/ridge-shaped
posterior and mixes terribly when both σ_p and ν_k are per-subject
hierarchical with limited data.

In the DDM, the equivalent ridge doesn't exist because diffusion noise is
fixed (σ²=1). σ_p only shifts drift center.

## 6. Identifiability fixes (in priority order)

1. **Group-level σ_p** — shared across subjects, still inferred. The prior
   is a population-level construct in the Khaw-Woodford framework anyway.
   Drastically reduces parameter count and removes per-subject ridge.
2. **Reparameterize** as $(\beta_k, \sigma_k^{\text{eff}}, \mu_p)$ with
   $\nu_k = \sigma_k^{\text{eff}}/\beta_k$ and σ_p derived. Same model,
   smoother sampling geometry.
3. **Stronger / informative prior** on σ_p (e.g. centered at empirical
   $\mathrm{std}(\log n)$). Regularization, no model change.

Empirically, fit_prior=True on 6 subjects × 200 trials gives r̂ > 1.8 and
ESS < 5. The same setup with σ_p fixed (data-derived) had only 60 divergences
and r̂ ~ 1.01.

## 7. Comparison table

| | drift uses σ_p? | noise uses σ_p? | tight-prior limit |
|---|---|---|---|
| static cumnorm | yes (mean shift) | no | P = 0.5 |
| DDM | yes (mean shift) | no (σ=1 fixed) | P = 0.5, RT = t_0 |
| RDM (σ_k = β_k ν_k) | yes | **yes** | P = 0.5, deterministic RT = t_0 + a/μ_p |
| RDM (σ_k = ν_k) | yes | no | P ≠ 0.5 (biased to noisier accumulator) |

The third row's "yes" in the noise column is the source of the
identifiability headache.
