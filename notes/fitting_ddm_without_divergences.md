# Fitting a DDM (and RDM) magnitude-comparison model without divergences and bad mixing

This is the document I wish I'd had before fighting the 66-subject dyscalculia
DDM fits. It collects the hard-won lessons into one place: the recipe that just
works, the failure modes and their fixes (each with the actual r̂ / ESS numbers),
how to diagnose convergence, and why the DDM and the RDM need *different* fixes.

Everything here is verified against bauer ≥ 0.3.0 (`bauer.models.ddm`,
`bauer.models.race`, `bauer.core`). Class names, kwargs and defaults are real and
post-0.3.0. The contaminant (`p_outlier`) is now baked into the **base** DDM/RDM
models — the old `*Lapse*` DDM/RDM classes are gone.

---

## 1. TL;DR recipe — the thing that just works

The single configuration that takes the 66-subject magnitude-comparison DDM from
"never mixes" to "r̂ 1.00, 0 divergences, min ESS ~4264 in ~51 min on an L4":

```python
import arviz as az
from bauer.models import DDMMagnitudeComparisonRegressionModel

# 0. DATA: choice is bool (True = option 2 chosen), rt in seconds,
#    `subject` in the index or as a column. Paradigm cols: n1, n2 (magnitude).
df = df[df['rt'] >= 0.20].copy()          # (1) drop fast RTs: flat WFPT gradient

# 1. MODEL — contaminant is ON by default (p_outlier = 0.05, fixed).
m = DDMMagnitudeComparisonRegressionModel(
        paradigm=df,
        regressors={'a': 'group', 'n1_evidence_sd': 'group',
                    'n2_evidence_sd': 'group'},   # only params you expect to move
        memory_model='shared_perceptual_noise',
        fit_v_scale=False,                # (2) ALWAYS: avoids v_scale<->evidence_sd
)
# defaults already set, but be explicit about the ones that matter:
m.p_outlier      = 0.05                   # (3) fixed contaminant (NOT 'hierarchical')
m.group_sd_dist  = 'halfnormal'           # (4) light-tailed group-SD prior (0.3.0 default)

# 2. BUILD
m.build_estimation_model(data=df, hierarchical=True)

# 3. SAMPLE — numpyro + vectorized on a GPU; pass draws/tune BY KEYWORD.
idata = m.sample(
        draws=1000, tune=2000,            # (5) keyword! a classic bug swaps them
        chains=4,
        target_accept=0.99,               # (6) 0.99 to converge; relax to ~0.9 later
        backend='numpyro',                # (7) NOT cpu pymc NUTS
        chain_method='vectorized',        #     all chains in one GPU kernel
        find_init='mapjitter',            # (8) the DDM default; dispersed MAP starts
)

# 4. CHECK before interpreting anything (see §3).
print(az.summary(idata, var_names=['~_offset'], filter_vars='like')
        [['r_hat', 'ess_bulk']].agg(['max', 'min']))
print("divergences:", int(idata.sample_stats['diverging'].sum()))
```

The eight load-bearing choices, in one line each:

1. **`rt >= 0.20 s` filter** — keeps NUTS out of the flat zero-gradient WFPT region.
2. **`fit_v_scale=False`** — fixes `v_scale = 1`; freeing it = perfect
   `v_scale ↔ evidence_sd` degeneracy. (This is the DDM default already.)
3. **fixed `p_outlier = 0.05`** — the HSSM-style RT contaminant. The single biggest
   lever. *Not* `'hierarchical'` (that diverges 4000/4000 on the DDM).
4. **`group_sd_dist = 'halfnormal'`** — light-tailed group-SD prior (the 0.3.0
   default). Decisive for the **RDM**; harmless-to-helpful for the DDM.
5. **`draws` / `tune` by keyword** — `sample(draws=1000, tune=2000, ...)`. Passing
   them positionally in the wrong order silently under-warms the chains.
6. **`target_accept = 0.99`** to drive divergences to 0 for the final fit; you can
   drop toward 0.9 for ~2.3× speed once you trust the geometry.
7. **`backend='numpyro'`** — note `sample()` defaults to `backend='pymc'`, so you
   *must* pass this. With `chain_method='vectorized'` all chains run in one GPU
   kernel. CPU pymc NUTS is the seed-lottery trap (§2.1).
8. **`find_init='mapjitter'`** — data-informed dispersed starting points (the DDM
   default; you don't have to pass it, but know it's on). Pathfinder is *not*
   needed once the contaminant is in.

> **RDM:** swap the class for `RaceDiffusionMagnitudeComparisonModel`
> (`RaceDiffusionMagnitudeComparisonRegressionModel` for covariates). Everything
> else is identical, but `group_sd_dist='halfnormal'` is **not optional** for the
> RDM — see §2.6 and §4.

---

## 2. The failure modes and their fixes

Each as: **symptom → diagnosis → fix**, with the numbers.

### 2.1 CPU pymc NUTS seed lottery → numpyro vectorized on GPU

**Symptom.** A hierarchical DDM "converges" on some random seeds and not others.
You re-run with a new `random_seed`, it works, you ship it — then it fails on the
next dataset. Wall-clock is also brutal: CPU NUTS on a 66-subject DDM is hours.

**Diagnosis.** Two separate problems wearing one coat. (a) The DDM posterior is a
long curved ridge; with a generic init *where the chains start* determines whether
they find the typical set — pure chance per seed. (b) CPU pymc NUTS samples chains
sequentially and slowly, so you can't afford the chains/tune that would paper over
(a). The real cure for (a) is the model fixes below; the cure for (b) is the
sampler.

**Fix.** `backend='numpyro'`, `chain_method='vectorized'`, on a GPU (an L4 is
plenty). Vectorized = all chains in a single JIT-compiled kernel. The 66-subject
DDM goes from "hours, unreliable" to **~51 min, reliable**. Caveat: `sample()`
defaults to `backend='pymc'` — you must pass `backend='numpyro'` explicitly.
Avoid `blackjax` (it fights HSSM's GPU progress bar). `pm.Slice` is a slow
gradient-free robustness check, never a default.

> The seed-lottery never fully closes by sampler choice alone — it closes when you
> add the contaminant (§2.5) and a good init (§2.3). numpyro+GPU is what makes the
> *correct* model affordable.

### 2.2 Swapped `draws` / `tune`

**Symptom.** Mysteriously under-warmed chains: r̂ a bit high, ESS low, divergences
that "shouldn't" be there given everything else looks right.

**Diagnosis.** `sample(1000, 2000, ...)` is **not** `sample(draws=1000,
tune=2000)`. The positional order is `sample(draws, tune, target_accept, chains,
...)`. If you pass `sample(2000, 1000)` thinking "2000 tune, 1000 draws" you've
actually asked for 2000 draws and **1000 tune** — half the warmup you intended. On
a DDM ridge, warmup is where the mass matrix is learned; halving it hurts.

**Fix.** Always pass `draws=` and `tune=` **by keyword**. It costs nothing and
removes an entire class of "why won't this mix" afternoons.

### 2.3 No starting-point finder → mapjitter

**Symptom.** Without an explicit init, a regression-DDM converges on roughly
**1 in 7 seeds**. With the generic backend jitter, chains scatter; some land in the
WFPT flat/−inf zone and never recover.

**Diagnosis.** A DDM/regression posterior is a long, curved ridge. The backend's
default "jitter around the origin" init is blind to the data, so most chains start
off the ridge. r̂ is then meaningless (chains in different basins).

**Fix.** `find_init='mapjitter'` — the DDM/RDM **default** (`recommended_init =
'mapjitter'` on `DDMMixin` and `RaceMixin`). It seeds each chain at the MAP centre
(`pm.find_MAP`, falling back to the prior-central point) and **disperses the chains
by a fraction of each parameter's prior SD**, so they sit *around* the typical set
rather than all at the mode (which would make r̂ falsely optimistic). It covers
every parameter automatically — `a`, `t0`, `prior_mu/sd`, `evidence_sd`, B-spline
coefficients, `p_outlier`. This alone turns the ~1/7 seed lottery into reliable
convergence. Pass `find_init=False` to disable, or your own `initvals` to override.

**When does Pathfinder help, and why is it usually unnecessary?**
`find_init='pathfinder'` runs multipath Pathfinder (variational; lands *in the
typical set* rather than at the MAP mode) and seeds each chain from a distinct
Pathfinder draw. In high dimensions the MAP mode is **not** in the typical set, so
on genuinely nasty geometries Pathfinder is the right centre. It needs
`pymc_extras` (in the `bauer_cuda` env) and falls back to mapjitter with a warning
if unavailable.

Two reasons it stays a break-glass lever, not a default:

- **It's not needed once the contaminant is in.** On the real 66-subject data,
  mapjitter and Pathfinder converge **identically** (r̂ 1.00; min ESS 4264 for
  mapjitter vs 3732 for Pathfinder). The contaminant fixes the geometry; a fancier
  init buys nothing.
- **It's expensive** — a `find_MAP` plus a multipath VI run before sampling even
  starts. And it has its own foot-gun: if you let Pathfinder's optimization start
  from a random point it can wander into the WFPT flat/−inf zone and hand NUTS a
  point from which *every* transition diverges (observed: independent-noise DDM →
  **4000/4000 divergences, within-chain SD 0**, frozen). bauer therefore **seeds
  Pathfinder from `pm.find_MAP`**; with that seeding the same fit went to **r̂ 1.03,
  0 divergences**. Use Pathfinder only after the contaminant + halfnormal + mapjitter
  ladder has failed.

### 2.4 Fast-RT contamination of the WFPT likelihood → `rt >= 0.20 s` filter

**Symptom.** Stuck chains and a pile of divergences concentrated early in warmup;
on the RDM specifically, **~80 divergences** that no `target_accept` setting clears.

**Diagnosis.** The Wiener first-passage (WFPT) density has a flat, **zero-gradient**
region whenever the non-decision time `t0` exceeds the fastest observed RT: there is
no `t0` value that can produce an RT below the floor, so the gradient w.r.t. `t0`
vanishes and NUTS gets stuck against the wall. A handful of implausibly fast
trials (anticipations, fat-finger responses) pin that floor far too low.

**Fix.** Drop `rt < 0.20 s` before fitting. This restores a usable `t0` gradient
and **killed ~80 RDM divergences**. 0.20 s is a reasonable motor floor for a
button-press magnitude task — tune it to your paradigm's true motor minimum, but
filter *something*. This is orthogonal to the contaminant: the filter removes
**implausibly fast** trials (no diffusion process is that quick); the contaminant
(§2.5) absorbs **implausibly slow** ones. You want both.

### 2.5 The slow-RT tail — *the big one*: three symptoms, one cause

This is the lesson that cracked the 66-subject fit. **Three apparently unrelated
problems were a single cause.**

**Symptoms (all three at once on the contaminant-free DDM at 66 subjects):**

1. **RT-tail PPC misfit.** The posterior-predictive RT distribution
   under-predicts the slow tail — real subjects produce long, lapse-like RTs the
   pure WFPT can't generate.
2. **Exploding Pareto-k → LOO unreliable.** Those slow-tail trials are
   high-leverage outliers under the WFPT likelihood; their PSIS-LOO Pareto-k
   shoots past 0.7 (often >1), so `az.loo` is not trustworthy and any model
   comparison built on it is invalid.
3. **Non-convergence.** r̂ ≈ **1.53** (HalfNormal SD) to **2.42** (HalfCauchy SD),
   min ESS **5–33**. The chains never mix.

**Diagnosis.** All three are **unmodeled lapse RTs**. On a fraction of trials the
subject isn't running the decision process at all (guess, attention slip, distracted
late response). A pure-WFPT likelihood has no way to explain a 4-second response
except by grotesquely distorting `a`, `t0`, and the drift — which roughens the
likelihood (→ non-convergence), under-covers the tail (→ PPC misfit), and makes
those trials enormous-leverage outliers (→ exploding Pareto-k). One disease, three
faces. The tell is the same as for static choice models: **synthetic data
simulated from the model converges fine; the real data doesn't** — because the
synthetic data has no lapses.

**Fix.** Turn on the contaminant — and in bauer ≥ 0.3.0 it's already on. The
likelihood becomes the stable HSSM-style mixture:

```
logp = log( (1 - p_outlier)*exp(ll_WFPT) + p_outlier*exp(lapse_logp) + 1e-29 )
```

where the contaminant is a flat density over RT on `[0, lapse_upper]`
(`lapse_upper = 20.0 s`, HSSM convention) times a 50/50 choice, so
`lapse_logp = log(0.5) - log(lapse_upper)` (a constant). Keep `p_outlier` **fixed
at 0.05** (a model *attribute*, not a sampled parameter, so it cannot funnel).

**The numbers (66-subject DDM, mapjitter):**

| config | r̂ | divergences | min ESS |
|---|---|---|---|
| plain DDM, no contaminant, HalfCauchy SD | 2.42 | 0 | 5 |
| plain DDM, no contaminant, HalfNormal SD | 1.53 | 0 | 33 |
| **+ fixed p_outlier=0.05, mapjitter** | **1.00** | **0** | **3211–4264** |

That is the headline: ESS **7 → 3211**, r̂ **1.59 → 1.00** on the worst-mixing
group node, from adding one fixed scalar. The contaminant simultaneously fixes the
PPC tail (the predictive now covers the slow RTs) and tames Pareto-k (the outliers
are now *expected* under the mixture, so LOO becomes usable).

**Controls (bauer ≥ 0.3.0):**

- `model.p_outlier = 0.05` — default, **fixed**. Use this.
- `model.p_outlier = 0.0` — recovers the pure WFPT (the old contaminant-free DDM).
- `model.p_outlier = 'hierarchical'` — per-subject estimated rate (Beta "1/x"
  group prior). **Avoid for the DDM**: weakly identified inside the WFPT mixture,
  it diverges **4000/4000** even with seeded Pathfinder. If you must estimate it,
  estimate a single scalar, not per-subject.
- `model.lapse_upper` (20.0 s), `model.lapse_choice_5050` (True) configure the
  contaminant; `model.simulate_contaminant` (True) makes `simulate`/`ppc` generate
  the contaminant so the PPC covers the RT tail (matches HSSM).

Set the attribute **before** `build_estimation_model` (or re-derive free
parameters after changing it). The contaminant also composes with the
`memory_as_sv` variant — the lapse mixin honors `sv`.

### 2.6 The group-SD funnel → HalfNormal prior

**Symptom.** Divergences concentrate at the `*_sd` (group-level standard
deviation) nodes. The **RDM** is the clear case: with `group_sd_dist='halfcauchy'`,
r̂ **1.12–1.20** and min ESS **~20** — the funnel returns even *with* the
contaminant on.

**Diagnosis.** A hierarchical group SD that is weakly identified (few subjects, or
a noise component the data barely constrain) under a **HalfCauchy** prior can run
away. HalfCauchy(0.25) has an **infinite-variance heavy tail**: there is always
non-trivial prior mass at very large SD, so the group SD wanders up, the
non-centered per-subject offsets `param = group_mu + group_sd * offset` blow up to
compensate, and you get the classic neck-of-the-funnel geometry NUTS diverges in.
(Note: the hierarchy is *already non-centered* in
`build_hierarchical_nodes` — this is not the subject-level funnel; it's the
group-SD prior tail.)

**Fix.** `group_sd_dist = 'halfnormal'` — the **0.3.0 default** (changed from
`'halfcauchy'`). HalfNormal has a Gaussian (light) tail: prior mass decays as
`exp(-x²)`, so a weakly-identified SD cannot escape to huge values, the funnel
neck never opens, and the per-subject offsets stay bounded. On the RDM this moves
min ESS from **~20 to hundreds** and r̂ to ~1.0.

**Why HalfNormal, the Gelman lineage.** Gelman (2006) originally recommended a
weakly-informative **HalfCauchy** on hierarchical scale parameters as a default,
precisely because its heavy tail is "open-minded" about large between-group
variance when you have many groups. But the same heavy tail is a liability when a
scale is *weakly identified* (few groups, or a barely-constrained component): the
infinite-variance tail lets the SD posterior leak into the funnel. The field's
practice evolved toward **HalfNormal** (and folded-/half-Student-t with finite df)
for exactly these regimes — a finite-variance tail that still puts most mass near
zero but cannot run away. bauer follows that evolution: HalfNormal by default,
HalfCauchy available per-instance (`group_sd_dist='halfcauchy'`) to reproduce
pre-0.3.0 fits.

**Trade-off.** Mild over-shrinkage if the true between-subject heterogeneity is
genuinely large. In practice, for these DDM/RDM noise components, the funnel is the
bigger risk, so HalfNormal wins.

### 2.7 The `v_scale ↔ evidence_sd` degeneracy → `fit_v_scale=False`

**Symptom.** You "added flexibility" by freeing `v_scale`; non-identifiability and
divergences appear, the noise parameters wander.

**Diagnosis.** The DDM drift is
`v = v_scale * (post_n2_mu - post_n1_mu) / sqrt(sd1² + sd2²)`. Multiply every
`evidence_sd` by `c` and divide `v_scale` by `c` and the drift is **bit-for-bit
unchanged** — a perfect multiplicative degeneracy. With a flat-ish prior the
likelihood ridge is exactly diagonal in `(v_scale, evidence_sd)` and NUTS cannot
resolve it.

**Fix.** `fit_v_scale=False` — fix `v_scale = 1` and let `evidence_sd` absorb the
overall drift scale. This is the **default** on the DDM models. Never free it
"for flexibility." (If someone freed it while debugging, that alone explains a lot
of divergence.)

---

## 3. Diagnosing convergence

Run these on **every** fit before interpreting a single parameter.

```python
import arviz as az

# 1. r-hat and ESS. Judge convergence on the GROUP-LEVEL (*_mu) nodes.
s = az.summary(idata)
group = s.filter(like='_mu', axis=0)          # group means: the inferential targets
print("max group r_hat :", group['r_hat'].max())      # want <= 1.01
print("min group ess   :", group['ess_bulk'].min())   # want >= 400
print("max overall rhat:", s['r_hat'].max())
print("min overall ess :", s['ess_bulk'].min())

# 2. Divergences. Want 0 for the final fit (a few are OK at ta~0.9 while iterating).
print("divergences:", int(idata.sample_stats['diverging'].sum()))

# 3. PSIS-LOO + Pareto-k — also a likelihood-misfit detector (see §2.5).
#    Needs the pointwise log-likelihood attached first:
idata = m.compute_log_likelihood(idata, paradigm=df)   # var_name='ll_ddm' (DDM)
loo = az.loo(idata, pointwise=True)
print(loo)
print("max pareto_k:", float(loo.pareto_k.max()))      # want < 0.7
```

What to check, and the thresholds:

- **Group-level `*_mu` r̂ ≤ 1.01.** These are the inferential targets (group means,
  group contrasts). A handful of weakly-identified *subject-level* params with mild
  r̂ (≤ ~1.05) is normal — don't chase it. Judge the fit on the group level.
- **min ESS ≥ 400** (per the standard rule of thumb for stable tail quantiles).
  On a converged contaminant DDM you'll see thousands.
- **divergences = 0** for the final, reportable fit. While iterating you can run
  `target_accept ≈ 0.9` and tolerate a handful, then raise to 0.95–0.99 to drive
  them to zero.
- **Pareto-k < 0.7** for LOO to be reliable. Exploding k is *itself a diagnosis*
  (unmodeled tail → §2.5), not just a LOO inconvenience.

**One stuck chain on a big hierarchical fit?** Even with mapjitter, a single chain
can occasionally wander off and stay there by chance, inflating r̂ across many
params at once. Tally which chain drives the high-r̂ params; if it's one chain,
the others are fine — just re-run with a different `random_seed` and/or longer
`tune`. This is a different animal from the systematic non-convergence of §2.5
(which **no** seed fixes, and which more warmup does **not** cure).

---

## 4. DDM vs RDM — orthogonal fixes

The DDM and the RDM have **two different geometry problems**, and they need
**two different fixes**. Conflating them costs days.

| | DDM | RDM |
|---|---|---|
| dominant problem | slow-RT tail (likelihood roughness) | group-SD funnel |
| fixed by | **contaminant** (`p_outlier=0.05`) | contaminant **+** `group_sd_dist='halfnormal'` |
| without the fix | r̂ 1.53–2.42, ESS 5–33 | r̂ 1.12–1.20, ESS ~20 |
| with the fix | r̂ 1.00, ESS ~4264 | r̂ ~1.0, ESS hundreds |

- **The DDM is a *tail* problem.** Adding the fixed contaminant cracks it
  (r̂ → 1.00, ESS → 4264). HalfNormal helps the contaminant-*free* DDM
  (r̂ 2.42 → 1.53) but is **not needed** once the contaminant is in.
- **The RDM is a *funnel* problem on top.** The contaminant alone is **not enough**:
  with HalfCauchy the RDM still funnels at the group-SD level (r̂ 1.12–1.20, ESS
  ~20). It needs **both** the contaminant **and** `group_sd_dist='halfnormal'` to
  reach r̂ ~1.0, ESS in the hundreds.

These two fixes are **orthogonal** — different parts of the model (the RT
likelihood vs. the group-SD prior), addressing different pathologies (tail
roughness vs. funnel geometry). The 0.3.0 defaults (contaminant on, HalfNormal
group SD) ship both, so a fresh DDM *or* RDM fit gets the right treatment out of
the box. Just don't reach for HalfNormal to fix a DDM tail problem, or for the
contaminant to fix an RDM funnel — match the fix to the pathology.

---

## 5. Escalation ladder (cheapest first)

If a fit still won't mix after the recipe in §1:

1. **Confirm the basics.** `rt >= 0.20` filtered? `fit_v_scale=False`? `p_outlier`
   = 0.05 (not 0.0, not 'hierarchical')? `backend='numpyro'` actually passed?
2. **Group-SD funnel?** Set `group_sd_dist='halfnormal'` (default in 0.3.0; verify
   you didn't override it). Decisive for the RDM.
3. **Raise `target_accept`** toward 0.99 for the final divergence-free fit. (It's a
   band-aid for residual geometry, not a substitute for the model fixes — measured
   ~2.3× slower at 0.99 vs 0.9 for the same r̂/ESS.)
4. **Still a seed lottery / stuck chains?** `find_init='pathfinder'` (MAP-seeded,
   needs `pymc_extras`). Expensive — escalation only, and rarely needed once the
   contaminant is in.
5. **Debug the plain DDM before the `memory_as_sv` (sv) DDM** — the across-trial
   `sv` integral is ~5 s/iteration; find the fix on the cheap model, then port it.
6. **Validate with parameter recovery**, not just the one real fit: simulate from
   the fitted posterior, refit, check r̂ across an ensemble and recoverability.
   Convergence on a single dataset isn't enough — a fix can converge to a biased
   estimate.

---

*Source material distilled here: `notes/sampling_robustness.md`,
`notes/fitting_ddm_models.md`, and the dyscalculic_ddm handoff notes
(`contaminant_model_handoff.md`, `why_noise_attribution_flips.md`,
`ddm_fitting_notes_for_team.md`). API verified against `bauer/models/ddm.py`,
`bauer/models/race.py`, `bauer/core.py` at v0.3.0.*
