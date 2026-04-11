"""Generate tutorial notebooks for the bauer documentation.

Usage:
    python make_notebooks.py          # regenerate all (skip unchanged)
    python make_notebooks.py --force  # regenerate all (overwrite even if unchanged)
    python make_notebooks.py 1 4      # regenerate only lessons 1 and 4
"""
import nbformat as nbf
import json, sys, os

def code(src): return nbf.v4.new_code_cell(src.strip())
def md(src):   return nbf.v4.new_markdown_cell(src.strip())

def write_if_changed(nb, path):
    """Write notebook only if the cell sources changed (preserves executed outputs)."""
    new_json = nbf.writes(nb)
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        # Compare only cell sources, not outputs/metadata
        old_sources = [(c['cell_type'], c['source']) for c in old.get('cells', [])]
        new_parsed = json.loads(new_json)
        new_sources = [(c['cell_type'], c['source']) for c in new_parsed.get('cells', [])]
        if old_sources == new_sources and '--force' not in sys.argv:
            print(f"{path} unchanged (skipped)")
            return
    with open(path, 'w') as f:
        f.write(new_json)
    print(f"{path} written")


# ─────────────────────────────────────────────────────────────────────────────
# Lesson 1 — Psychophysical Modelling: Theory and Barreto-Garcia et al. Magnitude Data
# ─────────────────────────────────────────────────────────────────────────────

nb1 = nbf.v4.new_notebook()
nb1.cells = [

md(r"""# Lesson 1: An Introduction to Psychophysical Modelling

## What is psychophysics?

**Psychophysics** studies the quantitative relationship between physical stimuli and the
perceptions or decisions they produce.  The key insight is that perception is not
deterministic: the same physical stimulus does not always produce the same internal
representation.  Noise — arising from photoreceptor variability, neural transmission,
working memory limitations, and random fluctuations throughout the sensory hierarchy —
means that every measurement is uncertain.

A **psychophysical model** formalises this noise to link three things:
1. The **stimulus** (a physical quantity such as the number of dots on a screen).
2. The **internal representation** (a noisy encoding of the stimulus).
3. The **response** (a forced choice, a rating, a reaction time).

By fitting such a model to behavioural data we can *infer* parameters that are not
directly observable — things like how much noise the brain adds to a given stimulus, or
what prior beliefs the observer brings into the experiment.  This inference is what
separates psychophysical modelling from simple curve-fitting: we get **posterior
distributions** over interpretable parameters, with proper uncertainty quantification.

## The bauer library

**bauer** (*B*ayesian Estimation of Perceptu*a*l, N*u*merical and *R*isky Choic*e*)
is a Python library that makes hierarchical Bayesian psychophysical modelling easy.
It is built on [PyMC](https://www.pymc.io) and [ArviZ](https://python.arviz.org) and
provides:

- Ready-to-use model classes for **magnitude comparison**, **psychometric functions**,
  and **risky choice** — no need to hand-code PyMC models.
- **Hierarchical fitting** by default: each participant gets their own parameters, but
  they are regularised by a shared group-level distribution.
- **Regression support** via patsy formulas — e.g. `regressors={'nu': 'C(condition)'}` to
  estimate noise separately per condition.
- **Posterior predictive checks** (PPC) with a single `model.ppc(data, idata)` call.
- Full **ArviZ** integration for trace diagnostics, HDI plots, ELPD comparison, and more.

In these tutorials we walk through the main model families, starting from first principles.
"""),

md(r"""## The Noisy Logarithmic Coding (NLC) model

When we judge quantities — the number of coins in a pile, the size of a reward — our
internal representations are noisy.  The **NLC model** posits that the brain encodes
numerical magnitude $n$ on a **logarithmic** scale, and that this log-representation is
corrupted by Gaussian noise:

$$r \sim \mathcal{N}(\log n, \; \nu^2)$$

The logarithmic encoding has two important consequences:

1. **Weber's law** falls out automatically.  Because the noise $\nu$ is constant on a
   log scale, the *absolute* noise on the original scale grows in proportion to the
   magnitude: equal log-space noise means proportionally larger linear-space uncertainty
   for big numbers than for small ones.

2. **Scale invariance**: when we plot the psychometric function against the log-ratio
   $\log(n_2/n_1)$, all curves for different reference magnitudes $n_1$ collapse onto a
   single sigmoid.  This is a direct and falsifiable prediction of the model.

### The decision rule

Given two stimuli $n_1$ and $n_2$ with independent log-space noise, the probability of
choosing $n_2$ as the larger is

$$P(\text{chose}\; n_2) = \Phi\!\left(\frac{\log(n_2/n_1)}{\sqrt{\nu_1^2 + \nu_2^2}}\right)$$

where $\Phi$ is the standard normal CDF, $\nu_1$ is the noise on $n_1$ (the first-presented
option) and $\nu_2$ is the noise on $n_2$ (the second-presented option).

In tasks where stimuli are shown **sequentially** (the observer perceives $n_1$, holds it
in working memory, then perceives $n_2$), memory retention may add noise.  The model
therefore allows $\nu_1 \neq \nu_2$.  In simultaneous-presentation tasks a single shared
$\nu$ is sufficient.
"""),

code("""\
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm as scipy_norm

# ── Scale invariance demo ─────────────────────────────────────────────────────
nu = 0.45                              # equal noise for both options
n1_vals = [5, 10, 20, 28]
n2_linear = np.linspace(1, 60, 300)
log_ratios = np.linspace(-1.8, 1.8, 300)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
pal = sns.color_palette('YlOrRd', len(n1_vals))

for (n1, c) in zip(n1_vals, pal):
    p_lin = scipy_norm.cdf(np.log(n2_linear / n1) / (np.sqrt(2) * nu))
    p_log = scipy_norm.cdf(log_ratios            / (np.sqrt(2) * nu))
    axes[0].plot(n2_linear, p_lin, color=c, lw=2, label=f'n1={n1}')
    axes[1].plot(log_ratios, p_log, color=c, lw=2, label=f'n1={n1}')

for ax in axes:
    ax.axhline(0.5, ls='--', c='gray', lw=1)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(title='n1', fontsize=8)
    sns.despine(ax=ax)

axes[0].axvline(0, ls='--', c='gray', lw=1)
axes[1].axvline(0, ls='--', c='gray', lw=1)
axes[0].set_xlabel('n2 (linear)')
axes[1].set_xlabel('log(n2 / n1)')
axes[0].set_ylabel('P(chose n2)')
axes[0].set_title('Linear scale — curves spread out')
axes[1].set_title('Log-ratio scale — curves collapse')
plt.suptitle('Scale invariance: log encoding makes all n\u2081 curves identical', y=1.02)
plt.tight_layout()
"""),

md(r"""## The psychometric function: noise, precision, and asymmetry

### 1. Noise level controls the slope

The slope of the psychometric function is entirely determined by the **combined noise**
$\sigma_\text{total} = \sqrt{\nu_1^2 + \nu_2^2}$.  More noise → shallower curve →
worse discrimination.  The left panel below illustrates this for equal noise on both
options ($\nu_1 = \nu_2 = \nu$).

### 2. Asymmetric noise flattens but does not shift the curve

When one option is noisier than the other — for instance because $n_1$ was encoded a few
seconds ago and is now degraded in working memory — the midpoint of the curve
($P = 0.5$ at $\log(n_2/n_1) = 0$) does not shift: **asymmetric noise alone does not
produce a bias**, it just makes discrimination harder.

The right panel illustrates a subtlety: even when the *total* noise budget $\nu_1 + \nu_2$
is held constant, redistributing it unequally raises
$\sigma_\text{total} = \sqrt{\nu_1^2 + \nu_2^2}$ (by Cauchy-Schwarz, this is minimised
when $\nu_1 = \nu_2$).  So working-memory degradation does not just "move" noise from
one option to the other — it genuinely costs precision.
"""),

code("""\
log_r = np.linspace(-2, 2, 300)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: noise level → precision (equal noise on both options)
ax = axes[0]
for nu_val, label, c in [(.2, '\u03bd = 0.20 (sharp)', '#08519c'),
                           (.5, '\u03bd = 0.50', '#3182bd'),
                           (1., '\u03bd = 1.00 (noisy)', '#bdd7e7')]:
    sigma = np.sqrt(2) * nu_val   # nu1 = nu2 = nu_val
    ax.plot(log_r, scipy_norm.cdf(log_r / sigma), lw=2, color=c, label=label)
ax.axhline(.5, ls='--', c='gray', lw=1); ax.axvline(0, ls='--', c='gray', lw=1)
ax.set_xlabel('log(n2/n1)'); ax.set_ylabel('P(chose n2)')
ax.set_title('Effect of noise level (\u03bd\u2081 = \u03bd\u2082 = \u03bd)')
ax.legend(); sns.despine(ax=ax)

# Right: same total noise budget ν₁ + ν₂ = 0.80, but distributed differently
# Key: σ_total = √(ν₁² + ν₂²) is MINIMISED when ν₁ = ν₂ (Cauchy-Schwarz).
# So asymmetric noise → higher σ_total → flatter curve, even at equal total.
ax = axes[1]
cases = [
    (0.4, 0.4, 'Equal  \u03bd\u2081=\u03bd\u2082=0.40  (sum=0.80)', '#1a9850'),
    (0.55, 0.25, 'Mild asymmetry  \u03bd\u2081=0.55, \u03bd\u2082=0.25  (sum=0.80)', '#f46d43'),
    (0.7, 0.1, 'Strong asymmetry  \u03bd\u2081=0.70, \u03bd\u2082=0.10  (sum=0.80)', '#a50026'),
]
for nu1, nu2, label, c in cases:
    sigma = np.sqrt(nu1**2 + nu2**2)
    ax.plot(log_r, scipy_norm.cdf(log_r / sigma), lw=2, color=c,
            label=f'{label}  (\u03c3={sigma:.2f})')
ax.axhline(.5, ls='--', c='gray', lw=1); ax.axvline(0, ls='--', c='gray', lw=1)
ax.set_xlabel('log(n2/n1)')
ax.set_title('Same total noise (\u03bd\u2081+\u03bd\u2082=0.80), different split \u2192 flatter curve')
ax.legend(fontsize=7.5); sns.despine(ax=ax)

plt.tight_layout()
"""),

md(r"""## Bayesian inference and the central tendency bias

### The Bayesian observer

The NLC observer does not simply report the noisy measurement $r = \log n + \epsilon$.
Instead, they act as an **ideal Bayesian observer**: they combine the noisy evidence with
a **prior** belief about what magnitudes are likely to appear in the experiment.  If the
prior is $\mathcal{N}(\mu_0, \sigma_0^2)$ and the likelihood is
$\mathcal{N}(\log n, \nu^2)$, Bayes' rule gives a Gaussian posterior:

$$\hat\mu = \underbrace{\frac{\sigma_0^2}{\sigma_0^2 + \nu^2}}_{\gamma}\, r
           + (1-\gamma)\,\mu_0, \qquad
\hat\sigma^2 = \frac{\sigma_0^2\,\nu^2}{\sigma_0^2 + \nu^2}$$

The posterior mean $\hat\mu$ is a **weighted average** of the raw evidence $r$ and the
prior mean $\mu_0$.  The weight given to the prior, $1 - \gamma$, is larger when the
noise $\nu$ is large (the evidence is unreliable) or the prior is narrow (the observer is
confident about the range of stimuli).

### Central tendency bias

Because the prior pulls representations toward $\mu_0$, stimuli larger than $\mu_0$ are
**underestimated** and stimuli smaller than $\mu_0$ are **overestimated**.  This
systematic compression toward the prior mean is the **central tendency effect** — a
well-documented perceptual bias.
"""),

code("""\
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

x = np.linspace(0.5, 5.5, 400)       # log-magnitude axis
log_n_true  = np.log(20)              # true stimulus: n=20
prior_mu    = np.log(10)              # prior centred at n=10
nu          = 0.4                     # measurement noise

for ax, (prior_sd, title) in zip(axes,
        [(1.0, 'Weak prior  (\u03c3\u2080 = 1.0)'),
         (0.25, 'Strong prior  (\u03c3\u2080 = 0.25)')]):

    gamma   = prior_sd**2 / (prior_sd**2 + nu**2)
    post_mu = prior_mu + gamma * (log_n_true - prior_mu)
    post_sd = np.sqrt(prior_sd**2 * nu**2 / (prior_sd**2 + nu**2))

    like  = scipy_norm.pdf(x, log_n_true, nu)
    prior = scipy_norm.pdf(x, prior_mu,  prior_sd)
    post  = scipy_norm.pdf(x, post_mu,   post_sd)
    mx    = max(like.max(), prior.max(), post.max())

    ax.fill_between(x, prior/mx, alpha=.3, color='#4393c3', label=f'Prior  \u03bc\u2080={prior_mu:.2f}')
    ax.fill_between(x, like/mx,  alpha=.3, color='#d6604d', label=f'Evidence  log(n)={log_n_true:.2f}')
    ax.fill_between(x, post/mx,  alpha=.5, color='#4dac26', label=f'Posterior  \u03bc\u0302={post_mu:.2f}')
    ax.axvline(log_n_true, ls='--', c='#d6604d', lw=1.5)
    ax.axvline(prior_mu,   ls='--', c='#4393c3', lw=1.5)
    ax.axvline(post_mu,    ls='-',  c='#4dac26', lw=2.5)
    ax.set_xlabel('Internal log-magnitude representation')
    ax.set_ylabel('Probability density (normalised)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    sns.despine(ax=ax)

plt.suptitle('Central tendency: posterior mean is pulled toward the prior', y=1.02)
plt.tight_layout()
"""),

md(r"""### Why asymmetric noise creates magnitude–order interactions

In a sequential task, $n_1$ is noisier than $n_2$ because of memory degradation:
$\nu_1 > \nu_2$.  The Bayesian posterior mean is

$$\hat\mu_k = \gamma_k \log n_k + (1-\gamma_k)\mu_0,
\quad \gamma_k = \frac{\sigma_0^2}{\sigma_0^2 + \nu_k^2}$$

The weight $\gamma_k$ is **different** for the two options whenever $\nu_1 \neq \nu_2$.

- The **noisier** option (usually $n_1$) gets a smaller weight $\gamma_1$, meaning its
  posterior is pulled *more* toward $\mu_0$.
- The **less noisy** option (usually $n_2$) gets a larger weight $\gamma_2$ and is
  pulled *less* toward $\mu_0$.

If $\mu_0$ is set to the mean of the log-stimulus distribution, then large stimuli are
underestimated and small stimuli are overestimated.  Because $n_1$ is underestimated
*more* than $n_2$, the observer is effectively biased **toward choosing $n_2$** when
both are large (big $n_1$ is shrunk, making $n_2$ look relatively bigger), and
**against choosing $n_2$** when both are small (small $n_1$ is inflated, making $n_2$
look relatively smaller).

**This is the source of the magnitude–order interaction**: it is not asymmetric noise
per se that causes it, but asymmetric noise interacting with a Bayesian prior that
compresses different options by different amounts.  Equal noise on both options would
produce a uniform shift of the midpoint — no interaction with the reference magnitude.

The NLC model captures all of this with just three parameters per subject: $\nu_1$,
$\nu_2$, and $\sigma_0$ (prior width).
"""),

code("""\
# Illustrate the interaction: how the choice bias changes with n1 and noise asymmetry
log_r_range = np.linspace(-2, 2, 400)
nu1_values  = [0.3, 0.5, 0.8]   # increasing memory noise on n1
nu2         = 0.3                # fixed perceptual noise on n2
prior_sd    = 0.6                # moderate prior
prior_mu    = 0.0                # centred (mean of log-stimulus distribution)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
pal = sns.color_palette('Reds_d', len(nu1_values))

# Left: large n1 (above prior mean); Right: small n1 (below prior mean)
for ax, log_n1, n1_label in zip(axes, [1.5, -1.5],
        ['Large n\u2081 (above prior \u2192 compressed down)',
         'Small n\u2081 (below prior \u2192 compressed up)']):
    for nu1, c in zip(nu1_values, pal):
        gamma1 = prior_sd**2 / (prior_sd**2 + nu1**2)
        gamma2 = prior_sd**2 / (prior_sd**2 + nu2**2)
        log_n2_vals = log_n1 + log_r_range
        mu1_hat = prior_mu + gamma1 * (log_n1      - prior_mu)
        mu2_hat = prior_mu + gamma2 * (log_n2_vals  - prior_mu)
        sigma_total = np.sqrt(prior_sd**2 * nu1**2 / (prior_sd**2 + nu1**2)
                            + prior_sd**2 * nu2**2 / (prior_sd**2 + nu2**2))
        p = scipy_norm.cdf((mu2_hat - mu1_hat) / (np.sqrt(2) * sigma_total))
        ax.plot(log_r_range, p, color=c, lw=2.5,
                label=f'\u03bd\u2081={nu1:.1f}, \u03bd\u2082={nu2:.1f}')
    ax.axhline(.5, ls='--', c='gray', lw=1)
    ax.axvline(0,  ls='--', c='gray', lw=1)
    ax.set_xlabel('log(n\u2082 / n\u2081)')
    ax.set_ylabel('P(chose n\u2082)')
    ax.set_title(n1_label)
    ax.legend(fontsize=8.5); sns.despine(ax=ax)

plt.suptitle('Asymmetric noise + prior: curve shift depends on reference magnitude',
             fontsize=12, y=1.02)
plt.tight_layout()
"""),

md("""## Barreto-Garc\u00eda et al. (2023): Magnitude comparison task

With the theory in place, we now fit models to real data.  The dataset from
**Barreto-Garc\u00eda et al. (2023)** is bundled with bauer and contains magnitude comparison
choices from 64 participants.  On each trial participants viewed two sequentially
presented coin clouds (visual dot arrays) and judged which contained more 1-CHF coins.

- **Reference magnitudes**: $n_1 \\in \\{5, 7, 10, 14, 20, 28\\}$
- **Comparison magnitudes**: $n_2$ varied over a wide range
- The `choice` column encodes `True` = chose $n_2$ (the second cloud)

bauer provides a one-liner to load the data:
"""),

code("""\
import pandas as pd
import arviz as az
from bauer.utils.data import load_garcia2022
from bauer.models import MagnitudeComparisonModel

data = load_garcia2022(task='magnitude')
print(f"Subjects: {data.index.get_level_values('subject').nunique()},  "
      f"Trials: {len(data)}")
data.head()
"""),

code("""\
# Compute log-ratio for each trial and bin for plotting
data['log(n2/n1)'] = np.log(data['n2'] / data['n1'])
data['bin'] = (pd.cut(data['log(n2/n1)'], bins=12)
                 .map(lambda x: x.mid).astype(float))

grouped = (data.groupby(['n1', 'bin'])['choice']
               .agg(['mean', 'count']).reset_index()
               .query('count >= 5'))

n1_unique = sorted(grouped['n1'].unique())
pal_n1 = sns.color_palette('YlOrRd', len(n1_unique))
n1_colors = dict(zip(n1_unique, pal_n1))

fig, ax = plt.subplots(figsize=(7, 5))
for n1_val in n1_unique:
    d = grouped[grouped['n1'] == n1_val]
    ax.plot(d['bin'], d['mean'], 'o-', ms=5, lw=1.5,
            color=n1_colors[n1_val], label=f'n\u2081 = {n1_val}')
ax.axhline(.5, ls='--', c='gray', lw=1)
ax.axvline(.0, ls='--', c='gray', lw=1)
ax.set_ylim(-.05, 1.05)
ax.set_xlabel('log(n\u2082 / n\u2081)')
ax.set_ylabel('P(chose n\u2082)')
ax.set_title('Log-ratio scale: all n\u2081 curves collapse (scale invariance)')
ax.legend(title='Reference n\u2081', fontsize=9)
sns.despine(); plt.tight_layout()
"""),

md(r"""The curves collapse when plotted against $\log(n_2/n_1)$, confirming the NLC
model's scale-invariance prediction.  The overlap is not perfect — small $n_1$ (5, 7)
produce slightly steeper curves, hinting at a mild departure from strict Weber's law
at low magnitudes.  Compare this with the natural-scale plots below.

## Weber's law: what linear encoding gets wrong

If we model choices in *natural* (linear) space and use a fixed noise $\nu$, the slope
of the psychometric function would be the same for all $n_1$ values.  But the data shows
a clear pattern: **steeper curves for small $n_1$, shallower for large $n_1$**.  This is
**Weber's law** — discrimination is proportionally harder for larger magnitudes.
"""),

code("""\
# Natural-space psychometric curves: slope and indifference point shift with n1
n1_unique = sorted(data['n1'].unique())
pal_n1 = sns.color_palette('YlOrRd', len(n1_unique))

fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharey=True)
axes = axes.flatten()

for ax, n1_val, c in zip(axes, n1_unique, pal_n1):
    d_n1 = data[data['n1'] == n1_val]
    grp  = (d_n1.groupby('n2')['choice']
               .agg(['mean', 'count']).reset_index()
               .query('count >= 3'))
    ax.scatter(grp['n2'], grp['mean'], color=c, s=25, alpha=.8)
    ax.plot(grp['n2'], grp['mean'], color=c, lw=1.5)
    ax.axhline(.5, ls='--', c='gray', lw=1)
    ax.axvline(n1_val, ls=':', c='tomato', lw=2, label=f'n2 = n1 = {n1_val}')
    ax.set_title(f'n1 = {n1_val}')
    ax.set_xlabel('n2  (linear scale)')
    ax.set_ylim(-.05, 1.05)
    ax.legend(fontsize=7.5)
    sns.despine(ax=ax)

axes[0].set_ylabel('P(chose n2)')
plt.suptitle(
    'Natural-space: slope decreases with n\u2081  \u2192  Weber\u2019s law',
    fontsize=12, y=1.01)
plt.tight_layout()
"""),

md(r"""### Recovering Weber's law non-parametrically

The NLC model assumes log-space encoding and derives Weber's law as a consequence.
But we can also test Weber's law *without* assuming log encoding, by fitting a model
that works in **natural space** — that is, the space of raw stimulus magnitudes $n$,
not $\log n$.

`PsychometricRegressionModel` is a simple psychometric-function model that takes two
stimuli $x_1, x_2$ and fits the probability of choosing $x_2$ as:

$$P(\text{chose}\; x_2) = \Phi\!\left(\frac{x_2 - x_1}{\nu}\right)$$

where the noise $\nu$ is now in the same units as $n$ (dots on screen), not in log-units.
By using `regressors={'nu': 'C(n1)'}`, we estimate a **separate $\nu$** for each reference
magnitude $n_1$ — no functional form is assumed.

If Weber's law holds, we expect $\nu(n_1) \propto n_1$: noise in natural space should
grow proportionally with magnitude.  This is the empirical signature of log-space encoding
when measured from outside, in the raw stimulus space.
"""),

code("""\
from bauer.models import PsychometricRegressionModel

# x1 = n1, x2 = n2 in natural space (raw magnitudes, not log-transformed)
data_lin = data.copy()
data_lin['x1'] = data_lin['n1'].astype(float)
data_lin['x2'] = data_lin['n2'].astype(float)

# C(n1): categorical coding — separate nu per n1 level, no linearity assumption
model_lin_reg = PsychometricRegressionModel(
    paradigm=data_lin,
    regressors={'nu': 'C(n1)'},
)
model_lin_reg.build_estimation_model(data=data_lin, hierarchical=True)
idata_lin_reg = model_lin_reg.sample(draws=150, tune=150, chains=4, progressbar=False)
"""),

code("""\
# Extract group-level nu at each unique n1 level via the model's design matrix
conditions_n1 = pd.DataFrame({'n1': n1_unique})
cond_pars = model_lin_reg.get_conditionwise_parameters(idata_lin_reg, conditions_n1, group=True)
nu_at_n1 = cond_pars.xs('nu', level='parameter')  # posterior samples × n1 levels

nu_mean = nu_at_n1.mean(0).values
nu_lo, nu_hi = np.percentile(nu_at_n1.values, [2.5, 97.5], axis=0)
n1_arr = np.array(n1_unique, dtype=float)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.fill_between(n1_arr, nu_lo, nu_hi, alpha=.30, color='#d73027', label='95 % credible interval')
ax.plot(n1_arr, nu_mean, lw=2, color='#d73027', marker='o', ms=6,
        label='Group-level \u03bd(n\u2081)  (C(n1) model)')

# Weber's law reference: proportional noise  \u03bd ∝ n1
k_weber = nu_mean[0] / n1_arr[0]
ax.plot(n1_arr, k_weber * n1_arr, '--k', lw=1.5,
        label=f'Weber reference  (\u03bd = {k_weber:.2f}\u00d7n\u2081)')

ax.set_xlabel('Reference magnitude  n\u2081')
ax.set_ylabel('Linear-space noise  \u03bd')
ax.set_title('Per-level \u03bd(n\u2081): proportional growth confirms Weber\u2019s law')
ax.legend(fontsize=9)
sns.despine()
plt.tight_layout()
"""),

md(r"""**Key takeaway:** $\nu$ grows proportionally to $n_1$.  This is precisely **Weber's
law**, and it is *automatically* accounted for when noise is constant on a log scale (the
NLC model).  There is no need to model the $n_1$-dependence of noise explicitly: logarithmic
encoding produces it for free.  This is why `MagnitudeComparisonModel` works so well —
and why the next section fits it to the data.
"""),

md(r"""## Fitting `MagnitudeComparisonModel` with bauer

### Why use bauer?

You could code a PyMC model by hand.  But bauer gives you:

| Feature | What it means in practice |
|---------|--------------------------|
| **One-line model construction** | `MagnitudeComparisonModel(paradigm=data)` |
| **Hierarchical structure automatically** | Group mean + between-subject SD inferred jointly with subject parameters |
| **Prior transforms baked in** | Noise parameters are always positive (softplus link), lapse rates in [0,1] |
| **Formula-based regression** | `regressors={'nu': 'C(condition)'}` — all patsy formulas work |
| **Built-in PPC** | `model.ppc(data, idata)` draws posterior-predictive choices |
| **ArviZ-native output** | All posteriors as `InferenceData`; use `az.plot_posterior`, `az.compare`, etc. directly |

### Why hierarchical modelling is essential

In a typical psychophysics experiment each participant sees a few hundred trials at most.
Fitting a multi-parameter cognitive model to 100--200 trials per subject gives very noisy,
often unidentifiable individual estimates.  The more expressive and theoretically
interesting the model, the worse this problem becomes: a 4-parameter model needs far more
data per subject than a 2-parameter model.

**Hierarchical (multilevel) modelling** solves this by assuming that participants are drawn
from a shared population.  Each subject $s$ gets their own parameters, but those parameters
are *regularised* toward the group mean — subjects with little data or extreme estimates
are pulled back toward the population, while subjects with lots of clear data are left
mostly alone.  This **partial pooling** gives you the best of both worlds:

- **Individual differences are preserved** — unlike a single group-level fit.
- **Noisy individuals are stabilised** — unlike fitting each subject independently.
- **Complex models become feasible** even at modest trial counts, because the group
  prior acts as a principled regulariser that prevents overfitting.

In practice, hierarchical fitting is what makes it possible to use models like the KLW
risk model (lesson 2) or the flexible-noise model (lesson 4) on real experimental data.
Without the group-level regularisation, the posterior for many subjects would be
dominated by the prior and the individual estimates would be meaningless.

### Hierarchical parameter structure in bauer

For the noise parameters, bauer sets up the hierarchy as:

$$\nu_k^{(s)} \sim \text{HalfNormal}(\mu_k,\, \sigma_k)$$

where $\mu_k$ (`n{k}_evidence_sd_mu`) is the **group mean** and $\sigma_k$
(`n{k}_evidence_sd_sd`) is the **between-subject spread**.  The posterior therefore
contains both the population-level estimate and the full distribution of individual
differences.

### MCMC sampling and posterior summaries

`.sample()` runs **Markov Chain Monte Carlo (MCMC)** — specifically, PyMC's No-U-Turn
Sampler (NUTS).  MCMC generates a large collection of *samples* that, after a warm-up
("tuning") phase, are drawn proportionally to the posterior probability.  The key idea:
rather than returning a single best-fit value, MCMC gives you a *distribution* over
parameter values that reflects both what the data support and how uncertain we are.

Two numbers summarise that distribution most usefully:
- **Posterior mean** — the expected value of the parameter.
- **Highest Density Interval (HDI)** — the shortest interval containing a given
  probability mass (typically 94 % or 95 %).  Unlike a frequentist confidence interval,
  the HDI can be read directly: *"there is 94 % posterior probability that the true
  value lies in this range."*

ArviZ's `az.plot_posterior` displays both automatically.  bauer exposes the samples as
an `arviz.InferenceData` object so all ArviZ diagnostics work out-of-the-box.
"""),

code("""\
model_mag = MagnitudeComparisonModel(paradigm=data)
model_mag.build_estimation_model(data=data, hierarchical=True, save_p_choice=True)
idata_mag = model_mag.sample(draws=200, tune=200, chains=4, progressbar=False)
"""),

md(r"""### Group-level posteriors

`az.plot_posterior` gives us the group-level posteriors for $\nu_1$ and $\nu_2$.
The 94 % HDI quantifies our uncertainty about the group mean.  We expect $\nu_1 > \nu_2$
because of working-memory degradation for the first-presented option.
"""),

code("""\
az.plot_posterior(
    idata_mag,
    var_names=['n1_evidence_sd_mu', 'n2_evidence_sd_mu'],
    figsize=(9, 3),
)
plt.suptitle('Group-level noise posteriors  (\u03bd\u2081 = first option,  \u03bd\u2082 = second option)', y=1.04)
plt.tight_layout()
"""),

md(r"""The group-level posterior for $\nu_1$ (`n1_evidence_sd_mu`) is consistently **larger**
than for $\nu_2$ (`n2_evidence_sd_mu`) — working-memory degradation in action.

### Individual subject estimates with 95 % credible intervals

The posterior also contains a full distribution for **each individual subject**.  bauer
provides two complementary tools:

- `plot_subjectwise_parameters(idata, parameter, ax=ax)` — direct call for a single parameter
- `get_subject_posterior_df(idata, parameters)` + `plot_subjectwise_pointplot` — tidy
  DataFrame workflow compatible with `sns.FacetGrid` (supports hue, facetting, etc.)
"""),

code("""\
from bauer.utils import (plot_subjectwise_parameters,
                          get_subject_posterior_df, plot_subjectwise_pointplot)

# ── FacetGrid workflow: one panel per parameter ───────────────────────────────
df_post = get_subject_posterior_df(
    idata_mag, ['n1_evidence_sd', 'n2_evidence_sd'])

g = sns.FacetGrid(df_post, col='parameter', sharey=True, height=4.5, aspect=1.1)
g.map_dataframe(plot_subjectwise_pointplot)  # uses default mean_col='mean', lo_col='lo', hi_col='hi'
g.set_axis_labels('Subject (sorted)', 'Noise  \u03bd')
g.set_titles(col_template='{col_name}')
g.figure.suptitle('Subject-level noise estimates (error bars = 94 % HDI)', y=1.04)

# Add group-mean line to each panel
for ax, param in zip(g.axes.flat, ['n1_evidence_sd', 'n2_evidence_sd']):
    mu_key = param + '_mu'
    if mu_key in idata_mag.posterior:
        gm = idata_mag.posterior[mu_key].values.mean()
        ax.axhline(gm, ls='--', lw=1.5, alpha=0.6, color='steelblue',
                   label='Group mean')
    sns.despine(ax=ax)

plt.tight_layout()
"""),

code("""\
# ── Scatter: ν₁ vs ν₂ per subject with 94 % HDI error bars ───────────────────
n_subj = idata_mag.posterior['n1_evidence_sd'].shape[-1]
s1 = idata_mag.posterior['n1_evidence_sd'].values.reshape(-1, n_subj)
s2 = idata_mag.posterior['n2_evidence_sd'].values.reshape(-1, n_subj)
nu1_mean, nu2_mean = s1.mean(0), s2.mean(0)
nu1_lo, nu1_hi = np.percentile(s1, [3, 97], axis=0)
nu2_lo, nu2_hi = np.percentile(s2, [3, 97], axis=0)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.errorbar(nu2_mean, nu1_mean,
            xerr=[nu2_mean - nu2_lo, nu2_hi - nu2_mean],
            yerr=[nu1_mean - nu1_lo, nu1_hi - nu1_mean],
            fmt='o', ms=5, alpha=.65, elinewidth=0.7, capsize=2.5,
            color='#2166ac', ecolor='#9ecae1', zorder=3)
lim = max(nu1_hi.max(), nu2_hi.max()) * 1.1
ax.plot([0, lim], [0, lim], 'k--', lw=1.2, label='\u03bd\u2081 = \u03bd\u2082')
ax.set_xlabel('Second-option noise  \u03bd\u2082')
ax.set_ylabel('First-option noise  \u03bd\u2081')
ax.set_title('\u03bd\u2081 vs \u03bd\u2082 — most subjects above diagonal (memory noise)')
ax.legend(fontsize=9); sns.despine(ax=ax)
plt.tight_layout()
"""),

md("""## Posterior predictive check

We draw predicted choice probabilities from the full posterior and overlay the 95 %
credible interval on the observed group-average data (one panel per $n_1$ value).
A good fit means the shaded band covers the observed dots — and the model should also
capture the different slopes across $n_1$ values (the Weber's-law signature).
"""),

code("""\
from bauer.utils import summarize_ppc_group

# Simulate binary choices from the posterior predictive
ppc_df  = model_mag.ppc(data, idata_mag, var_names=['ll_bernoulli'])
ppc_ll  = ppc_df.xs('ll_bernoulli', level='variable')   # trials \u00d7 posterior samples

ppc_flat = ppc_ll.reset_index()
ppc_flat['bin'] = (pd.cut(-ppc_flat['log(n1/n2)'], 12)
                     .map(lambda x: x.mid).astype(float))

g_ppc = summarize_ppc_group(ppc_flat, condition_cols=['n1', 'bin'])
g_ppc = g_ppc.rename(columns={'p_predicted': 'p_mean', 'hdi025': 'p_lo', 'hdi975': 'p_hi'})

data_copy = data.reset_index()
data_copy['bin'] = (pd.cut(-data_copy['log(n1/n2)'], 12)
                      .map(lambda x: x.mid).astype(float))
obs = (data_copy.groupby(['subject', 'n1', 'bin'])['choice'].mean()
                .groupby(['n1', 'bin']).mean())
g_ppc['choice'] = obs
g_ppc = g_ppc.reset_index()

import matplotlib.patches as mpatches

def draw_ppc(data, **kwargs):
    ax = plt.gca()
    ax.fill_between(data['bin'], data['p_lo'], data['p_hi'],
                    color='steelblue', alpha=.25)
    ax.plot(data['bin'], data['p_mean'], color='steelblue', lw=2)
    ax.scatter(data['bin'], data['choice'], color='steelblue', s=20, zorder=5)
    ax.axhline(.5, ls='--', c='gray', lw=1)
    ax.axvline(0,  ls='--', c='gray', lw=1)
    ax.set_ylim(-.05, 1.05)

g = sns.FacetGrid(g_ppc, col='n1', col_wrap=3, height=3.2, aspect=1.1, sharey=True)
g.map_dataframe(draw_ppc)
g.set_axis_labels('log(n2 / n1)', 'P(chose n2)')
g.set_titles('n\u2081 = {col_name}')
for ax in g.axes.flat:
    sns.despine(ax=ax)

legend_handles = [
    mpatches.Patch(color='steelblue', alpha=.25, label='95 % credible interval'),
    plt.Line2D([0], [0], color='steelblue', lw=2, label='Model mean'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
               markersize=6, label='Observed'),
]
g.figure.legend(handles=legend_handles, loc='lower center', ncol=3,
                fontsize=9, bbox_to_anchor=(.5, -.04))
g.figure.suptitle('Posterior predictive check — MagnitudeComparisonModel',
                   fontsize=13, y=1.02)
g.figure.tight_layout()
"""),

code("""\
# PPC overlay: all n1 values in one panel (hue = n1)
n1_pal = sns.color_palette('YlOrRd', g_ppc['n1'].nunique())
n1_colors = dict(zip(sorted(g_ppc['n1'].unique()), n1_pal))

fig, ax = plt.subplots(figsize=(8, 5))
for n1_val in sorted(g_ppc['n1'].unique()):
    d = g_ppc[g_ppc['n1'] == n1_val].sort_values('bin')
    c = n1_colors[n1_val]
    ax.fill_between(d['bin'], d['p_lo'], d['p_hi'], alpha=.15, color=c)
    ax.plot(d['bin'], d['p_mean'], lw=2, color=c, label=f'n\u2081 = {n1_val}')
    ax.scatter(d['bin'], d['choice'], s=20, color=c, zorder=5)
ax.axhline(.5, ls='--', c='gray', lw=1)
ax.axvline(0,  ls='--', c='gray', lw=1)
ax.set_ylim(-.05, 1.05)
ax.set_xlabel('log(n\u2082 / n\u2081)')
ax.set_ylabel('P(chose n\u2082)')
ax.set_title('PPC overlay: model predictions (bands) vs data (dots) by n\u2081')
ax.legend(title='Reference n\u2081', fontsize=9)
sns.despine(); plt.tight_layout()
"""),

md(r"""## Summary

In this lesson we built up the NLC model from first principles:

1. **Logarithmic encoding** of numerical magnitudes, with Gaussian noise $\nu$.
2. **Scale invariance**: the psychometric function collapses to a single sigmoid on the
   log-ratio axis.
3. **Noise level** controls slope; **asymmetric noise** ($\nu_1 > \nu_2$) flattens the
   curve without shifting it.
4. **Bayesian prior** produces the central tendency bias; asymmetric noise interacting
   with the prior creates **magnitude–order interactions**.
5. **Weber's law** falls out automatically from log-space encoding.
6. bauer fits all parameters hierarchically in a few lines, yielding full posterior
   distributions at both the group and individual-subject level.

In [Lesson 2](lesson2.ipynb) we move to **risky choice** and see how the same perceptual
noise parameters that distort magnitude perception also drive risk attitudes.
"""),

]

write_if_changed(nb1, 'lesson1.ipynb')


# ─────────────────────────────────────────────────────────────────────────────
# Lesson 2 — Risky Choice: Psychometric Functions and the Noise–Risk Link
# ─────────────────────────────────────────────────────────────────────────────

nb2 = nbf.v4.new_notebook()
nb2.cells = [

md(r"""# Lesson 2: Risky Choice — Psychometric Functions and the Noise–Risk Link

## From magnitude comparison to risky choice

The same 64 participants from Barreto-García et al. (2023) also made risky choices
**outside** the scanner.  On each trial they chose between a **sure payoff**
($p_\text{safe} = 1.0$, $n_\text{safe} \in \{5,7,10,14,20,28\}$) and a **risky gamble**
($p_\text{risky} = 0.55$, $n_\text{risky}$ varying).  Payoffs were shown in two
**formats** across separate blocks:

| Format | Representation |
|--------|----------------|
| `non-symbolic` | coin clouds — same format as the magnitude task |
| `symbolic` | Arabic numerals |

The paper's central argument: **the same perceptual noise that limits magnitude
discrimination also distorts risky-choice behaviour**, producing risk aversion as a
by-product of noisy numerical cognition.

## Approach: psychometric functions

Rather than building a full cognitive model (we will do that in lessons 3–4), we start
with the simplest possible analysis: fit a **psychometric function** to each participant's
risky-choice data and extract two numbers:

1. **Noise** ($\nu$) — how imprecise is the observer?  Low $\nu$ → steep curve → precise.
2. **Indifference point** ($\delta^*$, the `bias` parameter) — at what log-ratio does the
   observer switch from preferring safe to preferring risky?

If noise drives risk aversion, then noisier observers (higher $\nu$, lower precision)
should have a higher $\delta^*$ (shifted rightward = more risk-averse).
"""),

code("""\
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy.stats import norm as scipy_norm, spearmanr
from bauer.utils.data import load_garcia2022
from bauer.models import PsychometricModel

data_risk = load_garcia2022(task='risk')
print(f"Subjects: {data_risk.index.get_level_values('subject').nunique()},  "
      f"Trials: {len(data_risk)},  "
      f"Formats: {data_risk.index.get_level_values('format').unique().tolist()}")
"""),

md(r"""## Visualise risky-choice data

We plot the proportion of risky choices as a function of the log-ratio
$\log(n_\text{risky} / n_\text{safe})$.  The dashed vertical line marks the
**risk-neutral threshold** $\log(1/0.55) \approx 0.60$ — the point where a
risk-neutral observer is indifferent.

- Curve crosses 0.5 **to the right** of the dashed line → risk-averse
- Curve crosses 0.5 **to the left** → risk-seeking
"""),

code("""\
plot_data = data_risk.reset_index(level='format').copy()
plot_data['log(risky/safe)'] = np.log(plot_data['n2'] / plot_data['n1'])
plot_data['bin'] = (pd.cut(plot_data['log(risky/safe)'], bins=12)
                      .map(lambda x: x.mid).astype(float))

grouped = (plot_data.groupby(['format', 'bin'])['choice']
                    .agg(['mean', 'count']).reset_index()
                    .query('count >= 5'))

g = sns.FacetGrid(grouped, col='format', height=4.2, aspect=1.0,
                  palette={'non-symbolic': '#d95f02', 'symbolic': '#1f78b4'})
g.map(sns.lineplot, 'bin', 'mean', marker='o', markersize=5)
for ax in g.axes.flat:
    ax.axhline(.5,              ls='--', c='gray', lw=1)
    ax.axvline(np.log(1/.55),   ls='--', c='#d73027', lw=2, label='Risk-neutral threshold')
    ax.set_ylim(-.05, 1.05)
    ax.legend(fontsize=8)
g.set_axis_labels('log(risky / safe)', 'P(chose risky gamble)')
g.set_titles('Format: {col_name}')
plt.tight_layout()
"""),

md(r"""## The psychometric function for risky choice

bauer's `PsychometricModel` fits:

$$P(\text{chose risky}) = \Phi\!\left(\frac{\log(n_\text{risky}/n_\text{safe}) - \delta^*}{\sqrt{2}\,\nu}\right)$$

where $\Phi$ is the standard-normal CDF.  The two free parameters per subject are:

| Parameter | bauer name | Interpretation |
|-----------|-----------|----------------|
| $\nu$ | `nu` | **Noise** — SD of the internal log-magnitude representation.  Higher = noisier = shallower curve. |
| $\delta^*$ | `bias` | **Indifference point** — the log-ratio at which P(chose risky) = 0.5.  A risk-neutral observer has $\delta^* = \log(1/0.55) \approx 0.60$; larger values indicate risk aversion. |

Note that `bias` is **unconstrained**: it can be positive (risk-averse), zero, or negative
(risk-seeking).  This is important — some participants genuinely prefer the risky option
even when it has lower expected value, and the model can capture that.

### Why this works

On the log scale, choosing the risky option is optimal when:

$$\log n_\text{risky} - \log n_\text{safe} > \log(1/p_\text{risky})$$

i.e. when the risky payoff is large enough to compensate for the lower winning probability.
But this comparison is done with **noisy internal representations**, and the observer may
also have a **bias** toward or away from risk.  The psychometric function captures both
effects in a single curve.
"""),

code("""\
# Illustrate: how noise and bias shape the risky-choice curve
log_r = np.linspace(-.3, 2.2, 400)
ev_threshold = np.log(1/.55)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: varying bias (indifference point) at fixed noise
ax = axes[0]
for delta_star, label, c in [
        (0.3,           'Risk-seeking  (\u03b4* = 0.30)', '#2166ac'),
        (ev_threshold,  f'Risk-neutral  (\u03b4* = {ev_threshold:.2f})', '#878787'),
        (0.95,          'Mildly risk-averse  (\u03b4* = 0.95)', '#f46d43'),
        (1.45,          'Strongly risk-averse  (\u03b4* = 1.45)', '#d73027')]:
    p = scipy_norm.cdf((log_r - delta_star) * 1.6)
    ax.plot(log_r, p, color=c, lw=2.5, label=label)
ax.axhline(.5, ls='--', c='gray', lw=1)
ax.axvline(ev_threshold, ls=':', c='gray', lw=1.5, alpha=.5)
ax.set_xlabel('log(risky / safe)'); ax.set_ylabel('P(chose risky)')
ax.set_title('Varying indifference point \u03b4* (fixed noise)')
ax.legend(fontsize=8.5); sns.despine(ax=ax)

# Right: varying noise at fixed bias
ax = axes[1]
for nu, label, c in [
        (0.25, 'Low noise (\u03bd = 0.25) \u2192 steep', '#2166ac'),
        (0.50, 'Medium (\u03bd = 0.50)', '#878787'),
        (1.00, 'High noise (\u03bd = 1.00) \u2192 flat', '#d73027')]:
    sigma = np.sqrt(2) * nu
    p = scipy_norm.cdf((log_r - 0.8) / sigma)
    ax.plot(log_r, p, color=c, lw=2.5, label=label)
ax.axhline(.5, ls='--', c='gray', lw=1)
ax.set_xlabel('log(risky / safe)'); ax.set_ylabel('P(chose risky)')
ax.set_title('Varying noise \u03bd (fixed \u03b4* = 0.80)')
ax.legend(fontsize=8.5); sns.despine(ax=ax)

plt.suptitle('Two parameters of the risky-choice psychometric function', fontsize=12, y=1.02)
plt.tight_layout()
"""),

md(r"""## Fitting psychometric functions — separately per format

Because symbolic and non-symbolic formats produce different noise levels, we fit
`PsychometricModel` **separately** for each format.  This gives us per-subject estimates
of $\nu$ (noise) and $\delta^*$ (indifference point) within each format, which we can
then correlate with each other and with magnitude-task precision.
"""),

code("""\
# Prepare data: x1 = log(safe), x2 = log(risky), split by format
def prep_risk_data(data):
    df = data.reset_index()
    df['x1'] = np.log(df['n1'].astype(float))   # log(safe)
    df['x2'] = np.log(df['n2'].astype(float))   # log(risky)
    return df.set_index([c for c in data.index.names if c in df.columns])

data_sym    = prep_risk_data(data_risk.xs('symbolic',     level='format'))
data_nonsym = prep_risk_data(data_risk.xs('non-symbolic', level='format'))

print(f"Symbolic:      {len(data_sym)} trials, "
      f"{data_sym.index.get_level_values('subject').nunique()} subjects")
print(f"Non-symbolic:  {len(data_nonsym)} trials, "
      f"{data_nonsym.index.get_level_values('subject').nunique()} subjects")
"""),

code("""\
# Fit psychometric models — one per format
from bauer.models import PsychometricModel

model_sym = PsychometricModel(paradigm=data_sym)
model_sym.build_estimation_model(data=data_sym, hierarchical=True)
idata_sym = model_sym.sample(draws=500, tune=500, chains=4, progressbar=False)
print("Symbolic fit done")

model_nonsym = PsychometricModel(paradigm=data_nonsym)
model_nonsym.build_estimation_model(data=data_nonsym, hierarchical=True)
idata_nonsym = model_nonsym.sample(draws=500, tune=500, chains=4, progressbar=False)
print("Non-symbolic fit done")
"""),

code("""\
# Group-level posteriors: compare formats
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ev_threshold = np.log(1/.55)

for ax, param, title in zip(axes, ['nu_mu', 'bias_mu'],
                             ['Noise  \u03bd', 'Indifference point  \u03b4*']):
    sym_vals    = idata_sym.posterior[param].values.ravel()
    nonsym_vals = idata_nonsym.posterior[param].values.ravel()
    az.plot_kde(sym_vals,    label='Symbolic',     plot_kwargs={'color': '#1f78b4', 'lw': 2}, ax=ax)
    az.plot_kde(nonsym_vals, label='Non-symbolic',  plot_kwargs={'color': '#d95f02', 'lw': 2}, ax=ax)
    if param == 'bias_mu':
        ax.axvline(ev_threshold, ls=':', c='gray', lw=1.5, label='Risk-neutral')
    ax.set_title(title); ax.legend(fontsize=9); sns.despine(ax=ax)

plt.suptitle('Group-level posteriors by format', fontsize=12, y=1.02)
plt.tight_layout()
"""),

md(r"""### Format effect

Symbolic (Arabic numeral) payoffs produce **lower noise** and a **lower indifference
point** — i.e. less risk aversion.  This is exactly what the theory predicts: symbolic
numbers are encoded more precisely, so there is less Bayesian shrinkage, and the
perceived advantage of the risky option is less compressed.

### Difference distributions

Overlaying two posteriors can be misleading: even when the marginals overlap substantially,
the **difference** may be clearly non-zero.  This is especially important when the two
quantities are correlated (e.g. because they come from the same participants).  Here the
two formats are fitted as separate models, so the posteriors are independent — but showing
the difference distribution is still clearer than eyeballing overlap.
"""),

code("""\
# Difference distributions: symbolic minus non-symbolic
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, param, title in zip(axes, ['nu_mu', 'bias_mu'],
                             ['Noise difference  \u0394\u03bd', 'Indifference point difference  \u0394\u03b4*']):
    sym_vals    = idata_sym.posterior[param].values.ravel()
    nonsym_vals = idata_nonsym.posterior[param].values.ravel()
    # Sample from both and compute difference
    n = min(len(sym_vals), len(nonsym_vals))
    diff = sym_vals[:n] - nonsym_vals[:n]
    az.plot_kde(diff, plot_kwargs={'color': '#333333', 'lw': 2}, ax=ax)
    ax.axvline(0, ls='--', c='#d73027', lw=1.5, label='No difference')
    pct_below = (diff < 0).mean() * 100
    ax.set_title(title)
    ax.set_xlabel('Symbolic \u2212 Non-symbolic')
    ax.text(0.03, 0.95, f'{pct_below:.0f}% < 0', transform=ax.transAxes,
            va='top', fontsize=10, color='#333333')
    ax.legend(fontsize=9); sns.despine(ax=ax)

plt.suptitle('Format effect: posterior of symbolic \u2212 non-symbolic difference', fontsize=12, y=1.02)
plt.tight_layout()
"""),

md(r"""For both parameters, the difference distribution sits almost entirely below zero:
symbolic payoffs produce lower noise *and* a less risk-averse indifference point.
The `RiskRegressionModel` at the end of this lesson tests this contrast within a single
model, which is the cleanest approach.
"""),

md(r"""## Key result: noise predicts risk aversion (within each format)

We extract each subject's posterior mean for `nu` and `bias`, separately for each format,
and test whether noisier observers are more risk-averse **within** each format.
"""),

code("""\
def extract_subject_params(idata, format_label):
    subjects = idata.posterior['nu'].coords['subject'].values
    nu_flat   = idata.posterior['nu'].values.reshape(-1, len(subjects))
    bias_flat = idata.posterior['bias'].values.reshape(-1, len(subjects))
    df = pd.DataFrame({
        'subject':   subjects,
        'format':    format_label,
        'nu_mean':   nu_flat.mean(0),
        'nu_lo':     np.percentile(nu_flat, 3, 0),
        'nu_hi':     np.percentile(nu_flat, 97, 0),
        'bias_mean': bias_flat.mean(0),
        'bias_lo':   np.percentile(bias_flat, 3, 0),
        'bias_hi':   np.percentile(bias_flat, 97, 0),
    })
    df['prec']    = 1 / df['nu_mean']
    df['prec_lo'] = 1 / df['nu_hi']
    df['prec_hi'] = 1 / df['nu_lo']
    return df

df_sym_params    = extract_subject_params(idata_sym,    'symbolic')
df_nonsym_params = extract_subject_params(idata_nonsym, 'non-symbolic')

for label, df in [('Symbolic', df_sym_params), ('Non-symbolic', df_nonsym_params)]:
    rho, p = spearmanr(df['nu_mean'], df['bias_mean'])
    print(f"{label:15s}  noise-bias \u03c1 = {rho:.3f} (p = {p:.4f})")
"""),

code("""\
def scatter_hdi(ax, x, y, xerr, yerr, color, xlabel, ylabel, title,
                hline=None):
    ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                fmt='o', ms=4, alpha=.55, elinewidth=.7, capsize=2,
                color=color, ecolor=color)
    rho, p = spearmanr(x, y)
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, m*xs + b, '--', color=color, lw=1.5, alpha=.8,
            label=f'\u03c1 = {rho:.2f} (p = {p:.3f})')
    if hline is not None:
        ax.axhline(hline, ls=':', c='gray', lw=1.5, label='Risk-neutral')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=9); sns.despine(ax=ax)

ev_threshold = np.log(1/.55)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, df, label, color in zip(
        axes,
        [df_nonsym_params, df_sym_params],
        ['Non-symbolic (coin clouds)', 'Symbolic (Arabic numerals)'],
        ['#d95f02', '#1f78b4']):
    scatter_hdi(
        ax,
        x    = df['nu_mean'],
        y    = df['bias_mean'],
        xerr = np.array([df['nu_mean'] - df['nu_lo'], df['nu_hi'] - df['nu_mean']]),
        yerr = np.array([df['bias_mean'] - df['bias_lo'], df['bias_hi'] - df['bias_mean']]),
        color  = color,
        xlabel = 'Noise  \u03bd',
        ylabel = 'Indifference point  \u03b4*',
        title  = label,
        hline  = ev_threshold,
    )

plt.suptitle('Within-format: noise predicts risk aversion (bars = 94\u202f% HDI)',
             fontsize=13, y=1.02)
plt.tight_layout()
"""),


md(r"""## Interpreting the indifference point: from log-space to intuition

The `bias` parameter lives in **log-ratio space**, which is not immediately intuitive.
Three reparameterisations make it more concrete:

### 1. Indifference ratio: "how much larger must the risky option be?"

The indifference point is $\delta^* = -\text{bias}$ in log-ratio space.  Exponentiating
gives the **indifference ratio**:

$$R^* = e^{\delta^*} = e^{-\text{bias}}$$

This is the factor by which the risky payoff must exceed the safe payoff for the observer
to be indifferent.  A risk-neutral observer (with $p_\text{risky} = 0.55$) has
$R^* = 1/0.55 \approx 1.82$ — the risky option must be about **82 % larger**.  A
risk-averse observer requires an even larger premium.

### 2. Risk-neutral probability (RNP)

The **RNP** answers: "at what winning probability would a risk-neutral agent make the same
choices as this observer?"

$$\text{RNP} = e^{\text{bias}}$$

- RNP = 0.55 → risk-neutral (the observer acts as if the gamble wins at its true rate)
- RNP < 0.55 → risk-averse (acts as if the gamble wins *less* often than it actually does)
- RNP > 0.55 → risk-seeking

The RNP is simply the reciprocal of the indifference ratio scaled by the actual probability:
$\text{RNP} = p_\text{risky} / R^* \times (1/p_\text{safe})$.  But $e^{\text{bias}}$ is
the most direct computation.

### 3. Certainty equivalent discount

How many cents on the euro does the observer implicitly discount the risky option?
$\text{CE discount} = 1 - \text{RNP} / p_\text{risky} = 1 - e^{\text{bias}} / 0.55$.
A discount of 20 % means the observer treats every risky euro as worth only 80 cents.
"""),

code("""\
# Posterior distributions of the reparameterised group-level bias
bias_sym    = idata_sym.posterior['bias_mu'].values.ravel()
bias_nonsym = idata_nonsym.posterior['bias_mu'].values.ravel()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Indifference ratio R* = exp(-bias)
ax = axes[0]
for vals, label, c in [(bias_nonsym, 'Non-symbolic', '#d95f02'),
                        (bias_sym,    'Symbolic',     '#1f78b4')]:
    ratio = np.exp(-vals)
    az.plot_kde(ratio, label=label, plot_kwargs={'color': c, 'lw': 2}, ax=ax)
ax.axvline(1/0.55, ls=':', c='gray', lw=1.5, label='Risk-neutral (1/0.55)')
ax.set_xlabel('Indifference ratio  R* = exp(-bias)')
ax.set_title('How much larger must risky be?')
ax.legend(fontsize=8); sns.despine(ax=ax)

# 2. RNP = exp(bias)
ax = axes[1]
for vals, label, c in [(bias_nonsym, 'Non-symbolic', '#d95f02'),
                        (bias_sym,    'Symbolic',     '#1f78b4')]:
    rnp = np.exp(vals)
    az.plot_kde(rnp, label=label, plot_kwargs={'color': c, 'lw': 2}, ax=ax)
ax.axvline(0.55, ls=':', c='gray', lw=1.5, label='Risk-neutral (0.55)')
ax.set_xlabel('Risk-neutral probability  (RNP)')
ax.set_title('Implied winning probability')
ax.legend(fontsize=8); sns.despine(ax=ax)

# 3. CE discount = 1 - RNP/0.55
ax = axes[2]
for vals, label, c in [(bias_nonsym, 'Non-symbolic', '#d95f02'),
                        (bias_sym,    'Symbolic',     '#1f78b4')]:
    rnp = np.exp(vals)
    discount = 1 - rnp / 0.55
    az.plot_kde(discount, label=label, plot_kwargs={'color': c, 'lw': 2}, ax=ax)
ax.axvline(0, ls=':', c='gray', lw=1.5, label='Risk-neutral')
ax.set_xlabel('CE discount  (1 - RNP/0.55)')
ax.set_title('Certainty-equivalent discount')
ax.legend(fontsize=8); sns.despine(ax=ax)

plt.suptitle('Group-level posteriors: three views of risk aversion', fontsize=12, y=1.02)
plt.tight_layout()
"""),

md(r"""## Bonus: RiskRegressionModel for the format effect

Instead of fitting two separate models, bauer's `RiskRegressionModel` lets you test the
format effect *within a single model* using a patsy formula.  This gives a direct posterior
on the format contrast — cleaner than comparing two separate fits.
"""),

code("""\
from bauer.models import RiskRegressionModel

data_reg = data_risk.reset_index(level='format')
model_reg = RiskRegressionModel(
    paradigm=data_reg,
    regressors={'evidence_sd': 'C(format)'},
    prior_estimate='klw',
    fit_seperate_evidence_sd=False,
)
model_reg.build_estimation_model(data=data_reg, hierarchical=True)
idata_reg = model_reg.sample(draws=500, tune=500, chains=4, progressbar=False)

# Posterior of evidence_sd at each format condition
conditions = pd.DataFrame({'format': ['symbolic', 'non-symbolic']})
pars_cond  = model_reg.get_conditionwise_parameters(idata_reg, conditions, group=True)

sd_sym    = pars_cond.loc['evidence_sd']['symbolic'].values
sd_nonsym = pars_cond.loc['evidence_sd']['non-symbolic'].values

fig, ax = plt.subplots(figsize=(6, 4))
az.plot_kde(sd_sym,    label='Symbolic',     plot_kwargs={'color': '#1f78b4', 'lw': 2}, ax=ax)
az.plot_kde(sd_nonsym, label='Non-symbolic',  plot_kwargs={'color': '#d95f02', 'lw': 2}, ax=ax)
ax.set_xlabel('evidence_sd  (group level)')
ax.set_ylabel('Posterior density')
ax.set_title('RiskRegressionModel: format effect on noise')
ax.legend(); sns.despine(); plt.tight_layout()
"""),

md(r"""## Why noise produces risk aversion: the KLW mechanism

The correlation between noise and risk aversion is not a statistical accident — it has a
mechanistic explanation.  The **KLW model** (Khaw, Li & Woodford, 2021) shows that a
Bayesian observer with noisy magnitude representations will *systematically undervalue*
risky prospects:

1. Both the safe and risky payoffs are perceived with noise $\nu$ on the log scale.
2. A Bayesian prior pulls both percepts toward the mean of the payoff distribution.
3. Because the risky payoff is typically *larger* than the safe payoff, it gets pulled
   *down* more (toward the mean) than the safe payoff gets pulled *up*.
4. This asymmetric compression shrinks the perceived advantage of the risky option,
   shifting the indifference point rightward — i.e. producing risk aversion.

The amount of shrinkage is controlled by the ratio $\nu^2 / \sigma_0^2$ (noise vs.
prior width).  More noise $\rightarrow$ more shrinkage $\rightarrow$ more risk aversion.
This is exactly the correlation we see in the scatter plots above.

Critically, risk aversion here is not a preference — it is a **perceptual distortion**.
The observer is doing the best they can with noisy information, and the Bayesian-optimal
strategy happens to look risk-averse from the outside.

## Summary

1. A **psychometric function** fitted to risky choices gives two parameters per subject:
   noise ($\nu$) and indifference point ($\delta^*$).
2. **Format matters**: symbolic payoffs produce lower noise and less risk aversion than
   non-symbolic coin clouds.
3. **Within each format**: noisier subjects have higher $\delta^*$ (more risk-averse).
4. **RiskRegressionModel** provides a principled single-model test of the format effect.
5. The **KLW model** explains *why*: Bayesian shrinkage under noise compresses the
   perceived advantage of risky options.

In [Lesson 3](lesson3.ipynb) we move to a richer dataset (de Hollander et al., 2024, bioRxiv) where
**presentation order** is randomised, allowing bauer's `RiskModel` with separate
$\nu_1, \nu_2$ to capture striking order $\times$ stake-size interactions.
"""),


]

write_if_changed(nb2, 'lesson2.ipynb')



# ─────────────────────────────────────────────────────────────────────────────
# Lesson 3 — de Hollander et al. (2024, bioRxiv): EU vs KLW vs full prior
# ─────────────────────────────────────────────────────────────────────────────

nb3 = nbf.v4.new_notebook()
nb3.cells = [

md(r"""# Lesson 3: Stake effects and presentation order — de Hollander et al. (2024, bioRxiv)

## Background

De Hollander et al. (2024, *Nature Human Behaviour*) tested whether perceptual noise
during the encoding of numerical magnitudes explains risk aversion and its interaction
with presentation order.  The key design feature: the **order** in which the safe and
risky options are presented is randomised across trials, allowing the model to disentangle
$\nu_1$ (noise on the first-presented option) from $\nu_2$ (noise on the second-presented
option).

This produces a distinctive **presentation-order × stake-size interaction**: when the
safe option comes first, high safe stakes are compressed downward more strongly by the
prior → safe looks less attractive → the observer becomes more risk-seeking for high
stakes.  When the risky option comes first, the same mechanism operates on the risky
stakes → risk aversion for high stakes.

Standard models (EU, KLW with a shared noise) **cannot** capture this asymmetry.

### Models compared

| Model | Class | Key parameters |
|-------|-------|----------------|
| Expected Utility (EU) | `ExpectedUtilityRiskModel` | `alpha`, `sigma` |
| KLW | `RiskModel(prior_estimate='klw', fit_seperate_evidence_sd=False)` | `evidence_sd`, `prior_sd` (shared) |
| PMCM | `RiskModel(prior_estimate='full', fit_seperate_evidence_sd=True)` | `n1_evidence_sd`, `n2_evidence_sd`, `risky/safe_prior_mu/sd` |

We fit all three on both the **dot-cloud** (fMRI sessions 3T+7T) and the **symbolic**
(Arabic numerals) datasets, then compare posterior predictives against the interaction
pattern.
"""),

code("""\
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from bauer.utils.data import load_dehollander2024
from bauer.models import ExpectedUtilityRiskModel, RiskModel

# Load both fMRI sessions and combine
df_dot = load_dehollander2024(task='dotcloud', sessions=['3t2', '7t2'])
df_sym = load_dehollander2024(task='symbolic')

print(f"Dot clouds  — subjects: {df_dot.index.get_level_values('subject').nunique()}, "
      f"trials: {len(df_dot)}")
print(f"Symbolic    — subjects: {df_sym.index.get_level_values('subject').nunique()}, "
      f"trials: {len(df_sym)}")
df_dot.head()
"""),

code("""\
def prep_df(df):
    \"\"\"Add log_ratio, chose_risky, n_safe, order flag, and binned columns.
    Returns (df, stake_labels) where stake_labels lists the three stake-bin names
    with concrete magnitude ranges, e.g. ['Low (5–12)', 'Mid (12–24)', 'High (24–48)'].
    \"\"\"
    df = df.reset_index()
    risky_first = df['p1'] == 0.55
    df['log_ratio']   = np.log(
        np.where(risky_first, df['n1'], df['n2']) /
        np.where(risky_first, df['n2'], df['n1']))
    df['chose_risky'] = np.where(risky_first, ~df['choice'], df['choice'])
    n_safe            = np.where(risky_first, df['n2'], df['n1'])
    df['n_safe']      = n_safe
    df['risky_first'] = risky_first
    df['order']       = np.where(risky_first, 'Risky first', 'Safe first')
    df['log_ratio_bin'] = (pd.cut(df['log_ratio'], bins=10)
                             .map(lambda x: x.mid).astype(float))
    _, bins = pd.qcut(n_safe, q=3, retbins=True, duplicates='drop')
    stake_labels = [
        f'Low ({bins[0]:.0f}–{bins[1]:.0f})',
        f'Mid ({bins[1]:.0f}–{bins[2]:.0f})',
        f'High ({bins[2]:.0f}–{bins[3]:.0f})',
    ]
    df['n_safe_bin'] = pd.qcut(n_safe, q=3, labels=stake_labels, duplicates='drop')
    return df, stake_labels

df_dot_p, dot_stake_labels = prep_df(df_dot)
df_sym_p, sym_stake_labels = prep_df(df_sym)

# Sequential (light→dark) palette for stake sizes — hue in existing order\u00d7stake plots
def make_stake_pal(labels):
    cols = sns.color_palette('YlOrRd', len(labels))   # yellow → orange → red
    return dict(zip(labels, cols))

dot_stake_pal = make_stake_pal(dot_stake_labels)
sym_stake_pal = make_stake_pal(sym_stake_labels)

# Order palette: standard seaborn blue/orange
order_pal = {'Safe first': sns.color_palette()[0],    # blue
             'Risky first': sns.color_palette()[1]}   # orange
"""),

md(r"""## Presentation-order x stake-size interaction

Each panel shows P(chose risky) as a function of the log risky/safe magnitude ratio,
split by safe-option stake tertile.  The left column shows trials where the risky option
came first; the right column shows trials where the safe option came first.

The dashed vertical line marks the risk-neutral indifference point log(1/0.55).
"""),

code("""\
def plot_interaction(df_p, axes_row, task_label, stake_labels, stake_pal):
    \"\"\"Columns = order condition, hue = safe-stake tertile.\"\"\"
    for ax, order_val in zip(axes_row, ['Risky first', 'Safe first']):
        sub  = df_p[df_p['order'] == order_val]
        subj = (sub.groupby(['subject', 'log_ratio_bin', 'n_safe_bin'])['chose_risky']
                   .mean().reset_index())
        subj = subj[subj.groupby(['log_ratio_bin', 'n_safe_bin'])
                        ['subject'].transform('count') >= 3]
        sns.lineplot(data=subj, x='log_ratio_bin', y='chose_risky',
                     hue='n_safe_bin', style='n_safe_bin',
                     hue_order=stake_labels, style_order=stake_labels,
                     palette=stake_pal, markers=True, dashes=False,
                     errorbar='se', ax=ax)
        ax.axhline(.5,            ls='--', c='gray', lw=1)
        ax.axvline(np.log(1/.55), ls='--', c='#333333', lw=1.5, label='Risk-neutral')
        ax.set_ylim(-.05, 1.05)
        ax.set_xlabel('log(risky / safe)')
        ax.set_ylabel('P(chose risky)')
        ax.set_title(f'{task_label} — {order_val}', fontsize=10)
        ax.legend(title='Safe stake', fontsize=8)
        sns.despine(ax=ax)


def plot_interaction_by_stake(df_p, axes_row, task_label, stake_labels):
    \"\"\"Columns = safe-stake tertile, hue = order condition (orange/blue).\"\"\"
    for ax, sbin in zip(axes_row, stake_labels):
        sub  = df_p[df_p['n_safe_bin'] == sbin]
        subj = (sub.groupby(['subject', 'log_ratio_bin', 'order'])['chose_risky']
                   .mean().reset_index())
        subj = subj[subj.groupby(['log_ratio_bin', 'order'])
                        ['subject'].transform('count') >= 3]
        sns.lineplot(data=subj, x='log_ratio_bin', y='chose_risky',
                     hue='order', style='order',
                     hue_order=['Risky first', 'Safe first'],
                     palette=order_pal, markers=True, dashes=False,
                     errorbar='se', ax=ax)
        ax.axhline(.5,            ls='--', c='gray', lw=1)
        ax.axvline(np.log(1/.55), ls='--', c='#333333', lw=1.5)
        ax.set_ylim(-.05, 1.05)
        ax.set_xlabel('log(risky / safe)')
        ax.set_ylabel('P(chose risky)')
        ax.set_title(f'{task_label} — {sbin}', fontsize=10)
        ax.legend(title='Order', fontsize=8)
        sns.despine(ax=ax)


fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
plot_interaction(df_dot_p, axes[0], 'Dot clouds (3T + 7T)', dot_stake_labels, dot_stake_pal)
plot_interaction(df_sym_p, axes[1], 'Symbolic (Arabic numerals)', sym_stake_labels, sym_stake_pal)
plt.suptitle('Presentation-order \u00d7 safe-stake interaction  (hue = stake size)', fontsize=13, y=1.01)
plt.tight_layout()
"""),

md("""### Alternative view: stake size × order (hue = order)

The same data shown with columns = stake tertile and hue = presentation order.
Orange = risky option presented first; blue = safe option presented first.
"""),

code("""\
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
plot_interaction_by_stake(df_dot_p, axes[0], 'Dot clouds (3T + 7T)', dot_stake_labels)
plot_interaction_by_stake(df_sym_p, axes[1], 'Symbolic (Arabic numerals)', sym_stake_labels)
plt.suptitle('Stake-size \u00d7 order interaction  (hue = order)', fontsize=13, y=1.01)
plt.tight_layout()
"""),

md("""## Fit three models — dot-cloud data

Hierarchical MCMC, 100 draws / 100 tune / 2 chains.  We store log-likelihoods
(`log_likelihood=True`) for ELPD model comparison later.
"""),

code("""\
# ── 1. Expected Utility ──────────────────────────────────────────────────────
model_eu = ExpectedUtilityRiskModel(paradigm=df_dot)
model_eu.build_estimation_model(data=df_dot, hierarchical=True, save_p_choice=True)
idata_eu = model_eu.sample(draws=100, tune=100, chains=2, progressbar=False,
                            idata_kwargs={'log_likelihood': True})
"""),

code("""\
# ── 2. KLW (shared noise, shared prior) ─────────────────────────────────────
model_klw = RiskModel(paradigm=df_dot, prior_estimate='klw',
                      fit_seperate_evidence_sd=False)
model_klw.build_estimation_model(data=df_dot, hierarchical=True, save_p_choice=True)
idata_klw = model_klw.sample(draws=100, tune=100, chains=2, progressbar=False,
                              idata_kwargs={'log_likelihood': True})
"""),

code("""\
# ── 3. PMCM (separate noise + separate priors) ────────────────────
model_full = RiskModel(paradigm=df_dot, prior_estimate='full',
                       fit_seperate_evidence_sd=True)
model_full.build_estimation_model(data=df_dot, hierarchical=True, save_p_choice=True)
idata_full = model_full.sample(draws=100, tune=100, chains=2, progressbar=False,
                                idata_kwargs={'log_likelihood': True})
"""),

md("""## Posterior predictives — dot-cloud data

Dots = observed group average; line + shading = model mean and 95 % posterior interval.
"""),

code("""\
from bauer.utils import summarize_ppc_group

def add_model_ppc(df_orig, df_prepped, model, idata, model_name, stake_labels):
    \"\"\"Condition-level PPC via summarize_ppc_group (two-step subject averaging).\"\"\"
    ppc_df  = model.ppc(df_orig, idata, var_names=['ll_bernoulli'])
    ppc_ll  = ppc_df.xs('ll_bernoulli', level='variable')
    sample_cols = ppc_ll.columns.tolist()

    ppc_flat = ppc_ll.reset_index()
    risky_first = (ppc_flat['p1'] == 0.55)
    ppc_flat[sample_cols] = np.where(
        risky_first.values[:, None],
        1 - ppc_flat[sample_cols].values,
        ppc_flat[sample_cols].values
    )
    ppc_flat['order'] = np.where(risky_first, 'Risky first', 'Safe first')
    log_ratio = np.log(
        np.where(risky_first, ppc_flat['n1'], ppc_flat['n2']) /
        np.where(risky_first, ppc_flat['n2'], ppc_flat['n1']))
    ppc_flat['log_ratio_bin'] = (pd.cut(pd.Series(log_ratio), bins=10)
                                   .map(lambda x: x.mid).astype(float).values)
    n_safe = np.where(risky_first, ppc_flat['n2'], ppc_flat['n1'])
    ppc_flat['n_safe_bin'] = pd.qcut(n_safe, q=3, labels=stake_labels, duplicates='drop')

    result = summarize_ppc_group(
        ppc_flat,
        condition_cols=['order', 'n_safe_bin', 'log_ratio_bin']
    )
    return result.rename(columns={'p_predicted': 'p_mean',
                                   'hdi025': 'p_lo', 'hdi975': 'p_hi'}).reset_index()


def plot_ppc_interaction(df_pred, df_obs, model_name, axes_row, stake_labels, stake_pal):
    \"\"\"Columns = order; hue = stake tertile (sequential palette).\"\"\"
    for ax, order_val in zip(axes_row, ['Risky first', 'Safe first']):
        pred = df_pred[df_pred['order'] == order_val]
        obs  = df_obs[df_obs['order']   == order_val]
        for sbin in stake_labels:
            p = pred[pred['n_safe_bin'] == sbin]
            o = obs[obs['n_safe_bin']   == sbin]
            if len(o) == 0:
                continue
            c = stake_pal[sbin]
            ax.fill_between(p['log_ratio_bin'], p['p_lo'], p['p_hi'], color=c, alpha=.20)
            ax.plot(p['log_ratio_bin'], p['p_mean'], color=c, lw=2, label=sbin)
            ax.scatter(o['log_ratio_bin'], o['chose_risky'],
                       color=c, s=25, zorder=5, alpha=.85)
        ax.axhline(.5,            ls='--', c='gray', lw=1)
        ax.axvline(np.log(1/.55), ls='--', c='#333333', lw=1.5)
        ax.set_ylim(-.05, 1.05)
        ax.set_title(f'{model_name} — {order_val}', fontsize=9)
        ax.set_xlabel('log(risky / safe)'); ax.set_ylabel('P(chose risky)')
        ax.legend(title='Safe stake', fontsize=7, loc='upper left')
        sns.despine(ax=ax)


def plot_ppc_by_stake(df_pred, df_obs, model_name, axes_row, stake_labels):
    \"\"\"Columns = stake tertile; hue = order (orange/blue).\"\"\"
    for ax, sbin in zip(axes_row, stake_labels):
        pred = df_pred[df_pred['n_safe_bin'] == sbin]
        obs  = df_obs[df_obs['n_safe_bin']   == sbin]
        for order_val in ['Risky first', 'Safe first']:
            p = pred[pred['order'] == order_val]
            o = obs[obs['order']   == order_val]
            if len(o) == 0:
                continue
            c = order_pal[order_val]
            ax.fill_between(p['log_ratio_bin'], p['p_lo'], p['p_hi'], color=c, alpha=.20)
            ax.plot(p['log_ratio_bin'], p['p_mean'], color=c, lw=2, label=order_val)
            ax.scatter(o['log_ratio_bin'], o['chose_risky'],
                       color=c, s=25, zorder=5, alpha=.85)
        ax.axhline(.5,            ls='--', c='gray', lw=1)
        ax.axvline(np.log(1/.55), ls='--', c='#333333', lw=1.5)
        ax.set_ylim(-.05, 1.05)
        ax.set_title(f'{model_name} — {sbin}', fontsize=9)
        ax.set_xlabel('log(risky / safe)'); ax.set_ylabel('P(chose risky)')
        ax.legend(title='Order', fontsize=7, loc='upper left')
        sns.despine(ax=ax)


# Observed: two-step average matching the PPC computation
obs_dot = (df_dot_p
           .groupby(['subject', 'order', 'n_safe_bin', 'log_ratio_bin'])['chose_risky']
           .mean()
           .groupby(['order', 'n_safe_bin', 'log_ratio_bin']).mean()
           .reset_index())

fig, axes = plt.subplots(3, 2, figsize=(12, 13), sharey=True)
ppc_dot = {}
for (mdl, idat, name), row in zip(
        [(model_eu,   idata_eu,   'EU'),
         (model_klw,  idata_klw,  'KLW'),
         (model_full, idata_full, 'PMCM')],
        axes):
    ppc_dot[name] = add_model_ppc(df_dot, df_dot_p, mdl, idat, name, dot_stake_labels)
    plot_ppc_interaction(ppc_dot[name], obs_dot, name, row, dot_stake_labels, dot_stake_pal)

plt.suptitle('Posterior predictive checks — dot-cloud data  (dots = observed, shading = 95 % CI)',
             fontsize=11, y=1.01)
plt.tight_layout()
"""),

md("""### Alternative PPC view: columns = stake size, hue = order
"""),

code("""\
fig, axes = plt.subplots(3, 3, figsize=(15, 13), sharey=True)
for (name, df_pred), row in zip(ppc_dot.items(), axes):
    plot_ppc_by_stake(df_pred, obs_dot, name, row, dot_stake_labels)

plt.suptitle('PPC — dot-cloud  (columns = stake tertile, hue = order)',
             fontsize=11, y=1.01)
plt.tight_layout()
"""),

md("""## Fit three models — symbolic data
"""),

code("""\
# ── 1. EU ────────────────────────────────────────────────────────────────────
model_eu_sym = ExpectedUtilityRiskModel(paradigm=df_sym)
model_eu_sym.build_estimation_model(data=df_sym, hierarchical=True, save_p_choice=True)
idata_eu_sym = model_eu_sym.sample(draws=100, tune=100, chains=2, progressbar=False,
                                        idata_kwargs={'log_likelihood': True})
"""),

code("""\
# ── 2. KLW ───────────────────────────────────────────────────────────────────
model_klw_sym = RiskModel(paradigm=df_sym, prior_estimate='klw',
                           fit_seperate_evidence_sd=False)
model_klw_sym.build_estimation_model(data=df_sym, hierarchical=True, save_p_choice=True)
idata_klw_sym = model_klw_sym.sample(draws=100, tune=100, chains=2, progressbar=False,
                                          idata_kwargs={'log_likelihood': True})
"""),

code("""\
# ── 3. PMCM ────────────────────────────────────────────────────────
model_full_sym = RiskModel(paradigm=df_sym, prior_estimate='full',
                            fit_seperate_evidence_sd=True)
model_full_sym.build_estimation_model(data=df_sym, hierarchical=True, save_p_choice=True)
idata_full_sym = model_full_sym.sample(draws=100, tune=100, chains=2, progressbar=False,
                                            idata_kwargs={'log_likelihood': True})
"""),

md("""## Posterior predictives — symbolic data
"""),

code("""\
obs_sym = (df_sym_p
           .groupby(['subject', 'order', 'n_safe_bin', 'log_ratio_bin'])['chose_risky']
           .mean()
           .groupby(['order', 'n_safe_bin', 'log_ratio_bin']).mean()
           .reset_index())

fig, axes = plt.subplots(3, 2, figsize=(12, 13), sharey=True)
ppc_sym = {}
for (mdl, idat, name), row in zip(
        [(model_eu_sym,   idata_eu_sym,   'EU'),
         (model_klw_sym,  idata_klw_sym,  'KLW'),
         (model_full_sym, idata_full_sym, 'PMCM')],
        axes):
    ppc_sym[name] = add_model_ppc(df_sym, df_sym_p, mdl, idat, name, sym_stake_labels)
    plot_ppc_interaction(ppc_sym[name], obs_sym, name, row, sym_stake_labels, sym_stake_pal)

plt.suptitle('Posterior predictive checks — symbolic data  (dots = observed, shading = 95 % CI)',
             fontsize=11, y=1.01)
plt.tight_layout()
"""),

md("""### Alternative PPC view: columns = stake size, hue = order (symbolic)
"""),

code("""\
fig, axes = plt.subplots(3, 3, figsize=(15, 13), sharey=True)
for (name, df_pred), row in zip(ppc_sym.items(), axes):
    plot_ppc_by_stake(df_pred, obs_sym, name, row, sym_stake_labels)

plt.suptitle('PPC — symbolic  (columns = stake tertile, hue = order)',
             fontsize=11, y=1.01)
plt.tight_layout()
"""),

md(r"""## Parameter interpretation: $\nu_1$ vs $\nu_2$

bauer's `get_subject_posterior_df` creates a tidy summary DataFrame, and
`plot_subjectwise_pointplot` maps it onto a FacetGrid — one panel per parameter per task.
"""),

code("""\
from bauer.utils import get_subject_posterior_df, plot_subjectwise_pointplot

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for row_axes, (idat, task_label) in zip(axes,
        [(idata_full,     'Dot clouds'),
         (idata_full_sym, 'Symbolic')]):

    df_post = get_subject_posterior_df(idat, ['n1_evidence_sd', 'n2_evidence_sd'])

    # Left: ν₁ sorted pointplot (direct call with ax=)
    d1 = df_post[df_post['parameter'] == 'n1_evidence_sd'].reset_index(drop=True)
    plot_subjectwise_pointplot(d1, color='#d73027', ax=row_axes[0])
    row_axes[0].set_title(f'{task_label}: \u03bd\u2081 (first option)')
    row_axes[0].set_ylabel('Noise  \u03bd\u2081')

    # Middle: ν₂ sorted pointplot
    d2 = df_post[df_post['parameter'] == 'n2_evidence_sd'].reset_index(drop=True)
    plot_subjectwise_pointplot(d2, color='#2166ac', ax=row_axes[1])
    row_axes[1].set_title(f'{task_label}: \u03bd\u2082 (second option)')
    row_axes[1].set_ylabel('Noise  \u03bd\u2082')

    # Right: ν₁ vs ν₂ scatter
    n_subj = idat.posterior['n1_evidence_sd'].shape[-1]
    s1 = idat.posterior['n1_evidence_sd'].values.reshape(-1, n_subj)
    s2 = idat.posterior['n2_evidence_sd'].values.reshape(-1, n_subj)
    nu1_mean, nu2_mean = s1.mean(0), s2.mean(0)
    nu1_lo, nu1_hi = np.percentile(s1, [3, 97], axis=0)
    nu2_lo, nu2_hi = np.percentile(s2, [3, 97], axis=0)
    lim = max(nu1_hi.max(), nu2_hi.max()) * 1.1
    row_axes[2].errorbar(nu2_mean, nu1_mean,
                xerr=[nu2_mean - nu2_lo, nu2_hi - nu2_mean],
                yerr=[nu1_mean - nu1_lo, nu1_hi - nu1_mean],
                fmt='o', ms=4, alpha=.6, elinewidth=0.6, capsize=2,
                color='#2166ac', ecolor='#aec7e8', zorder=3)
    row_axes[2].plot([0, lim], [0, lim], 'k--', lw=1, label='\u03bd\u2081 = \u03bd\u2082')
    row_axes[2].set_xlim(0, lim); row_axes[2].set_ylim(0, lim)
    row_axes[2].set_xlabel('\u03bd\u2082'); row_axes[2].set_ylabel('\u03bd\u2081')
    row_axes[2].set_title(f'{task_label}: \u03bd\u2081 vs \u03bd\u2082')
    row_axes[2].legend(fontsize=9); sns.despine(ax=row_axes[2])

plt.suptitle('Subject-level noise estimates from PMCM (error bars = 94 % HDI)',
             fontsize=12, y=1.02)
plt.tight_layout()
"""),

md(r"""## Individual differences in noise asymmetry

The group average $\nu_1 > \nu_2$ suggests a **memory effect**: the first-presented option
is noisier because it must be held in working memory.  But this is only part of the story.

In the **symbolic** task, the picture is more nuanced.  Not every participant shows a memory
effect — some appear to show an **attentional primacy** effect where they focus *more* on
the first option ($\nu_1 < \nu_2$).  The distribution of $\nu_1 - \nu_2$ across
participants reveals this heterogeneity.
"""),

code("""\
# ── ν₁ − ν₂ difference per subject ──────────────────────────────────────────
from bauer.utils.math import softplus_np

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (idat, task_label) in zip(axes,
        [(idata_full, 'Dot clouds'), (idata_full_sym, 'Symbolic')]):
    n_subj = idat.posterior['n1_evidence_sd'].shape[-1]
    s1 = softplus_np(idat.posterior['n1_evidence_sd'].values.reshape(-1, n_subj))
    s2 = softplus_np(idat.posterior['n2_evidence_sd'].values.reshape(-1, n_subj))
    diff = s1 - s2   # positive = memory effect (ν₁ > ν₂)

    diff_mean = diff.mean(0)
    diff_lo   = np.percentile(diff, 3, 0)
    diff_hi   = np.percentile(diff, 97, 0)

    sort_idx = np.argsort(diff_mean)
    x = np.arange(n_subj)

    ax.errorbar(x, diff_mean[sort_idx],
                yerr=[diff_mean[sort_idx] - diff_lo[sort_idx],
                      diff_hi[sort_idx] - diff_mean[sort_idx]],
                fmt='o', ms=4, elinewidth=0.7, capsize=1.5, alpha=.7,
                color='#4393c3', ecolor='#aec7e8')
    ax.axhline(0, ls='--', c='#d73027', lw=1.5, label='\u03bd\u2081 = \u03bd\u2082')
    ax.set_xlabel('Subject (sorted by \u03bd\u2081 \u2212 \u03bd\u2082)')
    ax.set_ylabel('\u03bd\u2081 \u2212 \u03bd\u2082')
    ax.set_title(f'{task_label}')
    n_mem = (diff_lo[sort_idx] > 0).sum()
    n_att = (diff_hi[sort_idx] < 0).sum()
    ax.text(0.02, 0.98, f'Memory effect (\u03bd\u2081>\u03bd\u2082): {n_mem}\\nAttention primacy (\u03bd\u2081<\u03bd\u2082): {n_att}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=.8))
    ax.legend(fontsize=9); sns.despine(ax=ax)

plt.suptitle('Individual noise asymmetry: \u03bd\u2081 \u2212 \u03bd\u2082 (94\u202f% HDI per subject)',
             fontsize=12, y=1.02)
plt.tight_layout()
"""),

md(r"""### Interpreting the heterogeneity

For **dot clouds**, most participants show $\nu_1 > \nu_2$ (memory effect) — consistent with
the sequential presentation degrading the first option in working memory.

For **symbolic (Arabic numeral)** stimuli, the pattern is more mixed:
- Some participants still show a memory effect ($\nu_1 > \nu_2$)
- Others show the opposite: $\nu_1 < \nu_2$ — as if they allocate more **attention** to the
  first-presented option and less to the second

This suggests that with symbolic stimuli (which are faster to encode than dot clouds), the
bottleneck shifts from working memory to **attentional allocation**, and different participants
adopt different strategies.
"""),

md(r"""## Subject-level PPCs: do the models capture individual strategies?

Group-level PPCs can look fine while hiding poor fits for individual participants.
We pick three subjects — one with a strong memory effect ($\nu_1 \gg \nu_2$), one
balanced, and one with an attentional-primacy effect ($\nu_1 < \nu_2$) — and show
their individual posterior predictives under the three models.
"""),

code("""\
# Identify 5 most extreme subjects in each direction (symbolic task)
from bauer.utils.math import softplus_np
n_subj_sym = idata_full_sym.posterior['n1_evidence_sd'].shape[-1]
s1_sym = softplus_np(idata_full_sym.posterior['n1_evidence_sd'].values.reshape(-1, n_subj_sym))
s2_sym = softplus_np(idata_full_sym.posterior['n2_evidence_sd'].values.reshape(-1, n_subj_sym))
diff_sym = (s1_sym - s2_sym).mean(0)
sym_subjects = idata_full_sym.posterior['n1_evidence_sd'].coords['subject'].values

rank = np.argsort(diff_sym)
top5_memory    = sym_subjects[rank[-5:]]   # highest \u03bd\u2081 - \u03bd\u2082
top5_attention = sym_subjects[rank[:5]]    # lowest \u03bd\u2081 - \u03bd\u2082
print(f"Memory-effect subjects:    {top5_memory}  (mean \u0394\u03bd = {diff_sym[rank[-5:]].mean():.3f})")
print(f"Attention-primacy subjects: {top5_attention}  (mean \u0394\u03bd = {diff_sym[rank[:5]].mean():.3f})")
"""),

code("""\
# Subject-group PPCs using model probability (p), not binary ll_bernoulli
# Average P(chose risky) over 5 extreme subjects per group

def subgroup_ppc(model, idata, df_raw, df_prepped, subject_ids, group_label):
    # Get model-predicted P(chose option 2) for all subjects
    ppc_df = model.ppc(df_raw, idata, var_names=['p'])
    ppc_p = ppc_df.xs('p', level='variable')
    sample_cols = ppc_p.columns.tolist()

    rows = []
    for subj in subject_ids:
        ppc_subj = ppc_p.xs(subj, level='subject').reset_index()
        obs_subj = df_raw.xs(subj, level='subject').reset_index()

        risky_first = ppc_subj['p1'] == 0.55
        # P(chose risky): flip when risky is option 1
        pred_risky = ppc_subj[sample_cols].values.copy()
        pred_risky[risky_first] = 1 - pred_risky[risky_first]

        obs_choice = obs_subj['choice'].values.astype(float)
        obs_risky = np.where(risky_first, 1 - obs_choice, obs_choice)

        order = np.where(risky_first, 'Risky first', 'Safe first')
        n_safe = np.where(risky_first, obs_subj['n2'], obs_subj['n1'])
        n_safe_bin = np.array(pd.qcut(n_safe, q=3, labels=['Low', 'Mid', 'High']))

        for trial_i in range(len(obs_subj)):
            rows.append({
                'subject': subj, 'order': order[trial_i],
                'n_safe_bin': str(n_safe_bin[trial_i]),
                'obs': obs_risky[trial_i],
                **{s: pred_risky[trial_i, j] for j, s in enumerate(sample_cols)},
            })
    
    df_all = pd.DataFrame(rows)
    # Average over subjects within group, then by order x stake
    grouped = df_all.groupby(['order', 'n_safe_bin'])
    obs_mean = grouped['obs'].mean()
    pred_vals = grouped[sample_cols].mean()  # mean over subjects+trials per sample
    pred_mean = pred_vals.mean(1)
    pred_lo   = pred_vals.quantile(0.025, axis=1)
    pred_hi   = pred_vals.quantile(0.975, axis=1)
    return obs_mean, pred_mean, pred_lo, pred_hi

fig, axes = plt.subplots(3, 2, figsize=(12, 13))
groups = [
    ('Memory effect (top 5)', top5_memory),
    ('Attention primacy (top 5)', top5_attention),
]
model_list = [
    (model_eu_sym,   idata_eu_sym,   'EU'),
    (model_klw_sym,  idata_klw_sym,  'KLW'),
    (model_full_sym, idata_full_sym, 'PMCM'),
]
order_colors = {'Safe first': '#4393c3', 'Risky first': '#d73027'}

for col, (group_label, subj_ids) in enumerate(groups):
    for row, (model, idat, model_name) in enumerate(model_list):
        ax = axes[row, col]
        obs, pred, pred_lo, pred_hi = subgroup_ppc(
            model, idat, df_sym, df_sym_p, subj_ids, group_label)

        pos = 0
        x_pos, x_labels = [], []
        for order in ['Safe first', 'Risky first']:
            for stake in ['Low', 'Mid', 'High']:
                if (order, stake) not in obs.index:
                    continue
                x_pos.append(pos)
                x_labels.append(f'{order[0]}:{stake[0]}')
                c = order_colors[order]
                ax.bar(pos, obs.loc[(order, stake)], width=0.35,
                       color=c, alpha=0.4, edgecolor=c)
                ax.errorbar(pos, pred.loc[(order, stake)],
                           yerr=[[pred.loc[(order, stake)] - pred_lo.loc[(order, stake)]],
                                 [pred_hi.loc[(order, stake)] - pred.loc[(order, stake)]]],
                           fmt='s', ms=7, color='black', elinewidth=1.5, capsize=3, zorder=5)
                pos += 1
            pos += 0.5

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, ls=':', c='gray', lw=1)
        if col == 0: ax.set_ylabel('P(chose risky)')
        ax.set_title(f'{model_name} \u2014 {group_label}', fontsize=10)
        sns.despine(ax=ax)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='#4393c3', alpha=0.4, label='Data: safe first'),
    Patch(facecolor='#d73027', alpha=0.4, label='Data: risky first'),
    Line2D([0], [0], marker='s', color='black', lw=0, ms=7, label='Model (95 % CI)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
plt.suptitle('Subgroup PPCs (symbolic): 5 memory-effect vs 5 attention-primacy subjects',
             fontsize=12, y=1.01)
plt.tight_layout()
"""),



md(r"""## ELPD model comparison

ELPD (via PSIS-LOO) formally ranks the three models.  Since we stored log-likelihoods
during sampling, `az.compare` works directly.
"""),

code("""\
import arviz as az

print("=== Dot clouds ===")
compare_dot = az.compare({'EU': idata_eu, 'KLW': idata_klw, 'PMCM': idata_full})
print(compare_dot[['elpd_loo', 'p_loo', 'elpd_diff', 'dse', 'warning']].to_string())

print("\\n=== Symbolic ===")
compare_sym = az.compare({'EU': idata_eu_sym, 'KLW': idata_klw_sym, 'PMCM': idata_full_sym})
print(compare_sym[['elpd_loo', 'p_loo', 'elpd_diff', 'dse', 'warning']].to_string())
"""),

code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
az.plot_compare(compare_dot, ax=axes[0])
axes[0].set_title('Dot clouds — ELPD comparison')
az.plot_compare(compare_sym, ax=axes[1])
axes[1].set_title('Symbolic — ELPD comparison')
plt.tight_layout()
"""),

md(r"""## Summary

1. **Presentation order** creates asymmetric noise ($\nu_1 \neq \nu_2$), but the direction
   and magnitude of the asymmetry varies across participants.
2. For **dot clouds**: most participants show a memory effect ($\nu_1 > \nu_2$).  The PMCM
   captures the resulting order × stake-size interaction that EU and KLW cannot.
3. For **symbolic stimuli**: the pattern is more heterogeneous.  Some participants show
   memory effects, others show **attentional primacy** ($\nu_1 < \nu_2$) — as if they
   attend more to the first option when the encoding bottleneck is less severe.
4. **Individual PPCs** reveal whether the group-level fit hides poor individual fits —
   PMCM's separate $\nu_1, \nu_2$ can accommodate both strategies, while KLW and EU cannot.
5. **ELPD comparison** formally ranks the models: PMCM should dominate for dot clouds
   where the order effect is strong.

In [Lesson 4](lesson4.ipynb) we go one step further: instead of assuming a fixed
log-space noise, we let the noise curve $\nu(n)$ vary freely across magnitudes using
B-splines — and test whether Weber's law holds statistically via ELPD model comparison.
"""),

]

write_if_changed(nb3, 'lesson3.ipynb')


# ─────────────────────────────────────────────────────────────────────────────
# Lesson 4 — Flexible noise curves: why and when they help
# ─────────────────────────────────────────────────────────────────────────────

nb4 = nbf.v4.new_notebook()
nb4.cells = [

md(r"""# Lesson 4: Flexible Noise Curves — `FlexibleNoiseComparisonModel` and `FlexibleNoiseRiskModel`

## Motivation: when does Weber's law hold?

The log-space models from lessons 2–3 (MCM, PMCM) assume that the internal
representation of magnitude follows Weber's law: noise scales proportionally
with magnitude, $\sigma(n) \propto n$.  This is a good description for **perceptual**
stimuli such as dot arrays, where numerosity is read off directly from visual input.

But what about **symbolic** numerals?  When participants see the digit \u201c47\u201d, the
noise on their internal representation need not scale the same way.  The
**Flexible Noise models** let the data reveal the actual noise-vs-magnitude curve
rather than assuming it in advance.

The key difference from the log-space models:

| Model | Evidence space | Noise $\nu(n)$ | Weber's law |
|-------|---------------|----------------|-----------------|
| MCM / PMCM | log $n$ | fixed scalar $\nu_\text{log}$ | $\sigma_\text{nat}(n) = \nu_\text{log}\!\times\!n$ (linear) |
| **AffineNoise** | $n$ (natural) | $\text{softplus}(\beta_0 + \beta_1 \hat{n})$ | linear $\Leftrightarrow \beta_0 = 0$ |
| **FlexNoise** | $n$ (natural) | B-spline curve | linear $\nu(n) \propto n$ |

All three natural-space models share the same core computation:

$$\nu_k(n) = \text{softplus}\!\left(\sum_{j} \beta_{k,j}\,\phi_j(n)\right)$$

What changes between them is the **basis** $\phi_j(n)$:
- **FlexNoise** uses B-spline basis functions (default 5 per option) — very flexible
  but potentially over-parameterised.
- **AffineNoise** uses a simple two-element basis $[\,1,\; \hat n\,]$ where
  $\hat n = (n - n_\min)/(n_\max - n_\min)$.  This gives an intercept (baseline noise)
  plus a linear term (Weber-like scaling): $\nu(n) \approx a + b\,n$.

The affine model sits neatly in the complexity hierarchy:

$$\text{MCM (2 noise params)} \quad\subset\quad \text{AffineNoise (4 noise params)}
\quad\subset\quad \text{FlexNoise (10 noise params)}$$

If the FlexNoise ELPD advantage over MCM is driven by a non-zero noise floor rather than
genuine nonlinearity, the AffineNoise model should capture most of the improvement with
far fewer parameters.

- **Linear** $\nu(n) \propto n$ $\Rightarrow$ scale invariance / Weber's law
- **Flat** $\nu(n)$ $\Rightarrow$ constant absolute noise (sub-Weber at large $n$)
- **Affine** $\nu(n) = a + bn$ $\Rightarrow$ noise floor + Weber scaling
- **Super-linear** $\nu(n)$ $\Rightarrow$ worse precision at large $n$ (super-Weber)
"""),

code("""\
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from bauer.utils.data import load_garcia2022, load_dehollander2024
from bauer.models import (MagnitudeComparisonModel, FlexibleNoiseComparisonModel,
                           RiskModel, FlexibleNoiseRiskModel)

# ── AffineNoise: a simple alternative between MCM and FlexNoise ──────────────
# The key idea: FlexibleNoiseComparisonModel uses B-spline basis functions to
# parameterise ν(n).  If the deviation from Weber's law is just a constant noise
# floor (ν(n) = a + b·n instead of ν(n) = b·n), we can capture that with a
# two-element basis [1, n̂] instead of 5 B-spline bases.
#
# Implementation: subclass FlexibleNoiseComparisonModel, override make_dm()
# to return an affine design matrix, and fix polynomial_order=2.

class AffineNoiseComparisonModel(FlexibleNoiseComparisonModel):
    # Magnitude-comparison model with affine noise: v(n) = softplus(b0 + b1*n_hat)
    def __init__(self, paradigm, fit_seperate_evidence_sd=True,
                 fit_prior=False, memory_model='independent'):
        super().__init__(paradigm, fit_seperate_evidence_sd=fit_seperate_evidence_sd,
                         fit_prior=fit_prior, polynomial_order=2,
                         memory_model=memory_model)

    def make_dm(self, x, variable='n1_evidence_sd'):
        # Override: [1, n_hat] basis instead of B-splines
        min_n = self.paradigm[['n1', 'n2']].min().min()
        max_n = self.paradigm[['n1', 'n2']].max().max()
        x_norm = (np.asarray(x, dtype=float) - min_n) / (max_n - min_n)
        return np.column_stack([np.ones_like(x_norm), x_norm])


class AffineNoiseRiskModel(FlexibleNoiseRiskModel):
    # Risky-choice model with affine noise: v(n) = softplus(b0 + b1*n_hat)
    def __init__(self, paradigm, prior_estimate='full',
                 fit_seperate_evidence_sd=True, memory_model='independent'):
        super().__init__(paradigm, prior_estimate=prior_estimate,
                         fit_seperate_evidence_sd=fit_seperate_evidence_sd,
                         polynomial_order=2, memory_model=memory_model)

    def make_dm(self, x, variable='n1_evidence_sd'):
        # Override: [1, n_hat] basis instead of B-splines
        min_n = self.paradigm[['n1', 'n2']].min().min()
        max_n = self.paradigm[['n1', 'n2']].max().max()
        x_norm = (np.asarray(x, dtype=float) - min_n) / (max_n - min_n)
        return np.column_stack([np.ones_like(x_norm), x_norm])

# Garcia et al. (2022) — dot-array magnitude comparison
df_mag = load_garcia2022(task='magnitude')
print(f"Garcia magnitude  |  subjects: {df_mag.index.get_level_values('subject').nunique()},  "
      f"trials: {len(df_mag)}")

# de Hollander et al. (2024, bioRxiv) — Arabic-numeral gambles
df_sym = load_dehollander2024(task='symbolic')
print(f"Arabic numerals   |  subjects: {df_sym.index.get_level_values('subject').nunique()},  "
      f"trials: {len(df_sym)}")
"""),

md(r"""## Illustrating flexible noise curves

The left panel shows the three canonical noise-vs-magnitude shapes.
The right panel shows the **key signature of Weber's law**: when noise is
linear ($\nu \propto n$), psychometric curves plotted against the *relative*
difference $(n_2 - n_1)/n_1$ collapse onto a single curve regardless of the
reference magnitude $n_1$ — scale invariance.  Flat or super-linear noise
breaks this collapse.
"""),

code("""\
from scipy.stats import norm as scipy_norm

n_vals   = np.linspace(5, 45, 200)
nu0      = 0.30          # noise level at mean magnitude
rel_diffs = np.linspace(-0.5, 0.5, 300)   # (n2 - n1) / n1

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: noise curves in natural space
ax = axes[0]
noise_curves = [
    ('Linear  \u03bd(n) = \u03bd\u2080\u00b7n  [Weber / scale-invariant]',
     nu0 * n_vals / n_vals.mean(),        '#2166ac', '-'),
    ('Flat  \u03bd(n) = const  [constant absolute noise]',
     np.full_like(n_vals, nu0),           '#d73027', '--'),
    ('Super-linear  [worse precision at large n]',
     nu0 * (n_vals / n_vals.mean())**1.8, '#1a9850', ':'),
]
for label, nu_n, c, ls in noise_curves:
    ax.plot(n_vals, nu_n, lw=2.5, color=c, ls=ls, label=label)
ax.set_xlabel('Magnitude  n')
ax.set_ylabel('Natural-space noise  \u03bd(n)')
ax.set_title('Noise-vs-magnitude curve shapes')
ax.legend(fontsize=8.5); sns.despine(ax=ax)

# Right: scale-invariance test — plot P(chose n2) vs (n2-n1)/n1
# With linear noise \u03bd = \u03bd\u2080\u00d7n_ref: P = \u03a6(rel_diff \u00d7 n1 / (\u221a2 \u00d7 \u03bd\u2080 \u00d7 n1))
#                                    = \u03a6(rel_diff / (\u221a2 \u00d7 \u03bd\u2080))  — same for all n1!
# With flat noise \u03bd = const: P = \u03a6(rel_diff \u00d7 n1 / (\u221a2 \u00d7 \u03bd_const))  — varies with n1
ax = axes[1]
for n_ref, c, ls in [(8, '#4393c3', '-'), (20, '#1a9850', '--'), (36, '#d73027', ':')]:
    nu_lin  = nu0 * n_ref / n_vals.mean()   # linear noise at n_ref
    nu_flat = nu0                            # flat noise (constant)
    p_lin  = scipy_norm.cdf(rel_diffs * n_ref / (np.sqrt(2) * nu_lin))   # collapses!
    p_flat = scipy_norm.cdf(rel_diffs * n_ref / (np.sqrt(2) * nu_flat))  # shifts by n_ref
    ax.plot(rel_diffs, p_lin,  color=c, lw=2.5, ls=ls, label=f'Linear \u03bd (n\u2081={n_ref})')
    ax.plot(rel_diffs, p_flat, color=c, lw=1.2, ls=ls, alpha=.4)
ax.axhline(.5, ls='--', c='gray', lw=1)
ax.axvline(0,  ls='--', c='gray', lw=1)
ax.text(0.97, 0.07, 'Faint: flat \u03bd (curves shift with n\u2081)',
        transform=ax.transAxes, ha='right', fontsize=8, color='gray')
ax.set_xlabel('Relative difference  (n\u2082 − n\u2081) / n\u2081')
ax.set_ylabel('P(chose n\u2082)')
ax.set_title('Weber\u2019s law = scale invariance: linear \u03bd curves collapse')
ax.legend(fontsize=8.5); sns.despine(ax=ax)

plt.suptitle('Flexible noise: shape determines whether Weber\u2019s law holds',
             fontsize=12, y=1.02)
plt.tight_layout()
"""),

md(r"""## Part A: Dot-array magnitude comparison (Garcia et al. 2022)

Dot arrays are a classic **perceptual** numerosity stimulus where Weber's law is
well-established.  Fitting the Flexible Noise model here serves as a sanity check:
the recovered $\nu(n)$ should be approximately **linear**.  The MCM baseline assumes
log-space noise, which translates to $\sigma_\text{MCM}(n) = \nu_\text{log} \times n$
(a line through the origin) — the same Weber's-law prediction.
"""),

code("""\
# MagnitudeComparisonModel (MCM) — fixed log-space noise (Weber's-law null)
# idata_kwargs passes extra arguments to pymc.sample_posterior_predictive;
# log_likelihood=True stores trial-level log-likelihoods needed for ELPD comparison.
model_mcm = MagnitudeComparisonModel(paradigm=df_mag, fit_seperate_evidence_sd=True)
model_mcm.build_estimation_model(data=df_mag, hierarchical=True)
idata_mcm = model_mcm.sample(draws=200, tune=200, chains=4, progressbar=False,
                              idata_kwargs={'log_likelihood': True})
"""),

code("""\
# FlexibleNoiseComparisonModel — free noise curve fitted to dot arrays
model_flex_mag = FlexibleNoiseComparisonModel(paradigm=df_mag,
                                               fit_seperate_evidence_sd=True,
                                               polynomial_order=5)
model_flex_mag.build_estimation_model(paradigm=df_mag, hierarchical=True)
idata_flex_mag = model_flex_mag.sample(draws=200, tune=200, chains=4, progressbar=False,
                                        idata_kwargs={'log_likelihood': True})
"""),

code("""\
# AffineNoiseComparisonModel — intercept + linear noise (defined above)
model_affine_mag = AffineNoiseComparisonModel(paradigm=df_mag,
                                               fit_seperate_evidence_sd=True)
model_affine_mag.build_estimation_model(paradigm=df_mag, hierarchical=True)
idata_affine_mag = model_affine_mag.sample(draws=200, tune=200, chains=4, progressbar=False,
                                            idata_kwargs={'log_likelihood': True})
"""),

md(r"""### Posterior noise curves — dot arrays

If Weber's law holds, all three curves should overlap: both the FlexNoise (red) and
AffineNoise (green) should track the MCM reference line $\sigma = \nu_\text{log} \times n$
(dashed blue).  A non-zero intercept in the AffineNoise curve would indicate baseline
noise that does not scale with magnitude.
"""),

code("""\
sd_curves_mag = model_flex_mag.get_sd_curve(idata=idata_flex_mag, variable='both',
                                              group=True, data=df_mag.reset_index())
sd_curves_aff = model_affine_mag.get_sd_curve(idata=idata_affine_mag, variable='both',
                                                group=True, data=df_mag.reset_index())
nu1_mcm = idata_mcm.posterior['n1_evidence_sd_mu'].values.ravel()
nu2_mcm = idata_mcm.posterior['n2_evidence_sd_mu'].values.ravel()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = {'flex': '#d73027', 'affine': '#1a9850', 'mcm': '#4393c3'}

for ax, (var_col, nu_mcm_samp, title) in zip(
        axes,
        [('n1_evidence_sd', nu1_mcm, 'First option  \u03c3\u2081(n)'),
         ('n2_evidence_sd', nu2_mcm, 'Second option  \u03c3\u2082(n)')]):
    # FlexNoise (B-spline)
    grp = sd_curves_mag.groupby(level='x')[var_col]
    x_vals = grp.mean().index.values
    ax.fill_between(x_vals, grp.quantile(0.025).values, grp.quantile(0.975).values,
                    alpha=.18, color=colors['flex'])
    ax.plot(x_vals, grp.mean().values, lw=2.5, color=colors['flex'], label='FlexNoise (B-spline)')
    # AffineNoise (intercept + linear)
    grp_a = sd_curves_aff.groupby(level='x')[var_col]
    x_a = grp_a.mean().index.values
    ax.fill_between(x_a, grp_a.quantile(0.025).values, grp_a.quantile(0.975).values,
                    alpha=.18, color=colors['affine'])
    ax.plot(x_a, grp_a.mean().values, lw=2.5, color=colors['affine'], label='AffineNoise (a + b\u00b7n)')
    # MCM reference (Weber: \u03c3 = \u03bd\u2080 \u00d7 n)
    nu0 = nu_mcm_samp.mean()
    ax.plot(x_vals, nu0 * x_vals, ls='--', lw=2, color=colors['mcm'],
            label=f'MCM (Weber)  \u03bd\u2080={nu0:.2f}')
    ax.fill_between(x_vals,
                    np.percentile(nu_mcm_samp, 2.5) * x_vals,
                    np.percentile(nu_mcm_samp, 97.5) * x_vals,
                    alpha=.10, color=colors['mcm'])
    ax.set_xlabel('Magnitude  n')
    ax.set_ylabel('Natural-space noise  \u03c3(n)')
    ax.set_title(title); ax.legend(fontsize=8.5); sns.despine(ax=ax)

plt.suptitle('Dot arrays: MCM vs AffineNoise vs FlexNoise (95 % posterior interval)',
             fontsize=12, y=1.02)
plt.tight_layout()
"""),

md(r"""### Model comparison: MCM vs AffineNoise vs FlexNoise (dot arrays)

**ELPD** (Expected Log Pointwise Predictive Density, computed via PSIS-LOO) formally
tests whether the added flexibility is justified.  The three-way comparison lets us
disentangle two questions:

1. **Is Weber's law (MCM) adequate?** Compare MCM vs AffineNoise.
2. **Is the deviation captured by a simple intercept, or is genuine nonlinearity needed?**
   Compare AffineNoise vs FlexNoise.

If AffineNoise matches FlexNoise on ELPD while beating MCM, the deviation from Weber's
law is well-described by a noise floor — a simple, interpretable parameter.
"""),

code("""\
# ELPD model comparison — dot arrays (3-way)
compare_mag = az.compare({'MCM (Weber)':   idata_mcm,
                           'AffineNoise':   idata_affine_mag,
                           'FlexNoise':     idata_flex_mag})
print(compare_mag[['elpd_loo', 'p_loo', 'elpd_diff', 'dse', 'warning']].to_string())
"""),

code("""\
ax = az.plot_compare(compare_mag, figsize=(7, 3.5))
ax.set_title('ELPD comparison — dot-array magnitude comparison (higher = better prediction)')
plt.tight_layout()
"""),

code("""\
# ── Interpret the dot-array ELPD result (3-way) ─────────────────────────────
print("ELPD ranking (dot arrays):")
print(compare_mag[['elpd_loo', 'p_loo', 'elpd_diff', 'dse', 'warning']].to_string())
print()

# Pairwise interpretation
for i in range(1, len(compare_mag)):
    name = compare_mag.index[i]
    diff = compare_mag['elpd_diff'].iloc[i]
    dse  = compare_mag['dse'].iloc[i]
    ratio = abs(diff) / dse if dse > 0 else float('inf')
    winner = compare_mag.index[0]
    verdict = "distinguishable" if ratio > 2 else "NOT distinguishable"
    print(f"  {winner} vs {name}:  DELTA_ELPD = {diff:.1f},  SE = {dse:.1f},  "
          f"|ratio| = {ratio:.1f}  ->  {verdict}")

# Key question: does AffineNoise capture the FlexNoise advantage?
if 'AffineNoise' in compare_mag.index and 'FlexNoise' in compare_mag.index:
    aff_rank = list(compare_mag.index).index('AffineNoise')
    flex_rank = list(compare_mag.index).index('FlexNoise')
    print()
    if abs(aff_rank - flex_rank) <= 1:
        aff_elpd = compare_mag.loc['AffineNoise', 'elpd_loo']
        flex_elpd = compare_mag.loc['FlexNoise', 'elpd_loo']
        print(f"AffineNoise ELPD: {aff_elpd:.1f},  FlexNoise ELPD: {flex_elpd:.1f}")
        if abs(aff_elpd - flex_elpd) < 10:
            print("-> Affine noise captures most of the FlexNoise advantage.")
            print("   The deviation from Weber's law is well-described by a noise floor (a + b*n).")
        else:
            print("-> Affine noise does NOT fully capture the flexible curve's advantage.")
            print("   Genuine nonlinearity in the noise function is needed.")
"""),

md(r"""## Part B: Arabic-numeral risky choice (de Hollander et al., 2024, bioRxiv)

Arabic numerals are **symbolic**: participants read a printed digit rather than
estimating numerosity from a visual display.  The internal noise on symbolic
number representations need not scale proportionally with magnitude, so we
expect a potential **deviation from the linear Weber's-law prediction**.

We compare `FlexibleNoiseRiskModel` and `AffineNoiseRiskModel` against the
standard `RiskModel` (PMCM) on the same data.
"""),

code("""\
def prep_df(df):
    df = df.reset_index()
    risky_first = df['p1'] == 0.55
    df['log_ratio']     = np.log(
        np.where(risky_first, df['n1'], df['n2']) /
        np.where(risky_first, df['n2'], df['n1']))
    df['chose_risky']   = np.where(risky_first, ~df['choice'], df['choice'])
    df['n_safe']        = np.where(risky_first, df['n2'], df['n1'])
    df['risky_first']   = risky_first
    df['order']         = np.where(risky_first, 'Risky first', 'Safe first')
    df['log_ratio_bin'] = (pd.cut(df['log_ratio'], bins=10)
                             .map(lambda x: x.mid).astype(float))
    df['n_safe_bin']    = pd.qcut(df['n_safe'], q=3,
                                   labels=['Low stakes', 'Mid stakes', 'High stakes'])
    return df

df_sym_p = prep_df(df_sym)
"""),

code("""\
# PMCM (fixed log-space noise) — log_likelihood stored for ELPD comparison
model_pmcm = RiskModel(paradigm=df_sym, prior_estimate='full',
                        fit_seperate_evidence_sd=True)
model_pmcm.build_estimation_model(data=df_sym, hierarchical=True, save_p_choice=True)
idata_pmcm = model_pmcm.sample(draws=200, tune=200, chains=4, progressbar=False,
                                idata_kwargs={'log_likelihood': True})
"""),

code("""\
# FlexibleNoiseRiskModel — free noise curve on Arabic-numeral data
model_flex = FlexibleNoiseRiskModel(paradigm=df_sym, prior_estimate='full',
                                     fit_seperate_evidence_sd=True, polynomial_order=5)
model_flex.build_estimation_model(paradigm=df_sym, hierarchical=True, save_p_choice=True)
idata_flex = model_flex.sample(draws=200, tune=200, chains=4, progressbar=False,
                                idata_kwargs={'log_likelihood': True})
"""),

code("""\
# AffineNoiseRiskModel — intercept + linear noise for Arabic-numeral gambles
model_affine = AffineNoiseRiskModel(paradigm=df_sym, prior_estimate='full',
                                     fit_seperate_evidence_sd=True)
model_affine.build_estimation_model(paradigm=df_sym, hierarchical=True, save_p_choice=True)
idata_affine = model_affine.sample(draws=200, tune=200, chains=4, progressbar=False,
                                    idata_kwargs={'log_likelihood': True})
"""),

md(r"""### Posterior noise curves — Arabic numerals

Three models are overlaid: the B-spline FlexNoise (red), the two-parameter AffineNoise
(green), and the PMCM Weber reference (dashed blue).
"""),

code("""\
sd_curves = model_flex.get_sd_curve(idata=idata_flex, variable='both',
                                     group=True, data=df_sym.reset_index())
sd_curves_aff_sym = model_affine.get_sd_curve(idata=idata_affine, variable='both',
                                                group=True, data=df_sym.reset_index())
nu1_pmcm = idata_pmcm.posterior['n1_evidence_sd_mu'].values.ravel()
nu2_pmcm = idata_pmcm.posterior['n2_evidence_sd_mu'].values.ravel()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = {'flex': '#d73027', 'affine': '#1a9850', 'pmcm': '#4393c3'}

for ax, (var_col, nu_pmcm_samp, title) in zip(
        axes,
        [('n1_evidence_sd', nu1_pmcm, 'First option  \u03c3\u2081(n)'),
         ('n2_evidence_sd', nu2_pmcm, 'Second option  \u03c3\u2082(n)')]):
    # FlexNoise
    grp = sd_curves.groupby(level='x')[var_col]
    x_vals = grp.mean().index.values
    ax.fill_between(x_vals, grp.quantile(0.025).values, grp.quantile(0.975).values,
                    alpha=.18, color=colors['flex'])
    ax.plot(x_vals, grp.mean().values, lw=2.5, color=colors['flex'], label='FlexNoise (B-spline)')
    # AffineNoise
    grp_a = sd_curves_aff_sym.groupby(level='x')[var_col]
    x_a = grp_a.mean().index.values
    ax.fill_between(x_a, grp_a.quantile(0.025).values, grp_a.quantile(0.975).values,
                    alpha=.18, color=colors['affine'])
    ax.plot(x_a, grp_a.mean().values, lw=2.5, color=colors['affine'], label='AffineNoise (a + b\u00b7n)')
    # PMCM reference
    nu0 = nu_pmcm_samp.mean()
    ax.plot(x_vals, nu0 * x_vals, ls='--', lw=2, color=colors['pmcm'],
            label=f'PMCM (Weber)  \u03bd\u2080={nu0:.2f}')
    ax.fill_between(x_vals,
                    np.percentile(nu_pmcm_samp, 2.5) * x_vals,
                    np.percentile(nu_pmcm_samp, 97.5) * x_vals,
                    alpha=.10, color=colors['pmcm'])
    ax.set_xlabel('Payoff magnitude  n')
    ax.set_ylabel('Natural-space noise  \u03c3(n)')
    ax.set_title(title); ax.legend(fontsize=8.5); sns.despine(ax=ax)

plt.suptitle('Arabic numerals: PMCM vs AffineNoise vs FlexNoise (95\u202f% posterior interval)',
             fontsize=12, y=1.02)
plt.tight_layout()
"""),

md("""## Posterior predictive comparison (Arabic numerals)

We overlay both models' predictions against the observed presentation-order
\u00d7 stake-size pattern — the same diagnostic used in lesson 3.
"""),

code("""\
from bauer.utils import summarize_ppc_group
stake_pal = dict(zip(['Low stakes', 'Mid stakes', 'High stakes'], sns.color_palette('YlOrRd', 3)))

def add_model_ppc(df_orig, df_prepped, model, idata, model_name):
    \"\"\"Two-step PPC via summarize_ppc_group.\"\"\"
    ppc_df  = model.ppc(df_orig, idata, var_names=['ll_bernoulli'])
    ppc_ll  = ppc_df.xs('ll_bernoulli', level='variable')
    sample_cols = ppc_ll.columns.tolist()

    ppc_flat = ppc_ll.reset_index()
    risky_first = (ppc_flat['p1'] == 0.55)
    ppc_flat[sample_cols] = np.where(
        risky_first.values[:, None],
        1 - ppc_flat[sample_cols].values,
        ppc_flat[sample_cols].values
    )
    ppc_flat['order'] = np.where(risky_first, 'Risky first', 'Safe first')
    log_ratio = np.log(
        np.where(risky_first, ppc_flat['n1'], ppc_flat['n2']) /
        np.where(risky_first, ppc_flat['n2'], ppc_flat['n1']))
    ppc_flat['log_ratio_bin'] = (pd.cut(pd.Series(log_ratio), bins=10)
                                   .map(lambda x: x.mid).astype(float).values)
    n_safe = np.where(risky_first, ppc_flat['n2'], ppc_flat['n1'])
    ppc_flat['n_safe_bin'] = pd.qcut(n_safe, q=3,
                                      labels=['Low stakes', 'Mid stakes', 'High stakes'])

    result = summarize_ppc_group(ppc_flat,
                                  condition_cols=['order', 'n_safe_bin', 'log_ratio_bin'])
    return result.rename(columns={'p_predicted': 'p_mean',
                                   'hdi025': 'p_lo', 'hdi975': 'p_hi'}).reset_index()


def plot_ppc_row(df_pred, df_obs, model_name, axes_row):
    hue_order = ['Low stakes', 'Mid stakes', 'High stakes']
    for ax, order_val in zip(axes_row, ['Risky first', 'Safe first']):
        pred = df_pred[df_pred['order'] == order_val]
        obs  = df_obs[df_obs['order']   == order_val]
        for sbin in hue_order:
            p = pred[pred['n_safe_bin'] == sbin]
            o = obs[obs['n_safe_bin']   == sbin]
            if len(o) == 0:
                continue
            c = stake_pal[sbin]
            ax.fill_between(p['log_ratio_bin'], p['p_lo'], p['p_hi'], color=c, alpha=.2)
            ax.plot(p['log_ratio_bin'], p['p_mean'], color=c, lw=2, label=sbin)
            ax.scatter(o['log_ratio_bin'], o['chose_risky'],
                       color=c, s=25, zorder=5, alpha=.85)
        ax.axhline(.5, ls='--', c='gray', lw=1)
        ax.axvline(np.log(1/.55), ls='--', c='#333333', lw=1.5)
        ax.set_ylim(-.05, 1.05)
        ax.set_title(f'{model_name} — {order_val}', fontsize=9)
        ax.set_xlabel('log(risky / safe)'); ax.set_ylabel('P(chose risky)')
        ax.legend(title='Safe stake', fontsize=7, loc='upper left')
        sns.despine(ax=ax)


obs_sym_l4 = (df_sym_p
              .groupby(['subject', 'order', 'n_safe_bin', 'log_ratio_bin'])['chose_risky']
              .mean()
              .groupby(['order', 'n_safe_bin', 'log_ratio_bin']).mean()
              .reset_index())

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
for (mdl, idat, name), row in zip(
        [(model_pmcm, idata_pmcm, 'PMCM (fixed noise)'),
         (model_flex, idata_flex, 'Flexible Noise')],
        axes):
    df_pred = add_model_ppc(df_sym, df_sym_p, mdl, idat, name)
    plot_ppc_row(df_pred, obs_sym_l4, name, row)

plt.suptitle('Posterior predictive comparison: PMCM vs Flexible Noise (Arabic numerals)',
             fontsize=12, y=1.01)
plt.tight_layout()
"""),

md(r"""### Model comparison: PMCM vs AffineNoise vs FlexNoise (Arabic numerals)

The same three-way ELPD comparison for the risk models.
"""),

code("""\
# ELPD comparison — Arabic numerals risky choice (3-way)
compare_sym = az.compare({'PMCM (Weber)':   idata_pmcm,
                           'AffineNoise':    idata_affine,
                           'FlexNoise':      idata_flex})
print(compare_sym[['elpd_loo', 'p_loo', 'elpd_diff', 'dse', 'warning']].to_string())
"""),

code("""\
ax = az.plot_compare(compare_sym, figsize=(7, 3.5))
ax.set_title('ELPD comparison — Arabic-numeral risky choice (higher = better prediction)')
plt.tight_layout()
"""),

code("""\
# ── Interpret the Arabic-numeral ELPD result (3-way) ─────────────────────────
print("ELPD ranking (Arabic numerals):")
print(compare_sym[['elpd_loo', 'p_loo', 'elpd_diff', 'dse', 'warning']].to_string())
print()

for i in range(1, len(compare_sym)):
    name = compare_sym.index[i]
    diff = compare_sym['elpd_diff'].iloc[i]
    dse  = compare_sym['dse'].iloc[i]
    ratio = abs(diff) / dse if dse > 0 else float('inf')
    winner = compare_sym.index[0]
    verdict = "distinguishable" if ratio > 2 else "NOT distinguishable"
    print(f"  {winner} vs {name}:  DELTA_ELPD = {diff:.1f},  SE = {dse:.1f},  "
          f"|ratio| = {ratio:.1f}  ->  {verdict}")

if 'AffineNoise' in compare_sym.index and 'FlexNoise' in compare_sym.index:
    aff_elpd = compare_sym.loc['AffineNoise', 'elpd_loo']
    flex_elpd = compare_sym.loc['FlexNoise', 'elpd_loo']
    print(f"\\nAffineNoise ELPD: {aff_elpd:.1f},  FlexNoise ELPD: {flex_elpd:.1f}")
    if abs(aff_elpd - flex_elpd) < 10:
        print("-> Affine noise captures most of the FlexNoise advantage (if any).")
    else:
        print("-> Genuine nonlinearity in the noise function may be needed.")
"""),

md(r"""## Take-aways

**What the ELPD actually tells us in this dataset:**

- **Dot arrays (Garcia et al. 2022):** the flexible noise model wins with
  $|\Delta\text{ELPD}| / \text{SE} \approx 3$, so the data *do* discriminate between the
  two — the MCM's strictly linear noise curve is not quite right.  The recovered
  $\nu(n)$ is roughly linear but deviates, suggesting a mild departure from perfect
  Weber's law in perceptual dot-array numerosity.
- **Arabic numerals (de Hollander et al., 2024, bioRxiv):** $|\Delta\text{ELPD}| / \text{SE} < 2$,
  so there is *no statistically distinguishable difference* in this dataset.  Weber's law
  (PMCM) is adequate for symbolic numbers here.  This may seem counterintuitive — but
  with only ~250 trials per subject, the data are simply not powerful enough to detect
  the kind of moderate deviations a flexible spline can pick up.

**Methodological take-aways:**

- The ELPD comparison gives a principled answer where visual inspection of noise curves
  cannot: $|\Delta\text{ELPD}| / \text{SE} > 2$ is a reasonable threshold for claiming
  the models are distinguishable.
- Rhat and ESS warnings in these notebooks reflect the deliberately short chain settings
  (draws=200, tune=200) chosen for speed.  For real analyses use at least draws=1000,
  tune=1000.
- Sampling with `idata_kwargs={{'log_likelihood': True}}` is all that is needed to
  unlock `az.compare` — bauer handles everything else.
"""),

]

write_if_changed(nb4, 'lesson4.ipynb')


# ─────────────────────────────────────────────────────────────────────────────
# Lesson 5 — Why hierarchical modelling beats maximum likelihood
# ─────────────────────────────────────────────────────────────────────────────

nb5 = nbf.v4.new_notebook()
nb5.cells = [

md(r"""# Lesson 5: Why Hierarchical Modelling Beats Maximum Likelihood

## The problem with individual fitting

Maximum-likelihood estimation (MLE) and MAP estimation fit each participant independently.
With hundreds of trials per subject this works fine, but in practice most psychophysics
experiments give you **100--250 trials per condition**.  At those trial counts individual
MLE/MAP estimates are:

- **Noisy** — refit the same subject on a different random half of their data and you get
  a substantially different answer.
- **Biased at the boundaries** — with few trials, the likelihood surface is broad and
  the optimiser can land at extreme values.  Noise parameters ($\nu$) may converge to
  near-zero (the model "explains" every trial perfectly by overfitting) or explode to
  very large values (flat psychometric curve, no discrimination at all).  Lapse-rate
  parameters can rail at 0 or 1.  These boundary estimates are not meaningful — they
  reflect the instability of the optimisation, not the participant's true noise level.
- **Worse for complex models** — every additional parameter increases the volume of
  the parameter space that the optimiser must search.  The most theoretically interesting
  models (KLW with prior $\sigma_0$, FlexibleNoise with spline coefficients) are exactly
  the ones that suffer most, because they have more parameters per subject and more
  opportunities for the likelihood to be flat.

**Hierarchical Bayesian estimation** avoids all three problems.  By sharing statistical
strength across participants, the group prior acts as an *adaptive regulariser*: subjects
with extreme or unreliable data are pulled toward the population, while well-identified
subjects are left alone.

## Empirical demonstration: split-half reliability

The gold standard for measurement quality is **split-half reliability**: split each
participant's trials into two random halves, fit each half separately, and correlate the
parameter estimates across halves.  A reliable method produces the same answer from both
halves; an unreliable one does not.

We compare:

| Method | What it does | Regularisation |
|--------|-------------|----------------|
| **MLE** | `model.fit_map_individual(flat_prior=True)` — each subject fitted alone with flat priors | None — pure maximum likelihood |
| **Hierarchical Bayes** | `model.sample()` with `hierarchical=True` — full MCMC posterior | Adaptive group prior, posterior averaging |

at increasing trial counts (25, 50, 108 per half).  The Garcia et al. magnitude
data have 216 trials per subject, so 108 per half is the natural split.
"""),

code("""\
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bauer.utils.data import load_garcia2022
from bauer.models import MagnitudeComparisonModel

df_mag = load_garcia2022(task='magnitude')
subjects = df_mag.index.get_level_values('subject').unique()
print(f"Subjects: {len(subjects)},  trials per subject: "
      f"{df_mag.groupby(level='subject').size().iloc[0]}")

# ── Split each subject's trials into two random halves ───────────────────────
rng = np.random.default_rng(42)
half_a_rows, half_b_rows = [], []
for subj in subjects:
    mask = df_mag.index.get_level_values('subject') == subj
    subj_iloc = np.where(mask)[0]
    perm = rng.permutation(len(subj_iloc))
    mid = len(subj_iloc) // 2
    half_a_rows.extend(subj_iloc[perm[:mid]])
    half_b_rows.extend(subj_iloc[perm[mid:]])

half_a_full = df_mag.iloc[half_a_rows]
half_b_full = df_mag.iloc[half_b_rows]
print(f"Half A: {len(half_a_full)} trials,  Half B: {len(half_b_full)} trials")
"""),

md(r"""## Split-half analysis: 10 random splits $\times$ 5 trial counts

For each trial count (25, 50, 108) we:
1. Randomly split each subject's data into two halves
2. Subsample *k* trials per half
3. Fit MLE and hierarchical Bayes on each half
4. Correlate the parameter estimates across halves

We repeat this 10 times with different random splits to get a stable estimate of
reliability.  Three correlation metrics are shown: Spearman $\rho$, Pearson $r$,
and $R^2$.
"""),

code("""\
from scipy.stats import pearsonr

def subsample(df, k):
    return df.groupby(level='subject').head(k)

def split_data(df_mag, seed):
    # Random split of each subject's trials into two halves
    rng = np.random.default_rng(seed)
    a_rows, b_rows = [], []
    for subj in df_mag.index.get_level_values('subject').unique():
        mask = df_mag.index.get_level_values('subject') == subj
        iloc = np.where(mask)[0]
        perm = rng.permutation(len(iloc))
        mid = len(iloc) // 2
        a_rows.extend(iloc[perm[:mid]])
        b_rows.extend(iloc[perm[mid:]])
    return df_mag.iloc[a_rows], df_mag.iloc[b_rows]

def fit_mle(data):
    model = MagnitudeComparisonModel(paradigm=data, fit_seperate_evidence_sd=True)
    return model.fit_map_individual(data=data, flat_prior=True)

def fit_hierarchical(data, draws=500, tune=500, chains=2):
    model = MagnitudeComparisonModel(paradigm=data, fit_seperate_evidence_sd=True)
    model.build_estimation_model(data=data, hierarchical=True)
    idata = model.sample(draws=draws, tune=tune, chains=chains, progressbar=False)
    n_subj = len(data.index.unique(level='subject'))
    n1 = idata.posterior['n1_evidence_sd'].values.reshape(-1, n_subj).mean(0)
    n2 = idata.posterior['n2_evidence_sd'].values.reshape(-1, n_subj).mean(0)
    return pd.DataFrame({'n1_evidence_sd': n1, 'n2_evidence_sd': n2},
                         index=pd.Index(data.index.unique(level='subject'), name='subject'))

trial_counts = [25, 50, 108]
n_splits = 3
methods = {'MLE': fit_mle, 'Hierarchical Bayes': fit_hierarchical}
results = []

for split_i in range(n_splits):
    half_a, half_b = split_data(df_mag, seed=split_i)
    for k in trial_counts:
        a_sub = subsample(half_a, k)
        b_sub = subsample(half_b, k)
        for method_name, fit_fn in methods.items():
            est_a = fit_fn(a_sub)
            est_b = fit_fn(b_sub)
            # Also compute sum of both noise params
            est_a['total_sd'] = est_a['n1_evidence_sd'] + est_a['n2_evidence_sd']
            est_b['total_sd'] = est_b['n1_evidence_sd'] + est_b['n2_evidence_sd']
            for param in ['n1_evidence_sd', 'n2_evidence_sd', 'total_sd']:
                rho_p, _ = pearsonr(est_a[param], est_b[param])
                # Count boundary-collapsed estimates (noise near zero)
                n_zero_a = (est_a[param] < 0.01).sum()
                n_zero_b = (est_b[param] < 0.01).sum()
                results.append({
                    'split': split_i, 'k': k, 'method': method_name,
                    'parameter': param,
                    'Pearson r': rho_p, 'R\\u00b2': rho_p**2,
                    'n_collapsed': n_zero_a + n_zero_b,
                })
    print(f"Split {split_i+1}/{n_splits} done")

results_df = pd.DataFrame(results)
print(f"\\nTotal fits: {len(results_df)}")
"""),

code("""\
# ── Reliability vs trial count: Pearson r and R² x 3 parameters ──────────────
metrics = ['Pearson r', 'R\\u00b2']
params = ['n1_evidence_sd', 'n2_evidence_sd', 'total_sd']
param_labels = {'n1_evidence_sd': '\\u03bd\\u2081 (first option)',
                'n2_evidence_sd': '\\u03bd\\u2082 (second option)',
                'total_sd': '\\u03bd\\u2081 + \\u03bd\\u2082 (total)'}
pal = {'MLE': '#d73027', 'Hierarchical Bayes': '#4393c3'}

fig, axes = plt.subplots(len(metrics), len(params),
                          figsize=(5 * len(params), 3.5 * len(metrics)),
                          sharex=True, sharey='row')

for row, metric in enumerate(metrics):
    for col, param in enumerate(params):
        ax = axes[row, col]
        sub = results_df[results_df['parameter'] == param]
        for i_m, method in enumerate(['MLE', 'Hierarchical Bayes']):
            d = sub[sub['method'] == method]
            mean_by_k = d.groupby('k')[metric].mean()
            se_by_k   = d.groupby('k')[metric].std() / np.sqrt(n_splits)
            offset = (i_m - 0.5) * 1.5  # slight x-offset to avoid overlap
            ax.errorbar(mean_by_k.index + offset, mean_by_k,
                        yerr=1.96 * se_by_k,
                        fmt='o-', lw=2, ms=6, capsize=4, capthick=1.5,
                        color=pal[method], ecolor=pal[method], label=method)
        ax.axhline(0, ls=':', c='gray', lw=1)
        if row == 0:
            ax.set_title(param_labels[param])
        if row == len(metrics) - 1:
            ax.set_xlabel('Trials per half')
        if col == 0:
            ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        sns.despine(ax=ax)

plt.suptitle(f'Split-half reliability (mean \\u00b1 95% CI over {n_splits} splits)',
             fontsize=13, y=1.01)
plt.tight_layout()
"""),

md(r"""### The boundary-collapse problem

Why does MLE do so poorly on Pearson $r$ (which measures linear agreement) even when
Spearman $\rho$ (which only checks rank order) looks acceptable?  The answer is
**boundary collapse**: with few trials, the MLE optimiser can push noise parameters
$\nu_1$ or $\nu_2$ to near-zero, meaning the model "explains" every trial perfectly by
overfitting the noise away.  These zero-estimates are meaningless — they do not reflect
a participant with perfect perception, they reflect an optimiser that ran out of data.

The plot below shows how many subjects (out of $N \times 2$ halves) have $\nu < 0.01$
at each trial count:
"""),

code("""\
# ── Boundary collapse diagnostic ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for col, param in enumerate(params):
    ax = axes[col]
    sub = results_df[results_df['parameter'] == param]
    for i_m, method in enumerate(['MLE', 'Hierarchical Bayes']):
        d = sub[sub['method'] == method]
        mean_col = d.groupby('k')['n_collapsed'].mean()
        se_col   = d.groupby('k')['n_collapsed'].std() / np.sqrt(n_splits)
        offset = (i_m - 0.5) * 1.5
        ax.errorbar(mean_col.index + offset, mean_col,
                    yerr=1.96 * se_col,
                    fmt='o-', lw=2, ms=6, capsize=4, capthick=1.5,
                    color=pal[method], ecolor=pal[method], label=method)
    ax.axhline(0, ls=':', c='gray', lw=1)
    ax.set_title(param_labels[param])
    ax.set_xlabel('Trials per half')
    if col == 0:
        ax.set_ylabel('Subjects with \\u03bd < 0.01\\n(out of 2 \\u00d7 N halves)')
    ax.legend(fontsize=8)
    sns.despine(ax=ax)

plt.suptitle('Boundary collapse: MLE pushes noise to zero when data are scarce',
             fontsize=12, y=1.02)
plt.tight_layout()
"""),

md(r"""## What this means in practice

The plot shows what every psychophysics researcher should know but few textbooks
make explicit:

1. **Individual MAP/MLE is unreliable at typical trial counts.**  At 50--100 trials per
   subject the split-half correlation can be near zero — the estimates are dominated by
   sampling noise and tell you almost nothing about the participant.

2. **Hierarchical Bayes is dramatically more reliable.**  The group prior acts as adaptive
   regularisation: extreme/noisy estimates are pulled toward the group mean, which is
   exactly what reduces split-half variability.  The posterior mean is a *shrinkage
   estimator* — the optimal bias-variance trade-off.

3. **The advantage is largest when you need it most** — at low trial counts and for
   complex models with many parameters.  The KLW model (lesson 2) and the FlexibleNoise
   model (lesson 4) have 4+ parameters per subject; individual MLE would be hopeless
   without hundreds of trials, but hierarchical fitting works with the trial counts we
   actually have.

### Why MAP/MLE fails: the bias-variance trade-off

MAP gives the single point with highest posterior density — for a flat prior this is
just maximum likelihood.  At low trial counts:

- The likelihood surface is **flat** in some directions, so the optimiser lands at an
  arbitrary point in a large region of near-equal likelihood.
- **Boundary effects** distort estimates: noise parameters can converge to near-zero
  (overfitting a lucky sample) or blow up.
- There is **no uncertainty quantification**: you get a point estimate with no error bar,
  so you cannot distinguish a well-identified subject from a poorly-identified one.

Hierarchical Bayes solves all three: the group prior tilts the likelihood surface toward
sensible values, the posterior is a full distribution (not a point), and the amount of
shrinkage is automatically calibrated per subject.

### Rule of thumb

If you have < 200 trials per subject per condition, **always use hierarchical fitting**.
If you have < 100, individual fitting is essentially meaningless for models with more
than one or two parameters.  bauer makes hierarchical fitting the default for exactly
this reason.
"""),

code("""\
# ── Scatter: half A vs half B at every trial count, all 3 params ─────────────
half_a, half_b = split_data(df_mag, seed=0)
scatter_colors = {'MLE': '#d73027', 'Hierarchical Bayes': '#4393c3'}

for param, param_label in param_labels.items():
    n_k = len(trial_counts)
    fig, axes = plt.subplots(2, n_k, figsize=(3.5 * n_k, 7), sharey='row', sharex='row')

    for col, k in enumerate(trial_counts):
        a_sub = subsample(half_a, k)
        b_sub = subsample(half_b, k)
        ests = {}
        for name, fn in methods.items():
            ea, eb = fn(a_sub), fn(b_sub)
            ea['total_sd'] = ea['n1_evidence_sd'] + ea['n2_evidence_sd']
            eb['total_sd'] = eb['n1_evidence_sd'] + eb['n2_evidence_sd']
            ests[name] = (ea, eb)
        for row, method in enumerate(['MLE', 'Hierarchical Bayes']):
            ax = axes[row, col]
            est_a, est_b = ests[method]
            rho_p, _ = pearsonr(est_a[param], est_b[param])
            ax.scatter(est_a[param], est_b[param], s=25, alpha=.6,
                       color=scatter_colors[method])
            lims = [0, max(est_a[param].max(), est_b[param].max()) * 1.1]
            ax.plot(lims, lims, '--', color='gray', lw=1)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_title(f'{method}, k={k}  (R\\u00b2={rho_p**2:.2f})', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'Half B')
            if row == 1:
                ax.set_xlabel(f'Half A')
            sns.despine(ax=ax)

    plt.suptitle(f'Split-half scatter: {param_label}  (MLE top, Hierarchical bottom)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
"""),

md(r"""## Take-aways

- **Never use individual MLE/MAP for cognitive models at typical psychophysics trial
  counts.**  The estimates are unreliable and any downstream correlation (e.g. with
  neural data or clinical scores) will be attenuated toward zero.
- **Hierarchical Bayes is not just "nicer" — it is a prerequisite for valid individual-
  difference analyses** at the trial counts we actually work with.
- bauer defaults to hierarchical fitting (`hierarchical=True`) for exactly this reason.
  Individual MAP (`model.fit_map()`) is available for quick sanity checks but should not
  be used for final inference.
- These results generalise beyond magnitude comparison: the noisier your model and the
  fewer your trials, the bigger the hierarchical advantage.
"""),

]

write_if_changed(nb5, 'lesson5.ipynb')
