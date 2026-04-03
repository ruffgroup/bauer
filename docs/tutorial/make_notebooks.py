"""Generate tutorial notebooks for the bauer documentation."""
import nbformat as nbf

def code(src): return nbf.v4.new_code_cell(src.strip())
def md(src):   return nbf.v4.new_markdown_cell(src.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Lesson 1 — Psychophysical Modelling: Theory and Barreto-Garcia et al. Magnitude Data
# ─────────────────────────────────────────────────────────────────────────────

nb1 = nbf.v4.new_notebook()
nb1.cells = [

md(r"""# Lesson 1: Psychophysical Modelling

## The Noisy Logarithmic Coding (NLC) model

When we judge quantities — the number of coins in a pile, the size of a reward — our
internal representations are noisy.  The **NLC model** posits that the brain encodes
numerical magnitude $n$ on a logarithmic scale, and that this log-representation is
corrupted by Gaussian noise:

$$r \sim \mathcal{N}(\log n, \; \nu^2)$$

Given two stimuli $n_1$ and $n_2$ with independent noise, the probability of choosing
$n_2$ as the larger is

$$P(\text{chose}\; n_2) = \Phi\!\left(\frac{\log(n_2/n_1)}{\sqrt{\nu_1^2 + \nu_2^2}}\right)$$

where $\Phi$ is the standard normal CDF, $\nu_1$ is the noise on $n_1$ (the first-presented
option) and $\nu_2$ is the noise on $n_2$ (the second-presented option).

In many experimental paradigms stimuli are shown **sequentially**: the observer perceives
$n_1$, holds it in working memory, then perceives $n_2$.  Memory retention may add extra
noise on top of perception, so the model allows $\nu_1 \neq \nu_2$.  In tasks where both
options are visible simultaneously there is no reason to separate the two, and a single
shared $\nu$ is sufficient.

**Scale invariance**: when $x = \log(n_2/n_1)$ is used as the horizontal axis, the
psychometric function collapses to a single sigmoid regardless of the absolute magnitude
of $n_1$ — a direct prediction of the logarithmic encoding.
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
pal = sns.color_palette('Blues_d', len(n1_vals))

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
plt.tight_layout()
"""),

md(r"""### Effect of noise level and asymmetric noise

**Precision** $\gamma = 1/(\sqrt{2}\,\nu)$ is the slope of the psychometric curve on a
log-ratio axis.  Higher noise → shallower curve → less precise observer.

When stimuli are presented sequentially, $n_1$ must be retained in memory while $n_2$ is
being perceived, which often leads to $\nu_1 > \nu_2$.  Unequal noise alone (without a
prior) does **not** shift the crossing point — it only changes the effective slope.  The
shift arises from the Bayesian prior, shown next.
"""),

code("""\
log_r = np.linspace(-2, 2, 300)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: noise level → precision
ax = axes[0]
for nu_val, label, c in [(.2, 'ν = 0.20 (sharp)', '#08519c'),
                           (.5, 'ν = 0.50', '#3182bd'),
                           (1., 'ν = 1.00 (noisy)', '#bdd7e7')]:
    ax.plot(log_r, scipy_norm.cdf(log_r / (np.sqrt(2)*nu_val)), lw=2, color=c, label=label)
ax.axhline(.5, ls='--', c='gray', lw=1); ax.axvline(0, ls='--', c='gray', lw=1)
ax.set_xlabel('log(n2/n1)'); ax.set_ylabel('P(chose n2)')
ax.set_title('Effect of noise level on precision')
ax.legend(); sns.despine(ax=ax)

# Right: asymmetric noise  ν1 vs ν2
ax = axes[1]
for (nu1, nu2), label, c in [((0.4, 0.4), 'Equal noise  \u03bd\u2081=\u03bd\u2082=0.4', '#1a9850'),
                               ((0.7, 0.3), 'Asymm. noise  \u03bd\u2081=0.7, \u03bd\u2082=0.3', '#d73027')]:
    ax.plot(log_r, scipy_norm.cdf(log_r / np.sqrt(nu1**2+nu2**2)), lw=2, color=c, label=label)
ax.axhline(.5, ls='--', c='gray', lw=1); ax.axvline(0, ls='--', c='gray', lw=1)
ax.set_xlabel('log(n2/n1)')
ax.set_title('Asymmetric noise (same total) → different slope')
ax.legend(fontsize=8); sns.despine(ax=ax)

plt.tight_layout()
"""),

md(r"""### The Bayesian prior and central tendency bias

Pure measurement noise is symmetric.  The *shift* of the psychometric curve arises
from a **Bayesian prior** over log-magnitudes.  The observer combines noisy evidence
$r \sim \mathcal{N}(\log n, \nu^2)$ with a prior $\mathcal{N}(\mu_0, \sigma_0^2)$ to
form a posterior:

$$\hat\mu = \underbrace{\frac{\sigma_0^2}{\sigma_0^2 + \nu^2}}_{\gamma}\, r
           + (1-\gamma)\,\mu_0, \qquad
\hat\sigma^2 = \frac{\sigma_0^2\,\nu^2}{\sigma_0^2 + \nu^2}$$

The posterior mean is pulled toward $\mu_0$ — the **central tendency effect**.  A
stronger prior (smaller $\sigma_0$) compresses representations more, shifting choices
systematically toward the prior mean.
"""),

code("""\
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

x = np.linspace(0.5, 5.5, 400)       # log-magnitude axis
log_n_true  = np.log(20)              # true stimulus: n=20
prior_mu    = np.log(10)              # prior centred at n=10
nu          = 0.4                     # measurement noise

for ax, (prior_sd, title) in zip(axes,
        [(1.0, 'Weak prior  (σ₀ = 1.0)'),
         (0.25, 'Strong prior  (σ₀ = 0.25)')]):

    gamma   = prior_sd**2 / (prior_sd**2 + nu**2)
    post_mu = prior_mu + gamma * (log_n_true - prior_mu)
    post_sd = np.sqrt(prior_sd**2 * nu**2 / (prior_sd**2 + nu**2))

    like  = scipy_norm.pdf(x, log_n_true, nu)
    prior = scipy_norm.pdf(x, prior_mu,  prior_sd)
    post  = scipy_norm.pdf(x, post_mu,   post_sd)
    mx    = max(like.max(), prior.max(), post.max())

    ax.fill_between(x, prior/mx, alpha=.3, color='#4393c3', label=f'Prior  μ₀={prior_mu:.2f}')
    ax.fill_between(x, like/mx,  alpha=.3, color='#d6604d', label=f'Evidence  log(n)={log_n_true:.2f}')
    ax.fill_between(x, post/mx,  alpha=.5, color='#4dac26', label=f'Posterior  μ̂={post_mu:.2f}')
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

md("""## Barreto-Garc\u00eda et al. (2023): Magnitude comparison task

64 participants viewed two sequentially presented coin clouds and judged which contained
more 1-CHF coins.  Reference magnitudes $n_1 \\in \\{5, 7, 10, 14, 20, 28\\}$; comparison
magnitudes $n_2$ varied widely.  We load the bundled dataset and inspect it.
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
# Compute log-ratio for each trial
data['log(n2/n1)'] = np.log(data['n2'] / data['n1'])
data['bin'] = (pd.cut(data['log(n2/n1)'], bins=12)
                 .apply(lambda x: x.mid).astype(float))

grouped = (data.groupby(['n1', 'bin'])['choice']
               .agg(['mean', 'count']).reset_index()
               .query('count >= 5'))

g = sns.FacetGrid(grouped, col='n1', col_wrap=3, height=3.2, aspect=1.1)
g.map(plt.scatter, 'bin', 'mean', s=25, alpha=.8, color='#2166ac')
g.map(sns.lineplot,  'bin', 'mean', color='#2166ac', lw=1.5)
for ax in g.axes.flat:
    ax.axhline(.5, ls='--', c='gray', lw=1)
    ax.axvline(.0, ls='--', c='gray', lw=1)
    ax.set_ylim(-.05, 1.05)
g.set_axis_labels('log(n2 / n1)', 'P(chose n2)')
g.set_titles('n1 = {col_name}')
plt.suptitle('Group-averaged psychometric curves — scale invariance confirmed', y=1.02)
plt.tight_layout()
"""),

md("""## Fitting `MagnitudeComparisonModel`

We fit the full hierarchical NLC model.  Free parameters per subject:
- **`n1_evidence_sd`** ($\\nu_1$): noise on the first-presented cloud (perception + memory retention)
- **`n2_evidence_sd`** ($\\nu_2$): noise on the second-presented cloud (perception only)

The group-level means (`_mu`) and standard deviations (`_sd`) are estimated jointly.
"""),

code("""\
model_mag = MagnitudeComparisonModel(paradigm=data)
model_mag.build_estimation_model(data=data, hierarchical=True, save_p_choice=True)
idata_mag = model_mag.sample(draws=200, tune=200, chains=4, progressbar=True)
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

code("""\
# Subject-level posterior means: ν₁ vs ν₂
params = ['n1_evidence_sd', 'n2_evidence_sd']
means  = (idata_mag.posterior[params]
                   .mean(dim=['chain', 'draw'])
                   .to_dataframe()
                   .reset_index())

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(means['n2_evidence_sd'], means['n1_evidence_sd'],
           alpha=.7, color='steelblue', s=40, zorder=3)
lim = max(means[params].max()) * 1.15
ax.plot([0, lim], [0, lim], 'k--', lw=1, label='ν₁ = ν₂')
ax.set_xlabel('Second-option noise  \u03bd\u2082  (n2_evidence_sd)')
ax.set_ylabel('First-option noise  \u03bd\u2081  (n1_evidence_sd)')
ax.set_title('Subject-level noise estimates  (\u03bd\u2081 > \u03bd\u2082 for most subjects, consistent with memory retention)')
ax.legend(); sns.despine(); plt.tight_layout()
"""),

md("""## Posterior predictive check

We draw predicted choice probabilities from the full posterior and overlay the 95 %
credible interval on the observed group-average data (one panel per $n_1$ value).
Good model fit means the shaded region covers the observed dots.
"""),

code("""\
ppc_df = model_mag.ppc(data, idata_mag, var_names=['p'])
ppc_p  = ppc_df.xs('p', level='variable')

data_ppc            = data.copy()
data_ppc['p_mean']  = ppc_p.mean(axis=1).values
data_ppc['p_lo']    = ppc_p.quantile(.025, axis=1).values
data_ppc['p_hi']    = ppc_p.quantile(.975, axis=1).values
data_ppc['bin']     = (pd.cut(data_ppc['log(n2/n1)'], 12)
                         .apply(lambda x: x.mid).astype(float))

g_ppc = (data_ppc.groupby(['n1', 'bin'])[['choice', 'p_mean', 'p_lo', 'p_hi']]
                  .mean().reset_index())

import matplotlib.patches as mpatches

def draw_ppc(data, **kwargs):
    ax = plt.gca()
    ax.fill_between(data['bin'], data['p_lo'], data['p_hi'],
                    color='steelblue', alpha=.25, label='95 % CI')
    ax.plot(data['bin'], data['p_mean'], color='steelblue', lw=2, label='Model mean')
    ax.scatter(data['bin'], data['choice'], color='steelblue', s=20, zorder=5, label='Observed')
    ax.axhline(.5, ls='--', c='gray', lw=1)
    ax.axvline(0, ls='--', c='gray', lw=1)
    ax.set_ylim(-.05, 1.05)

g = sns.FacetGrid(g_ppc, col='n1', col_wrap=3, height=3.2, aspect=1.1, sharey=True)
g.map_dataframe(draw_ppc)
g.set_axis_labels('log(n2 / n1)', 'P(chose n2)')
g.set_titles('n\u2081 = {col_name}')
for ax in g.axes.flat:
    sns.despine(ax=ax)

legend_handles = [
    mpatches.Patch(color='steelblue', alpha=.25, label='95 % CI'),
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

]

with open('lesson1.ipynb', 'w') as f:
    nbf.write(nb1, f)
print("lesson1.ipynb written")


# ─────────────────────────────────────────────────────────────────────────────
# Lesson 2 — Risky Choice: KLW, Format Comparison, and Key Correlations
# ─────────────────────────────────────────────────────────────────────────────

nb2 = nbf.v4.new_notebook()
nb2.cells = [

md(r"""# Lesson 2: Risky Choice and the Key Correlations (Barreto-Garc\u00eda et al., 2023)

## From magnitude to risk

The same 64 participants also made risky choices **outside** the scanner.  On each trial
they chose between a **sure payoff** ($p_1 = 1.0$, $n_1 \in \{5,7,10,14,20,28\}$) and a
**risky gamble** ($p_2 = 0.55$, $n_2$ varying).  Crucially, payoffs were shown in two
**formats** across separate blocks:

| Format | Representation |
|--------|----------------|
| `non-symbolic` | coin clouds — same format as the magnitude task |
| `symbolic` | Arabic numerals |

The paper's central argument: **the same perceptual noise that limits magnitude
discrimination also distorts the subjective value of risky gambles**, producing risk
aversion as a by-product of noisy numerical cognition.
"""),

code("""\
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy.stats import norm as scipy_norm, pearsonr
from bauer.utils.data import load_garcia2022
from bauer.models import RiskModel, RiskRegressionModel

data_risk = load_garcia2022(task='risk')
print(f"Subjects: {data_risk.index.get_level_values('subject').nunique()},  "
      f"Trials: {len(data_risk)},  "
      f"Formats: {data_risk.index.get_level_values('format').unique().tolist()}")
data_risk.head()
"""),

md(r"""## Visualise risky-choice data

The dashed vertical line marks the **risk-neutral expected-value threshold**
$\log(1/0.55) \approx 0.60$.  A risk-neutral observer's curve crosses 0.5 exactly there.
Curves crossing to the right → risk aversion; to the left → risk seeking.

Symbolic payoffs (Arabic numerals) produce a steeper, left-shifted curve, indicating
lower noise and less risk aversion relative to non-symbolic coin clouds.
"""),

code("""\
plot_data = data_risk.reset_index(level='format').copy()
plot_data['log(risky/safe)'] = np.log(plot_data['n2'] / plot_data['n1'])
plot_data['bin'] = (pd.cut(plot_data['log(risky/safe)'], bins=12)
                      .apply(lambda x: x.mid).astype(float))

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

md(r"""## Risk-neutral probability and the indifference point

In this experiment the risky option pays out with probability $p_\text{risky} = 0.55$.
The **risk-neutral probability** (RNP) is simply that number: 55 % — it is the winning
probability at which a risk-neutral, expected-value-maximising observer would make exactly
the same choices as our model predicts.

A risk-neutral observer is indifferent between the safe payoff $n_\text{safe}$ and the
risky payoff $n_\text{risky}$ when their expected values are equal:

$$1.0 \times n_\text{safe} = 0.55 \times n_\text{risky}
\;\Longrightarrow\;
n_\text{risky} = \frac{n_\text{safe}}{0.55} = \frac{1}{0.55}\, n_\text{safe}$$

In log-ratio space the **indifference point** is therefore

$$\delta^*_\text{risk-neutral} = \log\!\left(\frac{1}{0.55}\right) \approx 0.598$$

This is the $x$-value at which the psychometric curve of a risk-neutral observer crosses
$P = 0.5$.  An observer with $\delta^* > \log(1/0.55)$ is **risk-averse** — they need
the risky option to be even larger than the fair-expected-value threshold before they
prefer it.
"""),

code("""\
log_r = np.linspace(-.3, 2.2, 400)
ev_threshold = np.log(1/.55)
slope = 1.6   # fixed for illustration

fig, ax = plt.subplots(figsize=(7, 4.5))

for delta_star, label, c in [
        (ev_threshold, f'Risk-neutral  (\u03b4* = log(1/0.55) \u2248 {ev_threshold:.2f})', '#4393c3'),
        (.95, 'Mildly risk-averse  (\u03b4* = 0.95)', '#1a9850'),
        (1.45, 'Strongly risk-averse  (\u03b4* = 1.45)', '#d73027')]:
    p = scipy_norm.cdf((log_r - delta_star) * slope)
    ax.plot(log_r, p, color=c, lw=2.5, label=label)
    ax.axvline(delta_star, color=c, ls=':', lw=1.5, alpha=.7)

ax.axhline(.5, ls='--', c='gray', lw=1)
ax.set_xlabel('log(n_risky / n_safe)')
ax.set_ylabel('P(chose risky gamble)')
ax.set_title('\u03b4* = indifference point (curve crosses P = 0.5); risk-neutral: \u03b4* = log(1/0.55) \u2248 0.598')
ax.set_ylim(-.04, 1.04); ax.legend(fontsize=9)
sns.despine(); plt.tight_layout()
"""),

md(r"""## KLW model

The **KLW (Khaw-Li-Woodford)** model applies the same Bayesian-observer NLC framework
to risky choice.  Crucially, in Barreto-García et al. the safe and risky payoffs were
shown **simultaneously** on screen, so there is no reason to expect different noise
for the two options.  We therefore use a **single shared noise** $\nu$:

$$\hat\mu_k = \gamma \log n_k + (1-\gamma)\mu_0, \quad
\gamma = \frac{\sigma_0^2}{\sigma_0^2 + \nu^2}$$

The prior mean $\mu_0$ is set to the objective batch mean of $\log n$.  Higher noise
→ smaller $\gamma$ → stronger prior pull → larger $\delta^*$ → more risk aversion.

Free parameters per subject: `evidence_sd` ($\nu$), `prior_sd` ($\sigma_0$).
"""),

code("""\
model_klw = RiskModel(paradigm=data_risk, prior_estimate='klw',
                      fit_seperate_evidence_sd=False)
model_klw.build_estimation_model(data=data_risk, hierarchical=True, save_p_choice=True)
idata_klw = model_klw.sample(draws=200, tune=200, chains=4, progressbar=True)
"""),

code("""\
az.plot_posterior(
    idata_klw,
    var_names=['evidence_sd_mu', 'prior_sd_mu'],
    figsize=(8, 3.5),
)
plt.suptitle('Group-level KLW posteriors', y=1.04)
plt.tight_layout()
"""),

md("""## Format comparison: symbolic vs non-symbolic

We fit the KLW model separately to each format to isolate the noise difference.
"""),

code("""\
data_sym    = data_risk.xs('symbolic',     level='format')
data_nonsym = data_risk.xs('non-symbolic', level='format')

model_sym = RiskModel(paradigm=data_sym, prior_estimate='klw',
                      fit_seperate_evidence_sd=False)
model_sym.build_estimation_model(data=data_sym, hierarchical=True)
idata_sym = model_sym.sample(draws=150, tune=150, chains=2, progressbar=True)

model_nonsym = RiskModel(paradigm=data_nonsym, prior_estimate='klw',
                         fit_seperate_evidence_sd=False)
model_nonsym.build_estimation_model(data=data_nonsym, hierarchical=True)
idata_nonsym = model_nonsym.sample(draws=150, tune=150, chains=2, progressbar=True)
"""),

code("""\
# Compare group-level posteriors across formats
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
params_compare = ['evidence_sd', 'prior_sd']

for ax, param in zip(axes, params_compare):
    sym_vals    = idata_sym.posterior[f'{param}_mu'].values.ravel()
    nonsym_vals = idata_nonsym.posterior[f'{param}_mu'].values.ravel()

    df_comp = pd.DataFrame({
        'value':  np.concatenate([sym_vals, nonsym_vals]),
        'format': ['symbolic']*len(sym_vals) + ['non-symbolic']*len(nonsym_vals),
    })
    sns.violinplot(data=df_comp, x='format', y='value', cut=0,
                   palette={'symbolic': '#1f78b4', 'non-symbolic': '#d95f02'},
                   inner='box', ax=ax)
    ax.set_title(f'Group-level  {param}')
    ax.set_xlabel('')
    sns.despine(ax=ax)

plt.suptitle('KLW model: symbolic vs non-symbolic', fontsize=13, y=1.02)
plt.tight_layout()
"""),

md(r"""### Regression model: format effect on noise

`RiskRegressionModel` with a patsy formula lets the shared noise `evidence_sd` vary
trial-by-trial as a function of covariates — here, `C(format)`.  This gives a direct
posterior on the **format effect on precision**.
"""),

code("""\
data_reg = data_risk.reset_index(level='format')   # format as a column

model_reg = RiskRegressionModel(
    paradigm=data_reg,
    regressors={'evidence_sd': 'C(format)'},
    prior_estimate='klw',
    fit_seperate_evidence_sd=False,
)
model_reg.build_estimation_model(data=data_reg, hierarchical=True)
idata_reg = model_reg.sample(draws=150, tune=150, chains=2, progressbar=True)
"""),

code("""\
# Posterior of evidence_sd at each format condition
conditions = pd.DataFrame({'format': ['symbolic', 'non-symbolic']})
pars_cond  = model_reg.get_conditionwise_parameters(idata_reg, conditions, group=True)

sd_sym    = pars_cond.loc['evidence_sd']['symbolic'].values
sd_nonsym = pars_cond.loc['evidence_sd']['non-symbolic'].values

fig, ax = plt.subplots(figsize=(5.5, 4))
for vals, label, c in [(sd_sym,    'symbolic',    '#1f78b4'),
                        (sd_nonsym, 'non-symbolic', '#d95f02')]:
    az.plot_kde(vals, label=label, plot_kwargs={'color': c, 'lw': 2}, ax=ax)
ax.set_xlabel('evidence_sd  (group level)')
ax.set_ylabel('Posterior density')
ax.set_title('Regression model: noise by format  (symbolic = lower noise)')
ax.legend(); sns.despine(); plt.tight_layout()
"""),

md(r"""## Key result: perceptual noise predicts risk aversion

With a single shared noise $\nu$ and prior SD $\sigma_0$, the implied indifference
log-ratio simplifies to

$$\delta^* = \frac{\log(1/p_\text{risky})}{\gamma}
= \log(1/0.55)\cdot\frac{\sigma_0^2 + \nu^2}{\sigma_0^2}$$

Noisier observers (high $\nu$) have a larger $\delta^*$ and are more risk-averse.
And because the single decision noise $\nu$ (risk task) correlates with perceptual
noise $\nu_2$ (magnitude task), **perceptual precision measured in the scanner predicts
risk aversion in a separate behavioural session** — the central result of Barreto-Garc\u00eda et al.

We show 94 % HDI crossbars per subject.
"""),

code("""\
# ── Helper: extract posterior mean and 94 % HDI per subject ──────────────────
def posterior_summary(idata, var):
    arr  = idata.posterior[var].values           # (chains, draws, subjects)
    subj = idata.posterior[var].coords['subject'].values
    vals = arr.reshape(-1, len(subj))            # (samples, subjects)
    return pd.DataFrame({
        'subject': subj,
        'mean': vals.mean(0),
        'lo':   np.percentile(vals, 3, 0),       # ≈ 94 % HDI lower
        'hi':   np.percentile(vals, 97, 0),      # ≈ 94 % HDI upper
    })

# ── Magnitude noise (n2 = perceptual noise, no WM contamination) ─────────────
from bauer.utils.data import load_garcia2022
data_mag = load_garcia2022(task='magnitude')
from bauer.models import MagnitudeComparisonModel
model_mag_l2 = MagnitudeComparisonModel(paradigm=data_mag)
model_mag_l2.build_estimation_model(data=data_mag, hierarchical=True, save_p_choice=False)
idata_mag_l2 = model_mag_l2.sample(draws=200, tune=200, chains=4, progressbar=True)

df_nu_mag = posterior_summary(idata_mag_l2, 'n2_evidence_sd').rename(
                columns={'mean': 'nu_mag', 'lo': 'nu_mag_lo', 'hi': 'nu_mag_hi'})

# ── Decision noise (single evidence_sd from KLW) ─────────────────────────────
df_nu_risk = posterior_summary(idata_klw, 'evidence_sd').rename(
                columns={'mean': 'nu_risk', 'lo': 'nu_risk_lo', 'hi': 'nu_risk_hi'})

# ── Implied δ* from KLW posterior ────────────────────────────────────────────
nu_arr   = idata_klw.posterior['evidence_sd'].values.reshape(-1, len(df_nu_risk))
prsd_arr = idata_klw.posterior['prior_sd'].values.reshape(-1, len(df_nu_risk))

gamma      = prsd_arr**2 / (prsd_arr**2 + nu_arr**2)
delta_star = np.log(1/.55) / gamma

df_delta = pd.DataFrame({
    'subject':    idata_klw.posterior['evidence_sd'].coords['subject'].values,
    'delta_mean': delta_star.mean(0),
    'delta_lo':   np.percentile(delta_star, 3, 0),
    'delta_hi':   np.percentile(delta_star, 97, 0),
})

# ── Merge on subject ──────────────────────────────────────────────────────────
df_corr = (df_nu_mag
           .merge(df_nu_risk, on='subject')
           .merge(df_delta,   on='subject'))
print(f"Aligned subjects: {len(df_corr)}")
"""),

code("""\
def scatter_hdi(ax, x, y, xerr, yerr, color, xlabel, ylabel, title, hline=None):
    \"\"\"Scatter plot with HDI crossbars and a regression line.\"\"\"
    ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                fmt='o', ms=4, alpha=.55, elinewidth=.7, capsize=2,
                color=color, ecolor=color)
    r, p = pearsonr(x, y)
    m, b = np.polyfit(x, y, 1)
    xs   = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, m*xs + b, '--', color=color, lw=1.5, alpha=.8,
            label=f'r = {r:.2f} (p={p:.3f})')
    if hline is not None:
        ax.axhline(hline, ls=':', c='gray', lw=1.5, label='Risk-neutral')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=9); sns.despine(ax=ax)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Perceptual noise vs decision noise
scatter_hdi(
    axes[0],
    x    = df_corr['nu_mag'],
    y    = df_corr['nu_risk'],
    xerr = np.array([df_corr['nu_mag']  - df_corr['nu_mag_lo'],
                     df_corr['nu_mag_hi'] - df_corr['nu_mag']]),
    yerr = np.array([df_corr['nu_risk']  - df_corr['nu_risk_lo'],
                     df_corr['nu_risk_hi'] - df_corr['nu_risk']]),
    color   = '#4393c3',
    xlabel  = 'Perceptual noise  \u03bd\u2082  (magnitude task)',
    ylabel  = 'Decision noise  \u03bd  (risk task)',
    title   = 'Perceptual \u2194 decision noise',
)

# 2. Perceptual noise vs \u03b4*
scatter_hdi(
    axes[1],
    x    = df_corr['nu_mag'],
    y    = df_corr['delta_mean'],
    xerr = np.array([df_corr['nu_mag']   - df_corr['nu_mag_lo'],
                     df_corr['nu_mag_hi'] - df_corr['nu_mag']]),
    yerr = np.array([df_corr['delta_mean'] - df_corr['delta_lo'],
                     df_corr['delta_hi']   - df_corr['delta_mean']]),
    color  = '#d6604d',
    xlabel = 'Perceptual noise  \u03bd\u2082  (magnitude task)',
    ylabel = 'Implied risk aversion  \u03b4*',
    title  = 'Perceptual noise \u2192 risk aversion',
    hline  = np.log(1/.55),
)

# 3. Decision noise vs \u03b4*  (within-task)
scatter_hdi(
    axes[2],
    x    = df_corr['nu_risk'],
    y    = df_corr['delta_mean'],
    xerr = np.array([df_corr['nu_risk']  - df_corr['nu_risk_lo'],
                     df_corr['nu_risk_hi'] - df_corr['nu_risk']]),
    yerr = np.array([df_corr['delta_mean'] - df_corr['delta_lo'],
                     df_corr['delta_hi']   - df_corr['delta_mean']]),
    color  = '#1a9850',
    xlabel = 'Decision noise  \u03bd  (risk task)',
    ylabel = 'Implied risk aversion  \u03b4*',
    title  = 'Decision noise \u2192 risk aversion  (within-task)',
    hline  = np.log(1/.55),
)

plt.suptitle('Key result: noise predicts risk aversion (bars = 94 % HDI per subject)',
             fontsize=13, y=1.02)
plt.tight_layout()
"""),

]

with open('lesson2.ipynb', 'w') as f:
    nbf.write(nb2, f)
print("lesson2.ipynb written")


# ─────────────────────────────────────────────────────────────────────────────
# Lesson 3 — de Hollander et al. (2024): EU vs KLW vs full prior
# ─────────────────────────────────────────────────────────────────────────────

nb3 = nbf.v4.new_notebook()
nb3.cells = [

md(r"""# Lesson 3: Stake effects and presentation order — de Hollander et al. (2024)

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
    \"\"\"Add log_ratio, chose_risky, n_safe, order flag, and binned columns.\"\"\"
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
                             .apply(lambda x: x.mid).astype(float))
    df['n_safe_bin']    = pd.qcut(df['n_safe'], q=3, labels=['Low stakes', 'Mid stakes', 'High stakes'])
    return df

df_dot_p = prep_df(df_dot)
df_sym_p = prep_df(df_sym)
"""),

md(r"""## Presentation-order x stake-size interaction

Each panel shows P(chose risky) as a function of the log risky/safe magnitude ratio,
split by safe-option stake tertile.  The left column shows trials where the risky option
came first; the right column shows trials where the safe option came first.

The dashed vertical line marks the risk-neutral indifference point log(1/0.55).
"""),

code("""\
stake_pal = {'Low stakes': '#4C72B0', 'Mid stakes': '#DD8452', 'High stakes': '#55A868'}

def plot_interaction(df_p, axes_row, task_label):
    \"\"\"Plot stake x order interaction, one axis per order condition.\"\"\"
    for ax, order_val in zip(axes_row, ['Risky first', 'Safe first']):
        sub  = df_p[df_p['order'] == order_val]
        subj = (sub.groupby(['subject', 'log_ratio_bin', 'n_safe_bin'])['chose_risky']
                   .mean().reset_index())
        subj = subj[subj.groupby(['log_ratio_bin', 'n_safe_bin'])
                        ['subject'].transform('count') >= 3]
        hue_order = ['Low stakes', 'Mid stakes', 'High stakes']
        sns.lineplot(data=subj, x='log_ratio_bin', y='chose_risky',
                     hue='n_safe_bin', style='n_safe_bin',
                     hue_order=hue_order, style_order=hue_order,
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

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
plot_interaction(df_dot_p, axes[0], 'Dot clouds (3T + 7T)')
plot_interaction(df_sym_p, axes[1], 'Symbolic (Arabic numerals)')
plt.suptitle('Presentation-order \u00d7 safe-stake interaction', fontsize=13, y=1.01)
plt.tight_layout()
"""),

md("""## Fit three models — dot-cloud data

Hierarchical MCMC, 100 draws / 100 tune / 2 chains.
"""),

code("""\
# ── 1. Expected Utility ──────────────────────────────────────────────────────
model_eu = ExpectedUtilityRiskModel(paradigm=df_dot)
model_eu.build_estimation_model(data=df_dot, hierarchical=True, save_p_choice=True)
idata_eu = model_eu.sample(draws=100, tune=100, chains=2, progressbar=True)
"""),

code("""\
# ── 2. KLW (shared noise, shared prior) ─────────────────────────────────────
model_klw = RiskModel(paradigm=df_dot, prior_estimate='klw',
                      fit_seperate_evidence_sd=False)
model_klw.build_estimation_model(data=df_dot, hierarchical=True, save_p_choice=True)
idata_klw = model_klw.sample(draws=100, tune=100, chains=2, progressbar=True)
"""),

code("""\
# ── 3. PMCM (separate noise + separate priors) ────────────────────
model_full = RiskModel(paradigm=df_dot, prior_estimate='full',
                       fit_seperate_evidence_sd=True)
model_full.build_estimation_model(data=df_dot, hierarchical=True, save_p_choice=True)
idata_full = model_full.sample(draws=100, tune=100, chains=2, progressbar=True)
"""),

md("""## Posterior predictives — dot-cloud data

Dots = observed group average; line + shading = model mean and 95 % posterior interval.
"""),

code("""\
def add_model_ppc(df_orig, df_prepped, model, idata, model_name, n_ppc_samples=50):
    \"\"\"Return df_prepped with added posterior-predictive columns.\"\"\"
    ppc_df = model.ppc(df_orig, idata, var_names=['p'])
    ppc_p  = ppc_df.xs('p', level='variable')
    cols   = np.random.choice(ppc_p.columns,
                               size=min(n_ppc_samples, ppc_p.shape[1]),
                               replace=False)
    ppc_p  = ppc_p[cols]
    risky_first = df_prepped['risky_first'].values
    p_risky = np.where(risky_first[:, None], 1 - ppc_p.values, ppc_p.values)
    df_out = df_prepped.copy()
    df_out['p_mean'] = p_risky.mean(1)
    df_out['p_lo']   = np.percentile(p_risky, 2.5,  axis=1)
    df_out['p_hi']   = np.percentile(p_risky, 97.5, axis=1)
    df_out['model']  = model_name
    return df_out


def plot_ppc_interaction(df_pred, model_name, axes_row):
    hue_order = ['Low stakes', 'Mid stakes', 'High stakes']
    for ax, order_val in zip(axes_row, ['Risky first', 'Safe first']):
        sub  = df_pred[df_pred['order'] == order_val]
        obs  = sub.groupby(['n_safe_bin', 'log_ratio_bin'])['chose_risky'].mean().reset_index()
        pred = (sub.groupby(['n_safe_bin', 'log_ratio_bin'])[['p_mean', 'p_lo', 'p_hi']]
                   .mean().reset_index())
        for sbin in hue_order:
            o = obs[obs['n_safe_bin']  == sbin]
            p = pred[pred['n_safe_bin'] == sbin]
            if len(o) == 0:
                continue
            c = stake_pal[sbin]
            ax.fill_between(p['log_ratio_bin'], p['p_lo'], p['p_hi'],
                            color=c, alpha=.20)
            ax.plot(p['log_ratio_bin'], p['p_mean'], color=c, lw=2, label=sbin)
            ax.scatter(o['log_ratio_bin'], o['chose_risky'],
                       color=c, s=25, zorder=5, alpha=.85)
        ax.axhline(.5,            ls='--', c='gray', lw=1)
        ax.axvline(np.log(1/.55), ls='--', c='#333333', lw=1.5)
        ax.set_ylim(-.05, 1.05)
        ax.set_title(f'{model_name} \u2014 {order_val}', fontsize=9)
        ax.set_xlabel('log(risky / safe)')
        ax.set_ylabel('P(chose risky)')
        ax.legend(title='Safe stake', fontsize=7, loc='upper left')
        sns.despine(ax=ax)


fig, axes = plt.subplots(3, 2, figsize=(12, 13), sharey=True)
for (mdl, idat, name), row in zip(
        [(model_eu,   idata_eu,   'EU'),
         (model_klw,  idata_klw,  'KLW'),
         (model_full, idata_full, 'PMCM')],
        axes):
    df_pred = add_model_ppc(df_dot, df_dot_p, mdl, idat, name)
    plot_ppc_interaction(df_pred, name, row)

plt.suptitle('Posterior predictive checks \u2014 dot-cloud data  (dots = observed, shading = 95 % CI)',
             fontsize=11, y=1.01)
plt.tight_layout()
"""),

md("""## Fit three models — symbolic data
"""),

code("""\
# ── 1. EU ────────────────────────────────────────────────────────────────────
model_eu_sym = ExpectedUtilityRiskModel(paradigm=df_sym)
model_eu_sym.build_estimation_model(data=df_sym, hierarchical=True, save_p_choice=True)
idata_eu_sym = model_eu_sym.sample(draws=100, tune=100, chains=2, progressbar=True)
"""),

code("""\
# ── 2. KLW ───────────────────────────────────────────────────────────────────
model_klw_sym = RiskModel(paradigm=df_sym, prior_estimate='klw',
                           fit_seperate_evidence_sd=False)
model_klw_sym.build_estimation_model(data=df_sym, hierarchical=True, save_p_choice=True)
idata_klw_sym = model_klw_sym.sample(draws=100, tune=100, chains=2, progressbar=True)
"""),

code("""\
# ── 3. PMCM ────────────────────────────────────────────────────────
model_full_sym = RiskModel(paradigm=df_sym, prior_estimate='full',
                            fit_seperate_evidence_sd=True)
model_full_sym.build_estimation_model(data=df_sym, hierarchical=True, save_p_choice=True)
idata_full_sym = model_full_sym.sample(draws=100, tune=100, chains=2, progressbar=True)
"""),

md("""## Posterior predictives — symbolic data
"""),

code("""\
fig, axes = plt.subplots(3, 2, figsize=(12, 13), sharey=True)
for (mdl, idat, name), row in zip(
        [(model_eu_sym,   idata_eu_sym,   'EU'),
         (model_klw_sym,  idata_klw_sym,  'KLW'),
         (model_full_sym, idata_full_sym, 'PMCM')],
        axes):
    df_pred = add_model_ppc(df_sym, df_sym_p, mdl, idat, name)
    plot_ppc_interaction(df_pred, name, row)

plt.suptitle('Posterior predictive checks \u2014 symbolic data  (dots = observed, shading = 95 % CI)',
             fontsize=11, y=1.01)
plt.tight_layout()
"""),

md(r"""## Parameter interpretation: $\nu_1$ vs $\nu_2$

For the full-prior NLC model we extract subject-level posterior means for the two noise
parameters.  If the first-presented option accumulates extra noise, we expect $\nu_1 >
\nu_2$ and most subjects to lie above the diagonal.
"""),

code("""\
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (idat, label) in zip(axes,
        [(idata_full,     'Dot clouds'),
         (idata_full_sym, 'Symbolic')]):
    nu1 = idat.posterior['n1_evidence_sd'].mean(('chain', 'draw')).values
    nu2 = idat.posterior['n2_evidence_sd'].mean(('chain', 'draw')).values
    lim = max(nu1.max(), nu2.max()) * 1.15
    ax.scatter(nu2, nu1, alpha=.7, s=45, color='#2166ac', zorder=3)
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='\u03bd\u2081 = \u03bd\u2082')
    ax.set_xlabel('Second-option noise  \u03bd\u2082')
    ax.set_ylabel('First-option noise  \u03bd\u2081')
    ax.set_title(label)
    ax.legend(fontsize=9); sns.despine(ax=ax)

plt.suptitle('Subject-level noise estimates from PMCM', fontsize=12, y=1.02)
plt.tight_layout()
"""),

]

with open('lesson3.ipynb', 'w') as f:
    nbf.write(nb3, f)
print("lesson3.ipynb written")


# ─────────────────────────────────────────────────────────────────────────────
# Lesson 4 — Flexible noise curves: why and when they help
# ─────────────────────────────────────────────────────────────────────────────

nb4 = nbf.v4.new_notebook()
nb4.cells = [

md(r"""# Lesson 4: Flexible Noise Curves — the `FlexibleNoiseRiskModel`

## Motivation: noise in natural space vs log space

The PMCM (lesson 3) assumes a single fixed noise level per presentation position:
$\nu_1$ for the first option and $\nu_2$ for the second.  These scalars are defined in
**natural (linear) payoff space**.

The NLC model predicts **scale invariance**: noise is constant on a *logarithmic* scale.
When translated back to natural space, constant log-scale noise implies that natural-space
noise grows **linearly** with magnitude — this is just Weber's law:

$$\nu_\text{natural}(n) = \nu_\text{log} \times n$$

So the scale-invariant prediction for the flexible model is a *linear* curve $\nu(n)
\propto n$, not a flat one.  A flat curve in natural space would mean constant absolute
discriminability — which would actually imply *improving* relative precision at higher
magnitudes (a violation of Weber's law in the opposite direction).

The **Flexible Noise Model** replaces fixed scalars with **polynomial curves**
$\nu_k(n)$, allowing the data to reveal the actual noise-vs-magnitude relationship:

$$\nu_k(n) = \text{softplus}\!\left(\sum_{j=0}^{p} \beta_{k,j}\, \phi_j(n)\right)$$

where $\phi_j$ are polynomial basis functions and $p$ is the polynomial order (default 5).

- **Linear posterior** $\Rightarrow$ scale invariance (Weber's law) holds
- **Sub-linear** $\Rightarrow$ better-than-Weber precision at large magnitudes
- **Super-linear or curved** $\Rightarrow$ violations (e.g., compressed representation at extremes)
"""),

code("""\
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from bauer.utils.data import load_dehollander2024
from bauer.models import RiskModel, FlexibleNoiseRiskModel

# Use dot-cloud data (both sessions) as in lesson 3
df_dot = load_dehollander2024(task='dotcloud', sessions=['3t2', '7t2'])
print(f"Subjects: {df_dot.index.get_level_values('subject').nunique()},  Trials: {len(df_dot)}")

# Prep helper (same as lesson 3)
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
                             .apply(lambda x: x.mid).astype(float))
    df['n_safe_bin']    = pd.qcut(df['n_safe'], q=3,
                                   labels=['Low stakes', 'Mid stakes', 'High stakes'])
    return df

df_dot_p = prep_df(df_dot)
"""),

md(r"""## Illustrating flexible noise curves

Before fitting, we show what different noise-curve shapes look like and how they
would shift the psychometric function at different magnitude levels.
"""),

code("""\
from scipy.stats import norm as scipy_norm

n_vals     = np.linspace(5, 45, 200)
nu_log     = 0.30          # log-scale Weber fraction
log_ratios = np.linspace(-1.5, 1.5, 300)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: noise curves in natural space — linear = Weber's law null
ax = axes[0]
noise_curves = [
    ('Linear  \u03bd(n) = \u03bd\u2080\u00b7n  [Weber / scale-invariant]',
     nu_log * n_vals / n_vals.mean(),        '#2166ac', '-'),
    ('Flat  \u03bd(n) = const  [better precision at large n]',
     np.full_like(n_vals, nu_log),           '#d73027', '--'),
    ('Super-linear  [worse precision at large n]',
     nu_log * (n_vals / n_vals.mean())**1.8, '#1a9850', ':'),
]
for label, nu_n, c, ls in noise_curves:
    ax.plot(n_vals, nu_n, lw=2.5, color=c, ls=ls, label=label)
ax.set_xlabel('Payoff magnitude  n  (natural space)')
ax.set_ylabel('Noise  \u03bd(n)')
ax.set_title('Noise-vs-magnitude curves in natural space')
ax.legend(fontsize=8.5); sns.despine(ax=ax)

# Right: linear = constant log-ratio discrimination (Weber law)
ax = axes[1]
for n_ref, c, ls in [(8, '#4393c3', '-'), (20, '#1a9850', '--'), (36, '#d73027', ':')]:
    nu_lin  = nu_log * n_ref / n_vals.mean()   # linear Weber noise
    nu_flat = nu_log                            # flat (constant) noise
    p_lin  = scipy_norm.cdf(log_ratios / (np.sqrt(2) * nu_lin))
    p_flat = scipy_norm.cdf(log_ratios / (np.sqrt(2) * nu_flat))
    ax.plot(log_ratios, p_lin,  color=c, lw=2.5, ls=ls, label=f'Linear \u03bd (n={n_ref})')
    ax.plot(log_ratios, p_flat, color=c, lw=1.2, ls=ls, alpha=.4)
ax.axhline(.5, ls='--', c='gray', lw=1)
ax.axvline(0,  ls='--', c='gray', lw=1)
ax.text(0.97, 0.07, 'Faint: flat \u03bd (constant)', transform=ax.transAxes,
        ha='right', fontsize=8, color='gray')
ax.set_xlabel('log(n2 / n1)')
ax.set_ylabel('P(chose n2)')
ax.set_title('Linear noise (Weber): psychometric slope varies with n')
ax.legend(fontsize=8.5); sns.despine(ax=ax)

plt.suptitle('Natural-space noise \u2014 linear = Weber / scale-invariant baseline',
             fontsize=12, y=1.02)
plt.tight_layout()
"""),

md("""## Fit models — PMCM vs Flexible Noise

We fit both the PMCM (fixed noise) and the Flexible Noise model on the dot-cloud data
so we can directly compare them.
"""),

code("""\
# ── PMCM (fixed noise) ───────────────────────────────────────────────────────
model_pmcm = RiskModel(paradigm=df_dot, prior_estimate='full',
                        fit_seperate_evidence_sd=True)
model_pmcm.build_estimation_model(data=df_dot, hierarchical=True, save_p_choice=True)
idata_pmcm = model_pmcm.sample(draws=100, tune=100, chains=2, progressbar=True)
"""),

code("""\
# ── Flexible Noise model (polynomial noise curves) ───────────────────────────
model_flex = FlexibleNoiseRiskModel(paradigm=df_dot, prior_estimate='full',
                                     fit_seperate_evidence_sd=True, polynomial_order=5)
model_flex.build_estimation_model(paradigm=df_dot, hierarchical=True, save_p_choice=True)
idata_flex = model_flex.sample(draws=100, tune=100, chains=2, progressbar=True)
"""),

md(r"""## Posterior noise curves $\nu_k(n)$

The group-level posterior noise curves show whether noise is flat (like PMCM assumes)
or varies systematically with magnitude.  The shaded band is the 95 % posterior interval
across posterior samples.
"""),

code("""\
# Get group-level noise curves from the flexible model
x = np.linspace(df_dot[['n1', 'n2']].min().min(),
                df_dot[['n1', 'n2']].max().max(), 100)

sd_curves = model_flex.get_sd_curve(idata=idata_flex, variable='both',
                                     group=True, data=df_dot.reset_index())

# sd_curves is (posterior_samples x x_points)
n1_samples = sd_curves['n1_evidence_sd'].values  # shape: (samples, 100)
n2_samples = sd_curves['n2_evidence_sd'].values

# PMCM fixed noise posteriors (scalar per sample)
nu1_pmcm = idata_pmcm.posterior['n1_evidence_sd_mu'].values.ravel()
nu2_pmcm = idata_pmcm.posterior['n2_evidence_sd_mu'].values.ravel()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = {'flex': '#d73027', 'pmcm': '#4393c3'}

for ax, (samples, nu_pmcm, label_flex, label_pmcm) in zip(
        axes,
        [(n1_samples, nu1_pmcm, 'FlexNoise  \u03bd\u2081(n)', 'PMCM  \u03bd\u2081 (fixed)'),
         (n2_samples, nu2_pmcm, 'FlexNoise  \u03bd\u2082(n)', 'PMCM  \u03bd\u2082 (fixed)')]):
    # Posterior mean and 95% CI for flexible model
    mean = samples.mean(0)
    lo   = np.percentile(samples, 2.5, axis=0)
    hi   = np.percentile(samples, 97.5, axis=0)
    ax.fill_between(x, lo, hi, alpha=.25, color=colors['flex'])
    ax.plot(x, mean, lw=2.5, color=colors['flex'], label=label_flex)
    # PMCM fixed noise
    ax.axhline(nu_pmcm.mean(), ls='--', lw=2, color=colors['pmcm'],
               label=f'{label_pmcm} = {nu_pmcm.mean():.2f}')
    ax.fill_between(x,
                    np.percentile(nu_pmcm, 2.5),
                    np.percentile(nu_pmcm, 97.5),
                    alpha=.15, color=colors['pmcm'])
    ax.set_xlabel('Payoff magnitude  n')
    ax.set_ylabel('Noise  \u03bd')
    ax.legend(fontsize=9); sns.despine(ax=ax)

axes[0].set_title('First-option noise  \u03bd\u2081(n)')
axes[1].set_title('Second-option noise  \u03bd\u2082(n)')
plt.suptitle('Flexible vs fixed noise (group level, 95 % posterior interval)',
             fontsize=12, y=1.02)
plt.tight_layout()
"""),

md("""## Posterior predictive comparison

We reuse the PPC helper from lesson 3 and overlay both models' predictions against the
observed presentation-order × stake-size interaction.
"""),

code("""\
stake_pal = {'Low stakes': '#4C72B0', 'Mid stakes': '#DD8452', 'High stakes': '#55A868'}

def add_model_ppc(df_orig, df_prepped, model, idata, model_name, n_ppc_samples=50):
    ppc_df = model.ppc(df_orig, idata, var_names=['p'])
    ppc_p  = ppc_df.xs('p', level='variable')
    cols   = np.random.choice(ppc_p.columns,
                               size=min(n_ppc_samples, ppc_p.shape[1]),
                               replace=False)
    ppc_p  = ppc_p[cols]
    risky_first = df_prepped['risky_first'].values
    p_risky = np.where(risky_first[:, None], 1 - ppc_p.values, ppc_p.values)
    df_out = df_prepped.copy()
    df_out['p_mean'] = p_risky.mean(1)
    df_out['p_lo']   = np.percentile(p_risky, 2.5,  axis=1)
    df_out['p_hi']   = np.percentile(p_risky, 97.5, axis=1)
    df_out['model']  = model_name
    return df_out


def plot_ppc_row(df_pred, model_name, axes_row):
    hue_order = ['Low stakes', 'Mid stakes', 'High stakes']
    for ax, order_val in zip(axes_row, ['Risky first', 'Safe first']):
        sub  = df_pred[df_pred['order'] == order_val]
        obs  = sub.groupby(['n_safe_bin', 'log_ratio_bin'])['chose_risky'].mean().reset_index()
        pred = (sub.groupby(['n_safe_bin', 'log_ratio_bin'])[['p_mean', 'p_lo', 'p_hi']]
                   .mean().reset_index())
        for sbin in hue_order:
            o = obs[obs['n_safe_bin']  == sbin]
            p = pred[pred['n_safe_bin'] == sbin]
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
        ax.set_title(f'{model_name} \u2014 {order_val}', fontsize=9)
        ax.set_xlabel('log(risky / safe)'); ax.set_ylabel('P(chose risky)')
        ax.legend(title='Safe stake', fontsize=7, loc='upper left')
        sns.despine(ax=ax)


fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
for (mdl, idat, name), row in zip(
        [(model_pmcm, idata_pmcm, 'PMCM (fixed noise)'),
         (model_flex, idata_flex, 'Flexible Noise')],
        axes):
    df_pred = add_model_ppc(df_dot, df_dot_p, mdl, idat, name)
    plot_ppc_row(df_pred, name, row)

plt.suptitle('Posterior predictive comparison: PMCM vs Flexible Noise',
             fontsize=12, y=1.01)
plt.tight_layout()
"""),

md(r"""## Take-aways

- If the posterior noise curves are **flat**, the flexible model reduces to the PMCM
  and adds no explanatory value.
- If the curves are **rising or curved**, this indicates that noise is not constant
  across magnitudes — violating the pure scale-invariance prediction of the NLC model.
- The flexible model's PPC will tend to capture the stake-dependent spread more
  accurately when noise truly varies, because it can assign different noise to high-
  and low-magnitude safe/risky options within the same trial.
- In practice, compare model fit with `az.compare({'PMCM': idata_pmcm, 'Flex': idata_flex})`
  after computing log-likelihoods (requires `idata_kwargs={'log_likelihood': True}` in
  `model.sample()`).
"""),

]

with open('lesson4.ipynb', 'w') as f:
    nbf.write(nb4, f)
print("lesson4.ipynb written")
