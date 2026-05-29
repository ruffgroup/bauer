# Risky-choice examples for Alina

Three model fits to **Alina's `ds-numloss` pilot data** (two pilot subjects,
gain + loss risky-choice trials, with reaction times). Designed as a
template that adapts cleanly to the full N when those subjects come in.

## What's here

| File | Purpose |
|------|---------|
| `data/pilot_data.tsv` | Combined trial-level dataset (798 trials × 2 pilot subjects, gains + losses, with RTs). Pre-baked from BIDS events TSVs at `/shares/zne.uzh/aldavy/ds-numloss/behavior/`. |
| `_make_pilot_dataset.py` | One-shot script that built `pilot_data.tsv`. Kept for provenance; not used by the example. |
| `helpers.py` | Shared plotting + summary helpers (natural-scale extraction, forest plot, contrast KDE, per-subject panels, EV-neutral line). |
| `fit_psychometric_simple.py` | `PsychophysicalLapseRegressionModel` — simple log-EU psychometric with `nu` (sensitivity), `bias` (risk premium), `p_lapse`, all regressed on domain. |
| `fit_psychometric_bayes.py` | `RiskLapseRegressionModel` — Bayesian observer with separate `n1_evidence_sd`/`n2_evidence_sd` (captures order effects) + lapses. |
| `fit_ddm.py` | `DDMRiskRegressionModel` — joint choice + RT likelihood; **hierarchical by default** (`--per-subject` to opt out), regressing `n{1,2}_evidence_sd` and `a` on domain (`t0` is a single param). |
| `notebook_1_psychometric.ipynb` | Walkthrough of both psychometric fits. |
| `notebook_2_ddm.ipynb` | DDM fit + explanation of how probability enters the drift. |
| `slurm/run_*.sh` | sbatch wrappers for the three fits on `sciencecluster`. |

## Two data conventions

Both fits and notebooks use these conventions consistently:

1. **Keep `n1, n2, p1, p2` in presentation order.** Don't canonicalise on
   which option is the gamble — bauer infers that from `p1 < p2`.
   Keeping presentation order lets `n1_evidence_sd` ≠ `n2_evidence_sd`
   absorb the working-memory degradation that affects the first-presented
   option.
2. **Flip `choice` on loss trials** (`prepare(df)` in each fit script).
   Gain trials ask "pick the larger EV"; loss trials ask "pick the
   smaller |EV|" — opposite perceptual questions. After flipping
   `choice` in losses, both domains share a single drift formula
   (`drift > 0 ⟹ chose option 2 ⟺ took the EV-improving action`), the
   psychometric slope direction is consistent (always increasing in
   $\log(EU_2/EU_1)$), and the per-subject indifference points line up
   across domains — which means a single shared `nu` / shared lapse can
   describe both domains, and gain/loss contrasts on those parameters
   reflect genuine domain differences rather than slope-direction
   artefacts.

## Running the pipeline

```bash
# 1. Local sanity check (~30 s per fit)
python fit_psychometric_simple.py --draws 200 --tune 200 --backend pymc
python fit_psychometric_bayes.py  --draws 200 --tune 200 --backend pymc
python fit_ddm.py                 --draws 200 --tune 200 --backend pymc

# 2. Production fits on the cluster
ssh sciencecluster '
  cd ~/git/bauer/examples/for_alina &&
  sbatch slurm/run_psychometric_simple.sh &&
  sbatch slurm/run_psychometric_bayes.sh  &&
  sbatch slurm/run_ddm.sh
'

# 3. Pull the InferenceData back and open the notebooks
rsync sciencecluster:git/bauer/examples/for_alina/results/*.nc ./results/
jupyter notebook notebook_1_psychometric.ipynb
jupyter notebook notebook_2_ddm.ipynb
```

## Pilot data shape

Two subjects (`pil02`, `pil03`), 2 sessions × 10 runs × 20 trials each ≈ 400
trials/subject after dropping no-response trials. Each subject has both
gain and loss blocks. Decision RTs (computed as
`feedback.rt − response_phase_start.onset`) are ~0.5–0.8 s.

```
P(choice=True) by (subject, domain):       Mean RT (s):
subject    gain   loss                     subject    gain   loss
pil02      0.40   0.62                     pil02      0.70   0.78
pil03      0.46   0.49                     pil03      0.46   0.42
```

**Hierarchical by default.** `fit_ddm.py` fits **one hierarchical model**
(partial pooling across subjects) — the realistic choice, and what you'll
use for the full sample. At N=2 the group-level posterior is naturally wide
(only two subjects inform it), but that's honest; partial pooling is still
the right structure and tightens automatically as subjects come in. (You
*can* fit each subject alone with `--per-subject`, but that's rarely what you
want for real data.) Convergence of the hierarchical DDM is handled by
bauer's starting-point finder (on by default — see below); it's what makes
this converge where a naive fit would have been a seed lottery.

## Convergence (read this before trusting the DDM fits)

The DDM is the fussiest model here. Two things keep it well-behaved:

1. **Drop fast RTs** (`--rt-floor 0.20`, already on). HSSM's WFPT
   likelihood has a zero-gradient floor when `t0 > min(rt)`, which freezes
   NUTS. This is the single most common cause of a stuck DDM.
2. **Don't regress `t0` on domain.** `t0` is motor delay; regressing it
   under-identifies a weak-signal subject. **pil03** is near-chance in
   losses (P≈0.49 → little drift information), and with `t0 ~ domain` it
   would not converge (r̂≈2.7); dropping that one regressor is necessary.
   The fit script does this by default; pass `--t0-domain` only when you
   have enough data to identify it.
3. **Sample with `--chain-method parallel`** (set in `run_ddm.sh`).
   numpyro's default `vectorized` shares one RNG seed across chains, so a
   bad seed freezes several at once — on these weak-signal fits that gives
   r̂≈2.5 *even with `t0` dropped*. `parallel` gives each chain an
   independent seed and is what gets pil02 and pil03 to r̂≈1.00.

Always check the per-subject `max r̂ ≤ 1.01` line the notebook prints
before interpreting a subject. The general "won't converge" playbook is
`notes/fitting_ddm_models.md`.

## Adapting to the full N

When the real dataset is ready, just point `load_pilot.py` (or whatever
loader you build for the full dataset) at the new directory. The fit
scripts and notebooks are data-agnostic — change `--data` on the CLI,
re-run, and every plot below the data-loading step works.

## What gets modelled

### `PsychophysicalLapseRegressionModel` (simple)

$$
P(\text{chose option 2}) \;=\; (1-\lambda)\,\Phi\!\left(\frac{x_2 - x_1 + \beta}{\sqrt{2}\,\nu}\right) + \tfrac{1}{2}\lambda
$$

with $x_k = \log(n_k \cdot p_k) = \log(\text{EU}_k)$, $\nu$ the sensitivity,
$\beta$ the risk premium, $\lambda$ the lapse rate. Fast, transparent
parameters.

### `RiskLapseRegressionModel` (Bayesian)

Bayesian observer over $\log(n_k)$ with the probability built into the
decision rule, separate `n1_evidence_sd` and `n2_evidence_sd` per option
(captures working-memory order effects), and a lapse rate. Risk attitudes
emerge from the prior shrinkage × encoding noise interaction (the KLW
mechanism).

### `DDMRiskRegressionModel`

Same cognitive front-end as the Bayesian psychometric, plus an HSSM
Wiener-WFPT likelihood for joint choice + RT. The drift is

$$
v \;=\; v_\text{scale} \cdot \frac{(\hat\mu_2 - \hat\mu_1) + \log(p_2/p_1)}{\sqrt{\eta_1^2 + \eta_2^2}}
$$

i.e. the SNR of the perceived log-EU difference. Probability enters as an
**additive log-odds shift** (the same threshold the static probit uses) —
not an EV-style multiplicative weighting.

> **Naming convention used in the plots and write-ups**: $\eta$ for
> encoding **noise** (`nu` / `n_evidence_sd` in bauer's source); $\sigma$
> reserved for the **prior SD** in the KLW framework. The notebooks
> label the plot axes accordingly.
