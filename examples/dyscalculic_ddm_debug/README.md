# Dyscalculic DDM convergence debugging (Task B)

Drivers for getting the bauer DDM (and DDM+lapse) to **converge** on the real
dyscalculic magnitude-comparison data, walking an incremental ladder
(smallest/cheapest first) and reporting r-hat / divergences / ESS at each step.

Run on a **GPU compute node** (never the login node).

## Files

- `fit_real_ddm.py` — one self-contained fit per `--step`. Loads
  `magjudge_behavior_DNumRisk.csv` (`rt>=0.20`, `choice=chose_n2`, indexed by
  `[subject, run, trial_nr]`), builds the model, samples, and writes a JSON of
  diagnostics (`max_rhat_mu`, `divergences`, `min_ess_bulk`, elapsed).
- `run_ladder.sh` — sbatch wrapper; submits the whole ladder to
  `--partition=standard --qos=normal --gres=gpu:1 --mem=48G` with the
  `bauer_cuda` env binary.

## Ladder (`--step`)

| Step | `--step` | What | Init | memory_model |
|---|---|---|---|---|
| 1 | `plain` | Plain DDM, no lapse (baseline) | mapjitter | shared_perceptual_noise |
| 2 | `plain_pf` | Plain DDM + **Pathfinder** (key test) | pathfinder | shared_perceptual_noise |
| 3 | `indep` | Independent n1/n2 noise (rotate ridge) | `--init` | independent |
| 4 | `lapse` | + RT-aware Beta lapse, regularized **down** | `--init` | `--memory-model` |
| 5 | scale | Re-run the winner on all 66 (`--n-subjects 0`) | | |

## Guardrails

- `fit_v_scale=False` always (perfect v_scale↔evidence_sd degeneracy).
- Plain DDM only; never the sv-DDM.
- `target_accept=0.99`, `tune=2000`, dense_mass (via the model's
  `recommended_nuts_kwargs`), numpyro `chain_method='vectorized'`.

## Run

```bash
export PY=~/data/conda/envs/bauer_cuda/bin/python
bash run_ladder.sh                  # 8-subject ladder (steps 1-4)
# inspect:
for f in ~/ddm_debug/*.json; do echo "== $f =="; cat "$f"; echo; done
# scale the winner (e.g. independent-noise) to all 66 subjects:
bash run_ladder.sh all indep
```

`--beta-mu-mean 0.02` (default) keeps the lapse contaminant regularized strongly
downward so it can't soak up real DDM structure.
