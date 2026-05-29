#!/bin/bash
#SBATCH --job-name=pil03_seedsweep
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:L4:1
#SBATCH --partition=lowprio
#SBATCH --time=01:00:00
#SBATCH --array=0-15%8
#SBATCH --output=/dev/null
#
# Multi-seed convergence-RATE sweep on pil03 (regression DDM): 8 seeds each of
# vectorized vs parallel at tune=2000. Answers whether the vec-vs-parallel
# difference is a real rate difference or just seed luck.
#   sbatch notes/experiments/run_pil03_seedsweep_array.sh
# Per-cell TSVs land in notes/experiments/seed_cells/.

set -eo pipefail
# index -> "model backend chain_method tune seed"
CONFIGS=(
  "regression numpyro vectorized 2000 0" "regression numpyro vectorized 2000 1"
  "regression numpyro vectorized 2000 2" "regression numpyro vectorized 2000 3"
  "regression numpyro vectorized 2000 4" "regression numpyro vectorized 2000 5"
  "regression numpyro vectorized 2000 6" "regression numpyro vectorized 2000 7"
  "regression numpyro parallel 2000 0"   "regression numpyro parallel 2000 1"
  "regression numpyro parallel 2000 2"   "regression numpyro parallel 2000 3"
  "regression numpyro parallel 2000 4"   "regression numpyro parallel 2000 5"
  "regression numpyro parallel 2000 6"   "regression numpyro parallel 2000 7"
)
read -r MODEL BACKEND CM TUNE SEED <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

LOGFILE="$HOME/logs/pil03_seedsweep_${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"; exec >"$LOGFILE" 2>&1
echo "=== task $SLURM_ARRAY_TASK_ID: $MODEL $BACKEND $CM t$TUNE s$SEED on $(hostname) ==="

export PYTHONUNBUFFERED=1
cd "$HOME/git/bauer"
"$HOME/data/conda/envs/bauer_cuda/bin/python" -u \
    notes/experiments/run_ddm_cell_pil03.py \
    --model "$MODEL" --backend "$BACKEND" --chain-method "$CM" \
    --tune "$TUNE" --draws 1000 --seed "$SEED"
