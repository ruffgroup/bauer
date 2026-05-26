#!/bin/bash
#SBATCH --job-name=pil03_init
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:L4:1
#SBATCH --partition=lowprio
#SBATCH --time=01:30:00
#SBATCH --array=0-11%6
#SBATCH --output=/dev/null
#
# Init-scheme experiment on pil03 (regression DDM), all on the FAILING baseline
# config (numpyro vectorized, tune=2000) so any improvement is attributable to
# init, not chain_method. map_jitter (MAP center, dispersed) and prior_scaled
# (center + frac*prior_sd jitter) × 6 seeds each. Compare convergence RATE to
# the default-init vectorized cells from the seedsweep array.
#   sbatch notes/experiments/run_pil03_init_array.sh

set -eo pipefail
INITS=(map_jitter prior_scaled)
IDX=$SLURM_ARRAY_TASK_ID
INIT=${INITS[$((IDX / 6))]}
SEED=$((IDX % 6))

LOGFILE="$HOME/logs/pil03_init_${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"; exec >"$LOGFILE" 2>&1
echo "=== task $IDX: init=$INIT seed=$SEED on $(hostname) ==="

export PYTHONUNBUFFERED=1
cd "$HOME/git/bauer"
"$HOME/data/conda/envs/bauer_cuda/bin/python" -u \
    notes/experiments/run_ddm_cell_pil03.py \
    --model regression --backend numpyro --chain-method vectorized \
    --tune 2000 --draws 1000 --seed "$SEED" --init "$INIT"
