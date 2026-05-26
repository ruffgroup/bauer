#!/bin/bash
#SBATCH --job-name=pil03_finder
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:L4:1
#SBATCH --partition=lowprio
#SBATCH --time=01:30:00
#SBATCH --array=0-15%8
#SBATCH --output=/dev/null
#
# Validate the shipped starting-point finder: pil03 regression DDM, the FAILING
# config (numpyro vectorized, tune=2000), 8 seeds with the finder (--init core)
# vs 8 seeds without it (--init default). Convergence RATE comparison.
#   sbatch notes/experiments/run_pil03_finder_array.sh

set -eo pipefail
IDX=$SLURM_ARRAY_TASK_ID
if [ "$IDX" -lt 8 ]; then INIT=core; SEED=$IDX; else INIT=default; SEED=$((IDX-8)); fi

LOGFILE="$HOME/logs/pil03_finder_${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"; exec >"$LOGFILE" 2>&1
echo "=== task $IDX: init=$INIT seed=$SEED on $(hostname) ==="

export PYTHONUNBUFFERED=1
cd "$HOME/git/bauer"
"$HOME/data/conda/envs/bauer_cuda/bin/python" -u \
    notes/experiments/run_ddm_cell_pil03.py \
    --model regression --backend numpyro --chain-method vectorized \
    --tune 2000 --draws 1000 --seed "$SEED" --init "$INIT"
