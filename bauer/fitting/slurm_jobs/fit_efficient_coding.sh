#!/bin/bash
#SBATCH --job-name=fit_effic
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/fit_efficient_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

# Fit efficient coding estimation model to abstract_values pilot data via MCMC.
# Submit as array job over subjects:
#   sbatch --array=1-13 --export=MODEL=perception fit_efficient_coding.sh
#
# Available models: perception, valuation, sequential
# SLURM_ARRAY_TASK_ID maps to subject index (1-based).

export PYTHONUNBUFFERED=1
PYTHON=$HOME/data/conda/envs/bauer/bin/python
REPO=$HOME/git/bauer
MODEL="${MODEL:-perception}"
GRID="${GRID:-31}"
DRAWS="${DRAWS:-1000}"
TUNE="${TUNE:-1000}"
CHAINS="${CHAINS:-4}"
BIDS_FOLDER="${BIDS_FOLDER:-/shares/zne.uzh/gdehol/ds-abstract_values_pilot}"

# Pilot subjects: 2,3,4,5,7,9,10,11,12,13,14,15,16
SUBJECTS=(2 3 4 5 7 9 10 11 12 13 14 15 16)
SUBJECT=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}

echo "=== Fit efficient coding: sub-${SUBJECT}, model=${MODEL} ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Grid: ${GRID}, Draws: ${DRAWS}, Tune: ${TUNE}, Chains: ${CHAINS}"

$PYTHON -u "$REPO/bauer/fitting/fit_efficient_coding.py" "$SUBJECT" \
    --model "$MODEL" \
    --grid-resolution "$GRID" \
    --draws "$DRAWS" \
    --tune "$TUNE" \
    --chains "$CHAINS" \
    --bids-folder "$BIDS_FOLDER" \
    --output-dir /shares/zne.uzh/gdehol/ds-abstract_values_pilot/derivatives/bauer

echo "=== Done: $(date) ==="
