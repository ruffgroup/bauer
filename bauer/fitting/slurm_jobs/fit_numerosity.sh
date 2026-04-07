#!/bin/bash
#SBATCH --job-name=fit_numer
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/fit_numerosity_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

# Fit numerosity estimation model to neural_priors data via MCMC.
# Submit as array job over subjects:
#   sbatch --array=1-39 --export=MODEL=log_encoding fit_numerosity.sh
#
# Available models: log_encoding, flexible_shared, flexible_condition, efficient_encoding
# SLURM_ARRAY_TASK_ID maps to subject index (1-based).

export PYTHONUNBUFFERED=1
PYTHON=$HOME/data/conda/envs/bauer/bin/python
REPO=$HOME/git/bauer
MODEL="${MODEL:-log_encoding}"
GRID="${GRID:-31}"
DRAWS="${DRAWS:-1000}"
TUNE="${TUNE:-1000}"
CHAINS="${CHAINS:-4}"

# Map array task ID to subject ID
# Subjects in bundled data: 1-41 (excluding 11, 23) = 39 subjects
SUBJECTS=(1 2 3 4 5 6 7 8 9 10 12 13 14 15 16 17 18 19 20 21 22 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41)
SUBJECT=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}

echo "=== Fit numerosity: sub-${SUBJECT}, model=${MODEL} ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Grid: ${GRID}, Draws: ${DRAWS}, Tune: ${TUNE}, Chains: ${CHAINS}"

$PYTHON -u "$REPO/bauer/fitting/fit_numerosity.py" "$SUBJECT" \
    --model "$MODEL" \
    --grid-resolution "$GRID" \
    --draws "$DRAWS" \
    --tune "$TUNE" \
    --chains "$CHAINS" \
    --output-dir /shares/zne.uzh/gdehol/ds-neuralpriors/derivatives/bauer

echo "=== Done: $(date) ==="
