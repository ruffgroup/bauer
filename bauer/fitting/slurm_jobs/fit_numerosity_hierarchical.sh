#!/bin/bash
#SBATCH --job-name=fit_num_hier
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/fit_numerosity_hier_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Fit hierarchical numerosity model across all subjects via MCMC.
# Single job (not array) — all subjects in one model.
#
# Usage:
#   sbatch --export=MODEL=log_encoding fit_numerosity_hierarchical.sh

export PYTHONUNBUFFERED=1
PYTHON=$HOME/data/conda/envs/bauer/bin/python
REPO=$HOME/git/bauer
MODEL="${MODEL:-log_encoding}"
GRID="${GRID:-21}"
DRAWS="${DRAWS:-1000}"
TUNE="${TUNE:-1000}"
CHAINS="${CHAINS:-4}"

echo "=== Hierarchical fit: numerosity, model=${MODEL} ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Grid: ${GRID}, Draws: ${DRAWS}, Tune: ${TUNE}, Chains: ${CHAINS}"

$PYTHON -u "$REPO/bauer/fitting/fit_numerosity.py" \
    --hierarchical \
    --model "$MODEL" \
    --grid-resolution "$GRID" \
    --draws "$DRAWS" \
    --tune "$TUNE" \
    --chains "$CHAINS" \
    --output-dir /shares/zne.uzh/gdehol/ds-neuralpriors/derivatives/bauer

echo "=== Done: $(date) ==="
