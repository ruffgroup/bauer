#!/bin/bash
#SBATCH --job-name=test_effic
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/test_mcmc_efficient_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Quick test: fit EfficientPerceptionModel to subject 5 with 100 draws.

export PYTHONUNBUFFERED=1
PYTHON=$HOME/data/conda/envs/bauer/bin/python
REPO=$HOME/git/bauer

echo "=== Test MCMC: efficient coding ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Python: $PYTHON"

$PYTHON -u "$REPO/bauer/fitting/fit_efficient_coding.py" 5 \
    --model perception \
    --grid-resolution 21 \
    --draws 100 \
    --tune 100 \
    --chains 2 \
    --output-dir /shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/bauer

echo "=== Done: $(date) ==="
