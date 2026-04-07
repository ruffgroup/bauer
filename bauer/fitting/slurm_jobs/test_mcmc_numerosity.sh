#!/bin/bash
#SBATCH --job-name=test_numer
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/test_mcmc_numerosity_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Quick test: fit LogEncodingEstimationModel to subject 1 with 100 draws.
# Should complete in ~30 min with grid_resolution=21.

export PYTHONUNBUFFERED=1
PYTHON=$HOME/.conda/envs/bauer/bin/python
REPO=$HOME/git/bauer

echo "=== Test MCMC: numerosity ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Python: $PYTHON"

$PYTHON -u "$REPO/bauer/fitting/fit_numerosity.py" 1 \
    --model log_encoding \
    --grid-resolution 21 \
    --draws 100 \
    --tune 100 \
    --chains 2 \
    --output-dir "$REPO/results/numerosity"

echo "=== Done: $(date) ==="
