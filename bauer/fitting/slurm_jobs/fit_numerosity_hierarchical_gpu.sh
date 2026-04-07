#!/bin/bash
#SBATCH --job-name=fit_num_hier
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/fit_numerosity_hier_gpu_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# Fit hierarchical numerosity model across all subjects on GPU.
# All subjects in one model with group-level priors.
#
# Usage:
#   sbatch --export=MODEL=log_encoding fit_numerosity_hierarchical_gpu.sh

export PYTHONUNBUFFERED=1
module load gpu 2>/dev/null
module load cuda/12.2.1 2>/dev/null

PYTHON=$HOME/data/conda/envs/bauer/bin/python
REPO=$HOME/git/bauer
MODEL="${MODEL:-log_encoding}"
GRID="${GRID:-101}"
DRAWS="${DRAWS:-1000}"
TUNE="${TUNE:-1000}"
CHAINS="${CHAINS:-4}"

echo "=== Hierarchical fit (GPU): numerosity, model=${MODEL}, grid=${GRID} ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

$PYTHON -u -c "import jax; print('JAX devices:', jax.devices())"

$PYTHON -u "$REPO/bauer/fitting/fit_numerosity.py" \
    --hierarchical \
    --model "$MODEL" \
    --grid-resolution "$GRID" \
    --draws "$DRAWS" \
    --tune "$TUNE" \
    --chains "$CHAINS" \
    --nuts-sampler numpyro \
    --output-dir /shares/zne.uzh/gdehol/ds-neuralpriors/derivatives/bauer

echo "=== Done: $(date) ==="
