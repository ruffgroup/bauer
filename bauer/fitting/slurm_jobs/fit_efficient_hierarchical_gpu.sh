#!/bin/bash
#SBATCH --job-name=fit_eff_hier
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/fit_efficient_hier_gpu_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:3
#SBATCH --time=06:00:00

# Hierarchical efficient coding model on GPU (all 13 subjects).
# Requests 3 GPUs so numpyro can run chains in parallel.
#
# Usage:
#   sbatch --export=MODEL=perception fit_efficient_hierarchical_gpu.sh

export PYTHONUNBUFFERED=1
module load gpu 2>/dev/null
module load cuda/12.2.1 2>/dev/null

PYTHON=$HOME/data/conda/envs/bauer/bin/python
REPO=$HOME/git/bauer
MODEL="${MODEL:-perception}"
GRID="${GRID:-51}"
DRAWS="${DRAWS:-1000}"
TUNE="${TUNE:-1000}"
CHAINS="${CHAINS:-3}"

echo "=== Hierarchical fit (GPU): efficient coding, model=${MODEL}, grid=${GRID} ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

$PYTHON -u -c "import jax; print('JAX devices:', jax.devices()); print('Device count:', jax.device_count())"

$PYTHON -u "$REPO/bauer/fitting/fit_efficient_coding.py" \
    --hierarchical \
    --model "$MODEL" \
    --grid-resolution "$GRID" \
    --draws "$DRAWS" \
    --tune "$TUNE" \
    --chains "$CHAINS" \
    --nuts-sampler numpyro \
    --bids-folder /shares/zne.uzh/gdehol/ds-abstract_values_pilot \
    --output-dir /shares/zne.uzh/gdehol/ds-abstract_values_pilot/derivatives/bauer

echo "=== Done: $(date) ==="
