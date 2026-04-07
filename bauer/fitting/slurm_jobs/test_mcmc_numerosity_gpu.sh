#!/bin/bash
#SBATCH --job-name=test_num_gpu
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/test_mcmc_numerosity_gpu_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Test MCMC on GPU with numpyro (JAX) backend, grid=101.

export PYTHONUNBUFFERED=1
module load gpu
module load cuda/12.2.1

PYTHON=$HOME/data/conda/envs/bauer/bin/python
REPO=$HOME/git/bauer

echo "=== Test MCMC: numerosity (GPU, grid=101) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# Verify GPU visible to JAX
$PYTHON -u -c "import jax; print('JAX devices:', jax.devices())"

# Run MCMC with numpyro sampler (uses JAX, runs on GPU)
$PYTHON -u "$REPO/bauer/fitting/fit_numerosity.py" 1 \
    --model log_encoding \
    --grid-resolution 101 \
    --draws 200 \
    --tune 200 \
    --chains 2 \
    --nuts-sampler numpyro \
    --output-dir /shares/zne.uzh/gdehol/ds-neuralpriors/derivatives/bauer

echo "=== Done: $(date) ==="
