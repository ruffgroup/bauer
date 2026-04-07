#!/bin/bash
#SBATCH --job-name=fit_numer
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/fit_numerosity_gpu_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Fit numerosity model on GPU via numpyro (JAX), grid=101.
# Submit as array job:
#   sbatch --array=1-39 --export=MODEL=log_encoding fit_numerosity_gpu.sh
#   sbatch --array=1-39 --export=MODEL=flexible_condition fit_numerosity_gpu.sh

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

SUBJECTS=(1 2 3 4 5 6 7 8 9 10 12 13 14 15 16 17 18 19 20 21 22 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41)
SUBJECT=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}

echo "=== Fit numerosity (GPU): sub-${SUBJECT}, model=${MODEL}, grid=${GRID} ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

$PYTHON -u -c "import jax; print('JAX devices:', jax.devices())"

$PYTHON -u "$REPO/bauer/fitting/fit_numerosity.py" "$SUBJECT" \
    --model "$MODEL" \
    --grid-resolution "$GRID" \
    --draws "$DRAWS" \
    --tune "$TUNE" \
    --chains "$CHAINS" \
    --nuts-sampler numpyro \
    --output-dir /shares/zne.uzh/gdehol/ds-neuralpriors/derivatives/bauer

echo "=== Done: $(date) ==="
