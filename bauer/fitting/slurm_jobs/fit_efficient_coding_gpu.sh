#!/bin/bash
#SBATCH --job-name=fit_effic
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/fit_efficient_gpu_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Fit efficient coding model on GPU via numpyro (JAX), grid=101.
# Submit as array job:
#   sbatch --array=1-13 --export=MODEL=perception fit_efficient_coding_gpu.sh
#   sbatch --array=1-13 --export=MODEL=sequential fit_efficient_coding_gpu.sh

export PYTHONUNBUFFERED=1
module load gpu 2>/dev/null
module load cuda/12.2.1 2>/dev/null

PYTHON=$HOME/data/conda/envs/bauer/bin/python
REPO=$HOME/git/bauer
MODEL="${MODEL:-perception}"
GRID="${GRID:-101}"
DRAWS="${DRAWS:-1000}"
TUNE="${TUNE:-1000}"
CHAINS="${CHAINS:-4}"
BIDS_FOLDER="${BIDS_FOLDER:-/shares/zne.uzh/gdehol/ds-abstract_values_pilot}"

SUBJECTS=(2 3 4 5 7 9 10 11 12 13 14 15 16)
SUBJECT=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}

echo "=== Fit efficient coding (GPU): sub-${SUBJECT}, model=${MODEL}, grid=${GRID} ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

$PYTHON -u -c "import jax; print('JAX devices:', jax.devices())"

$PYTHON -u "$REPO/bauer/fitting/fit_efficient_coding.py" "$SUBJECT" \
    --model "$MODEL" \
    --grid-resolution "$GRID" \
    --draws "$DRAWS" \
    --tune "$TUNE" \
    --chains "$CHAINS" \
    --nuts-sampler numpyro \
    --bids-folder "$BIDS_FOLDER" \
    --output-dir /shares/zne.uzh/gdehol/ds-abstract_values_pilot/derivatives/bauer

echo "=== Done: $(date) ==="
