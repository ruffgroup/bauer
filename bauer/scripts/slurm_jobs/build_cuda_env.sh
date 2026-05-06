#!/bin/bash
#SBATCH --job-name=bauer_cuda_build
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null

# Builds the bauer_cuda conda env on a GPU node (so jax CUDA install can
# verify drivers at install time). Logs to a custom file so we can tail it.
#
# Submit with:
#     sbatch ~/git/bauer/bauer/scripts/slurm_jobs/build_cuda_env.sh

set -e
LOGFILE="$HOME/logs/bauer_cuda_build_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "================================================================"
echo "build started at $(date)  on $(hostname)"
echo "================================================================"

module load gpu
module load cuda/12.2.1
nvidia-smi || echo "(nvidia-smi unavailable)"

. $HOME/init_conda.sh

YML=$HOME/git/bauer/environment_cuda.yml
if conda env list | awk '{print $1}' | grep -qx 'bauer_cuda'; then
  echo "[update] bauer_cuda env exists — updating from $YML"
  conda env update -n bauer_cuda -f "$YML" --prune
else
  echo "[create] bauer_cuda env not found — creating from $YML"
  conda env create -f "$YML"
fi

echo "================================================================"
echo "verifying GPU JAX..."
$HOME/data/conda/envs/bauer_cuda/bin/python -c "
import jax, jax.numpy as jnp
print('jax version:', jax.__version__)
print('default backend:', jax.default_backend())
print('devices:', jax.devices())
x = jnp.arange(1_000_000, dtype=jnp.float32)
print('1M dot:', float(jnp.dot(x, x)))
print('OK')
"

echo "================================================================"
echo "build finished at $(date)"
