#!/bin/bash
#SBATCH --job-name=bauer_fit
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null

# Generic bauer fit runner. Reads the model module + args from CLI:
#   sbatch run_fit.sh <env> <module> [args...]
# where <env> is 'bauer' (CPU) or 'bauer_cuda' (GPU). For GPU mode also
# pass `--gres=gpu:1` to sbatch on the command line, e.g.:
#
#   sbatch --gres=gpu:1 --job-name=garcia_ddm_jaxgpu run_fit.sh bauer_cuda \
#       bauer.scripts.fit_garcia ddm --n-subjects all --backend numpyro \
#       --no-ppc --out-dir /shares/zne.uzh/gdehol/bauer_results/garcia
#
# Logs land at ~/logs/<jobname>_<jobid>.txt.

set -e
ENV_NAME="${1:?Error: pass conda env name as 1st arg (e.g. bauer or bauer_cuda)}"
shift
MODULE="${1:?Error: pass python module as 2nd arg (e.g. bauer.scripts.fit_garcia)}"
shift

JOBNAME="${SLURM_JOB_NAME:-bauer_fit}"
LOGFILE="$HOME/logs/${JOBNAME}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "================================================================"
echo "  job: $JOBNAME (id=$SLURM_JOB_ID) on $(hostname)"
echo "  start: $(date)"
echo "  env:   $ENV_NAME"
echo "  cmd:   python -u -m $MODULE $*"
echo "================================================================"

# GPU detection: --gres=gpu:1 sets CUDA_VISIBLE_DEVICES. Conda envs ship
# their own CUDA runtime (jax[cuda12], tensorflow-gpu), so no `module load`
# is needed. On CPU-only nodes, pin JAX to CPU to avoid a benign but noisy
# "cuInit failed" warning when the CUDA plugin is bundled in the env.
if [ -n "${SLURM_JOB_GPUS:-}" ] || [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  nvidia-smi 2>/dev/null | head -12 || true
else
  export JAX_PLATFORMS=cpu
fi

export PYTHONUNBUFFERED=1
PY="$HOME/data/conda/envs/$ENV_NAME/bin/python"

cd "$HOME/git/bauer"
$PY -u -m "$MODULE" "$@"
rc=$?

echo "================================================================"
echo "  end: $(date)  exit=$rc"
exit $rc
