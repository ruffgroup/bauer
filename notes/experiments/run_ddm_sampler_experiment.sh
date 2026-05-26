#!/bin/bash
#SBATCH --job-name=ddm_sampler_exp
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:L4:1
#SBATCH --partition=lowprio
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#
# Controlled DDM sampler experiment (basic vs regression × backend ×
# chain_method × tune × seed) on the pil03 testbed. Writes a TSV.
#   sbatch notes/experiments/run_ddm_sampler_experiment.sh
# Logs: ~/logs/ddm_sampler_exp_<jobid>.txt

set -eo pipefail
JOBNAME="${SLURM_JOB_NAME:-ddm_sampler_exp}"
LOGFILE="$HOME/logs/${JOBNAME}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "=== $JOBNAME (id=$SLURM_JOB_ID) on $(hostname) | start $(date) ==="
nvidia-smi 2>/dev/null | head -12 || true

export PYTHONUNBUFFERED=1
PY="$HOME/data/conda/envs/bauer_cuda/bin/python"
REPO="$HOME/git/bauer"
cd "$REPO"
$PY -u notes/experiments/run_ddm_sampler_experiment.py \
    --out notes/experiments/ddm_sampler_results.tsv

echo "=== end $(date) ==="
