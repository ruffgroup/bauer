#!/bin/bash
#SBATCH --job-name=alina_ddm
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null

# Fit DDMRiskRegressionModel on the simulated gain/loss data.
# GPU + numpyro backend. Submit with:
#   sbatch ~/git/bauer/examples/for_alina/slurm/run_ddm.sh
# Logs: ~/logs/alina_ddm_<jobid>.txt

set -e
ENV_NAME="bauer_cuda"
JOBNAME="${SLURM_JOB_NAME:-alina_ddm}"
LOGFILE="$HOME/logs/${JOBNAME}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

T_START=$(date +%s)
echo "================================================================"
echo "  job: $JOBNAME (id=$SLURM_JOB_ID) on $(hostname)"
echo "  start: $(date)"
echo "================================================================"

if [ -n "${SLURM_JOB_GPUS:-}" ] || [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  nvidia-smi 2>/dev/null | head -12 || true
else
  export JAX_PLATFORMS=cpu
fi

export PYTHONUNBUFFERED=1
PY="$HOME/data/conda/envs/${ENV_NAME}/bin/python"
REPO="$HOME/git/bauer"

cd "$REPO"
$PY -u "$REPO/examples/for_alina/fit_ddm.py" \
    --data    "$REPO/examples/for_alina/data/pilot_data.tsv" \
    --out     "$REPO/examples/for_alina/results/ddm_idata.nc" \
    --draws   2000 \
    --tune    4000 \
    --chains  4 \
    --target-accept 0.99 \
    --backend numpyro \
    --chain-method parallel    # independent RNG per chain — vectorized gets caught
                               # by bad seeds on these weak-signal per-subject fits
rc=$?

T_END=$(date +%s)
ELAPSED=$((T_END - T_START))
H=$((ELAPSED / 3600)); M=$(((ELAPSED % 3600) / 60)); S=$((ELAPSED % 60))
echo "================================================================"
printf "  end: %s | exit=%d | elapsed=%dh %dm %ds\n" "$(date)" "$rc" "$H" "$M" "$S"
echo "================================================================"
exit $rc
