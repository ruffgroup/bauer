#!/bin/bash
#SBATCH --job-name=alina_psy_simple
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null

# Fit the simple PsychophysicalLapseRegressionModel (x = log(EV)) on the
# simulated gain/loss data. Fast — usually 5-10 min on a GPU.
#
# Submit with:
#   sbatch ~/git/bauer/examples/for_alina/slurm/run_psychometric_simple.sh
# Logs: ~/logs/alina_psy_simple_<jobid>.txt

set -e
ENV_NAME="bauer_cuda"
JOBNAME="${SLURM_JOB_NAME:-alina_psy_simple}"
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
$PY -u "$REPO/examples/for_alina/fit_psychometric_simple.py" \
    --data    "$REPO/examples/for_alina/data/pilot_data.tsv" \
    --out     "$REPO/examples/for_alina/results/psychometric_simple_idata.nc" \
    --draws   1500 \
    --tune    2500 \
    --chains  4 \
    --target-accept 0.95 \
    --backend numpyro
rc=$?

T_END=$(date +%s)
ELAPSED=$((T_END - T_START))
H=$((ELAPSED / 3600)); M=$(((ELAPSED % 3600) / 60)); S=$((ELAPSED % 60))
echo "================================================================"
printf "  end: %s | exit=%d | elapsed=%dh %dm %ds\n" "$(date)" "$rc" "$H" "$M" "$S"
echo "================================================================"
exit $rc
