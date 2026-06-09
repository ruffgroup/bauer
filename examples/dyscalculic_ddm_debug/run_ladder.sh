#!/usr/bin/env bash
# Submit the incremental DDM convergence ladder on the dyscalculic magnitude
# data to GPU compute nodes (NEVER the login node). One sbatch per ladder step.
#
# Resource spec (per the task): --partition=standard --qos=normal
#   --gres=gpu:1 --mem=48G, env binary ~/data/conda/envs/bauer_cuda/bin/python.
#
# Usage:
#   bash run_ladder.sh                 # 8-subject ladder (steps 1-4)
#   bash run_ladder.sh all <winner>    # scale winner to all 66 subjects,
#                                      #   e.g.: bash run_ladder.sh all indep
#
# After jobs land, scan results:
#   for f in ~/ddm_debug/*.json; do echo "$f"; cat "$f"; echo; done
#   for f in ~/ddm_debug/*.log;  do echo "== $f =="; grep -A6 RESULT "$f"; done
set -euo pipefail

PY=${PY:-$HOME/data/conda/envs/bauer_cuda/bin/python}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DRIVER="$HERE/fit_real_ddm.py"
DATA=${DATA:-$HOME/git/dyscalculic_ddm/data/magjudge_behavior_DNumRisk.csv}
OUTDIR=${OUTDIR:-$HOME/ddm_debug}
mkdir -p "$OUTDIR"

SBATCH_COMMON=(--partition=standard --qos=normal --account=zne.uzh
               --gres=gpu:1 --mem=48G --cpus-per-task=4 --time=04:00:00)

submit() {  # name  extra-driver-args...
    local name=$1; shift
    sbatch "${SBATCH_COMMON[@]}" \
        --job-name="ddm_${name}" \
        --output="$OUTDIR/${name}.log" \
        --wrap="$PY $DRIVER --data $DATA --out $OUTDIR/${name}.json \
                --nc-out $OUTDIR/${name}.nc $*"
}

if [[ "${1:-}" == "all" ]]; then
    WINNER=${2:-indep}
    echo "Scaling step '$WINNER' to ALL subjects..."
    case "$WINNER" in
        plain)    submit all_plain    --step plain    --n-subjects 0 ;;
        plain_pf) submit all_plain_pf --step plain_pf --n-subjects 0 ;;
        indep)    submit all_indep_pf --step indep --init pathfinder --n-subjects 0 ;;
        lapse)    submit all_lapse    --step lapse --init pathfinder \
                      --memory-model independent --beta-mu-mean 0.02 --n-subjects 0 ;;
        *) echo "Unknown winner: $WINNER" >&2; exit 1 ;;
    esac
    exit 0
fi

echo "Submitting 8-subject convergence ladder to $OUTDIR ..."
# 1. plain DDM baseline (mapjitter, shared_perceptual_noise)
submit step1_plain    --step plain    --n-subjects 8
# 2. plain + Pathfinder (the key test)
submit step2_plain_pf --step plain_pf --n-subjects 8
# 3. independent noise (rotate the ridge), Pathfinder init
submit step3_indep_pf --step indep --init pathfinder --n-subjects 8
# 3b. independent noise, mapjitter (control for init effect)
submit step3_indep_mj --step indep --init mapjitter  --n-subjects 8
# 4. lapse on independent noise, Pathfinder, regularized DOWN
submit step4_lapse    --step lapse --init pathfinder \
    --memory-model independent --beta-mu-mean 0.02 --n-subjects 8

echo "Submitted. Poll with: squeue -u \$USER ; sacct -X --format=JobName%20,State,Elapsed"
