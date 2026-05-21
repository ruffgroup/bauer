#!/bin/bash
# Resubmit only the flex variants with tune=2000 instead of the default 1000.
# Why: flex models have ~10x more spline coefficients than non-flex; the
# 1000-step warmup left mass-matrix adaptation incomplete (max r̂ = 1.06,
# min ESS = 83 on Garcia choice-flex, with prior_mu blowing up to 15.7).
#
# Usage:
#     bash bauer/scripts/slurm_jobs/submit_flex_warmup2k.sh

set -e
SLURM=$HOME/git/bauer/bauer/scripts/slurm_jobs
ROOT=/shares/zne.uzh/gdehol/bauer_results

submit_l4() {
  local NAME="$1"; shift
  local TIME="$1"; shift
  sbatch --job-name="$NAME" --gres=gpu:L4:1 --time="$TIME" \
         "$SLURM/run_fit.sh" bauer_cuda "$@"
}

W='--tune 2000 --draws 1000'

# Garcia 64
GARCIA="$ROOT/garcia"
submit_l4 garcia_choice_flex     01:00:00 bauer.scripts.fit_garcia choice --flex       --n-subjects all --backend numpyro $W --out-dir "$GARCIA"
submit_l4 garcia_ddm_flex        02:30:00 bauer.scripts.fit_garcia ddm --flex          --n-subjects all --v-scale fixed --backend numpyro $W --out-dir "$GARCIA"
submit_l4 garcia_rdm_flex        02:00:00 bauer.scripts.fit_garcia rdm --flex          --n-subjects all --backend numpyro $W --out-dir "$GARCIA"

# Garcia spline-order sweep on RDM-flex: {3, 7, 9, 11, 13}
GARCIA_SWEEP="$ROOT/garcia_spline_sweep"
for SO in 3 7 9 11 13; do
  submit_l4 "garcia_rdm_flex_so${SO}" 02:00:00 \
    bauer.scripts.fit_garcia rdm --flex --spline-order $SO \
    --n-subjects all --backend numpyro $W --out-dir "$GARCIA_SWEEP/so${SO}"
done

# Dehollander dotcloud 30
DH_DC="$ROOT/dehollander_dotcloud"
submit_l4 dh_dc_choice_flex      01:00:00 bauer.scripts.fit_dehollander2024 choice --flex     --task dotcloud --n-subjects all --prior-estimate full --backend numpyro $W --out-dir "$DH_DC"
submit_l4 dh_dc_ddm_flex         02:30:00 bauer.scripts.fit_dehollander2024 ddm --flex        --task dotcloud --n-subjects all --prior-estimate full --v-scale free --backend numpyro $W --out-dir "$DH_DC"
submit_l4 dh_dc_rdm_flex         02:00:00 bauer.scripts.fit_dehollander2024 rdm --flex        --task dotcloud --n-subjects all --prior-estimate full --backend numpyro $W --out-dir "$DH_DC"

# Dehollander symbolic 58
DH_SYM="$ROOT/dehollander_symbolic"
submit_l4 dh_sym_choice_flex     01:00:00 bauer.scripts.fit_dehollander2024 choice --flex     --task symbolic --n-subjects all --prior-estimate full --backend numpyro $W --out-dir "$DH_SYM"
submit_l4 dh_sym_ddm_flex        03:00:00 bauer.scripts.fit_dehollander2024 ddm --flex        --task symbolic --n-subjects all --prior-estimate full --v-scale free --backend numpyro $W --out-dir "$DH_SYM"
submit_l4 dh_sym_rdm_flex        02:30:00 bauer.scripts.fit_dehollander2024 rdm --flex        --task symbolic --n-subjects all --prior-estimate full --backend numpyro $W --out-dir "$DH_SYM"

# TMS 35
TMS="$ROOT/tms"
submit_l4 tms_choice_flex        01:00:00 bauer.scripts.fit_dehollander_tms choice --flex     --prior-estimate full --backend numpyro $W --out-dir "$TMS"
submit_l4 tms_ddm_flex           02:30:00 bauer.scripts.fit_dehollander_tms ddm --flex        --prior-estimate full --v-scale free --backend numpyro $W --out-dir "$TMS"
submit_l4 tms_rdm_flex           02:00:00 bauer.scripts.fit_dehollander_tms rdm --flex        --prior-estimate full --backend numpyro $W --out-dir "$TMS"
submit_l4 tms_choice_flex_reg    01:30:00 bauer.scripts.fit_dehollander_tms choice --flex --regression --reg-on n1_evidence_sd,n2_evidence_sd --prior-estimate full --backend numpyro $W --out-dir "$TMS"
submit_l4 tms_ddm_flex_reg       03:00:00 bauer.scripts.fit_dehollander_tms ddm --flex --regression --reg-on n1_evidence_sd,n2_evidence_sd --prior-estimate full --v-scale free --backend numpyro $W --out-dir "$TMS"
submit_l4 tms_rdm_flex_reg       02:30:00 bauer.scripts.fit_dehollander_tms rdm --flex --regression --reg-on n1_evidence_sd,n2_evidence_sd --prior-estimate full --backend numpyro $W --out-dir "$TMS"

echo "submitted flex jobs with tune=2000."
