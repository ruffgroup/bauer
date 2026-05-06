#!/bin/bash
# All production-grade bauer fits on cluster L4 GPUs (numpyro backend).
# Re-runnable: each sbatch writes to a deterministic path; no skip logic
# (so you can clean target dirs first if you want a fresh run).
#
# Layout (under /shares/zne.uzh/gdehol/bauer_results/):
#   garcia/64subj/{choice,choice_flex,ddm_freescale,ddm_flex_fixedscale,rdm,rdm_flex}.nc
#   garcia/64subj_spline_sweep/rdm_flex_so{3,5,7,9,11,13}.nc
#   dehollander_dotcloud/30subj/{choice,choice_flex,ddm,ddm_flex,rdm,rdm_flex}_full.nc
#   dehollander_symbolic/58subj/{choice,choice_flex,ddm,ddm_flex,rdm,rdm_flex}_full.nc
#   tms/35subj/{choice,choice_flex,ddm,ddm_flex,rdm,rdm_flex}_full.nc
#                  + {choice,choice_flex,ddm_flex,rdm_flex}_reg_full.nc
#
# Sampler: numpyro NUTS via JAX, target_accept=0.95, tune=1000, draws=1000,
# chains=4. Single L4 GPU per fit.

set -e
SLURM=$HOME/git/bauer/bauer/scripts/slurm_jobs
ROOT=/shares/zne.uzh/gdehol/bauer_results

submit_l4() {
  local NAME="$1"; shift
  local TIME="$1"; shift
  # Default to lowprio partition (more L4 nodes, much shorter wait); jobs can
  # be pre-empted by standard-partition users but our 5-30 min fits rarely
  # get caught. Override with PARTITION env var.
  sbatch --partition="${PARTITION:-lowprio}" \
         --job-name="$NAME" --gres=gpu:L4:1 --time="$TIME" \
         "$SLURM/run_fit.sh" bauer_cuda "$@"
}

# ============================================================
# Garcia 64 — non-symbolic dot-comparison
# ============================================================
GARCIA="$ROOT/garcia"
submit_l4 garcia_choice          00:30:00 bauer.scripts.fit_garcia choice              --n-subjects all --backend numpyro --out-dir "$GARCIA"
submit_l4 garcia_choice_flex     00:45:00 bauer.scripts.fit_garcia choice --flex       --n-subjects all --backend numpyro --out-dir "$GARCIA"
submit_l4 garcia_ddm             01:30:00 bauer.scripts.fit_garcia ddm                 --n-subjects all --v-scale free  --backend numpyro --out-dir "$GARCIA"
submit_l4 garcia_ddm_flex        02:00:00 bauer.scripts.fit_garcia ddm --flex          --n-subjects all --v-scale fixed --backend numpyro --out-dir "$GARCIA"
submit_l4 garcia_rdm             01:00:00 bauer.scripts.fit_garcia rdm                 --n-subjects all --backend numpyro --out-dir "$GARCIA"
submit_l4 garcia_rdm_flex        01:30:00 bauer.scripts.fit_garcia rdm --flex          --n-subjects all --backend numpyro --out-dir "$GARCIA"

# Garcia spline-order sweep on RDM-flex: {3, 7, 9, 11, 13}; spline_order=5 already
# covered above. Outputs go into garcia/64subj_spline_sweep/ for clarity.
GARCIA_SWEEP="$ROOT/garcia_spline_sweep"
for SO in 3 7 9 11 13; do
  submit_l4 "garcia_rdm_flex_so${SO}" 01:30:00 \
    bauer.scripts.fit_garcia rdm --flex --spline-order $SO \
    --n-subjects all --backend numpyro --out-dir "$GARCIA_SWEEP/so${SO}"
done

# ============================================================
# Dehollander 2024 — dotcloud (N=30)
# ============================================================
DH_DC="$ROOT/dehollander_dotcloud"
submit_l4 dh_dc_choice           00:30:00 bauer.scripts.fit_dehollander2024 choice            --task dotcloud --n-subjects all --prior-estimate full --backend numpyro --out-dir "$DH_DC"
submit_l4 dh_dc_choice_flex      00:45:00 bauer.scripts.fit_dehollander2024 choice --flex     --task dotcloud --n-subjects all --prior-estimate full --backend numpyro --out-dir "$DH_DC"
submit_l4 dh_dc_ddm              01:30:00 bauer.scripts.fit_dehollander2024 ddm               --task dotcloud --n-subjects all --prior-estimate full --v-scale free --backend numpyro --out-dir "$DH_DC"
submit_l4 dh_dc_ddm_flex         02:00:00 bauer.scripts.fit_dehollander2024 ddm --flex        --task dotcloud --n-subjects all --prior-estimate full --v-scale free --backend numpyro --out-dir "$DH_DC"
submit_l4 dh_dc_rdm              01:00:00 bauer.scripts.fit_dehollander2024 rdm               --task dotcloud --n-subjects all --prior-estimate full --backend numpyro --out-dir "$DH_DC"
submit_l4 dh_dc_rdm_flex         01:30:00 bauer.scripts.fit_dehollander2024 rdm --flex        --task dotcloud --n-subjects all --prior-estimate full --backend numpyro --out-dir "$DH_DC"

# ============================================================
# Dehollander 2024 — symbolic (N=58)
# ============================================================
DH_SYM="$ROOT/dehollander_symbolic"
submit_l4 dh_sym_choice          00:30:00 bauer.scripts.fit_dehollander2024 choice            --task symbolic --n-subjects all --prior-estimate full --backend numpyro --out-dir "$DH_SYM"
submit_l4 dh_sym_choice_flex     00:45:00 bauer.scripts.fit_dehollander2024 choice --flex     --task symbolic --n-subjects all --prior-estimate full --backend numpyro --out-dir "$DH_SYM"
submit_l4 dh_sym_ddm             02:00:00 bauer.scripts.fit_dehollander2024 ddm               --task symbolic --n-subjects all --prior-estimate full --v-scale free --backend numpyro --out-dir "$DH_SYM"
submit_l4 dh_sym_ddm_flex        02:30:00 bauer.scripts.fit_dehollander2024 ddm --flex        --task symbolic --n-subjects all --prior-estimate full --v-scale free --backend numpyro --out-dir "$DH_SYM"
submit_l4 dh_sym_rdm             01:30:00 bauer.scripts.fit_dehollander2024 rdm               --task symbolic --n-subjects all --prior-estimate full --backend numpyro --out-dir "$DH_SYM"
submit_l4 dh_sym_rdm_flex        02:00:00 bauer.scripts.fit_dehollander2024 rdm --flex        --task symbolic --n-subjects all --prior-estimate full --backend numpyro --out-dir "$DH_SYM"

# ============================================================
# TMS 35 — sessions 2/3 only (default tms_only=True)
# ============================================================
TMS="$ROOT/tms"
# Baseline (no regression)
submit_l4 tms_choice             00:30:00 bauer.scripts.fit_dehollander_tms choice            --prior-estimate full --backend numpyro --out-dir "$TMS"
submit_l4 tms_choice_flex        00:45:00 bauer.scripts.fit_dehollander_tms choice --flex     --prior-estimate full --backend numpyro --out-dir "$TMS"
submit_l4 tms_ddm                01:30:00 bauer.scripts.fit_dehollander_tms ddm               --prior-estimate full --v-scale free --backend numpyro --out-dir "$TMS"
submit_l4 tms_ddm_flex           02:00:00 bauer.scripts.fit_dehollander_tms ddm --flex        --prior-estimate full --v-scale free --backend numpyro --out-dir "$TMS"
submit_l4 tms_rdm                01:00:00 bauer.scripts.fit_dehollander_tms rdm               --prior-estimate full --backend numpyro --out-dir "$TMS"
submit_l4 tms_rdm_flex           01:30:00 bauer.scripts.fit_dehollander_tms rdm --flex        --prior-estimate full --backend numpyro --out-dir "$TMS"

# Regression on stimulation_condition (noise only: n1/n2_evidence_sd)
submit_l4 tms_choice_reg         00:45:00 bauer.scripts.fit_dehollander_tms choice --regression --reg-on n1_evidence_sd,n2_evidence_sd --prior-estimate full --backend numpyro --out-dir "$TMS"
submit_l4 tms_choice_flex_reg    01:00:00 bauer.scripts.fit_dehollander_tms choice --flex --regression --reg-on n1_evidence_sd,n2_evidence_sd --prior-estimate full --backend numpyro --out-dir "$TMS"
submit_l4 tms_ddm_flex_reg       02:30:00 bauer.scripts.fit_dehollander_tms ddm --flex --regression --reg-on n1_evidence_sd,n2_evidence_sd --prior-estimate full --v-scale free --backend numpyro --out-dir "$TMS"
submit_l4 tms_rdm_flex_reg       02:00:00 bauer.scripts.fit_dehollander_tms rdm --flex --regression --reg-on n1_evidence_sd,n2_evidence_sd --prior-estimate full --backend numpyro --out-dir "$TMS"

echo "submitted; squeue -u \$USER  to watch."
