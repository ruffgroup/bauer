#!/bin/bash
# Submits all production-grade bauer fits on cluster L4 GPUs with numpyro
# backend. Designed to be re-runnable: each sbatch will write to a known
# path; if you want to skip already-fitted ones, remove the `.nc` files you
# don't want refitted before running.
#
# Output layout:
#   /shares/zne.uzh/gdehol/bauer_results/
#     garcia/64subj/{choice,choice_flex,ddm,ddm_flex,rdm,rdm_flex}.nc
#     dehollander_dotcloud/30subj/{choice,ddm,rdm,rdm_flex}.nc
#     dehollander_symbolic/58subj/{choice,ddm,rdm,rdm_flex}.nc
#     tms/35subj/{choice,ddm,rdm,rdm_flex,choice_reg,ddm_reg,rdm_reg}.nc
#
# Tuning: target_accept=0.95 (script default), tune=1000, draws=1000,
# chains=4. JAX vmaps the 4 chains on a single L4 GPU at ~free cost.

set -e
SLURM=$HOME/git/bauer/bauer/scripts/slurm_jobs
ROOT=/shares/zne.uzh/gdehol/bauer_results

submit_l4() {
  local NAME="$1"; shift
  local TIME="$1"; shift
  sbatch --job-name="$NAME" --gres=gpu:L4:1 --time="$TIME" \
         "$SLURM/run_fit.sh" bauer_cuda "$@"
}

# ---------- Garcia 64 ----------
GARCIA="$ROOT/garcia"
submit_l4 garcia_choice       00:30:00 bauer.scripts.fit_garcia choice           --n-subjects all --no-ppc --out-dir "$GARCIA"
submit_l4 garcia_choice_flex  00:45:00 bauer.scripts.fit_garcia choice --flex    --n-subjects all --no-ppc --out-dir "$GARCIA"
submit_l4 garcia_ddm          01:30:00 bauer.scripts.fit_garcia ddm              --n-subjects all --v-scale free  --no-ppc --out-dir "$GARCIA"
submit_l4 garcia_ddm_flex     02:00:00 bauer.scripts.fit_garcia ddm --flex       --n-subjects all --v-scale fixed --no-ppc --out-dir "$GARCIA"
submit_l4 garcia_rdm          01:00:00 bauer.scripts.fit_garcia rdm              --n-subjects all --no-ppc --out-dir "$GARCIA"
submit_l4 garcia_rdm_flex     01:30:00 bauer.scripts.fit_garcia rdm --flex       --n-subjects all --no-ppc --out-dir "$GARCIA"

# ---------- Dehollander dotcloud 30 ----------
DH_DC="$ROOT/dehollander_dotcloud"
submit_l4 dh_dc_choice        00:30:00 bauer.scripts.fit_dehollander2024 choice  --task dotcloud --n-subjects all --prior-estimate full --no-ppc --out-dir "$DH_DC"
submit_l4 dh_dc_ddm           01:30:00 bauer.scripts.fit_dehollander2024 ddm     --task dotcloud --n-subjects all --prior-estimate full --v-scale free --no-ppc --out-dir "$DH_DC"
submit_l4 dh_dc_rdm           01:00:00 bauer.scripts.fit_dehollander2024 rdm     --task dotcloud --n-subjects all --prior-estimate full --no-ppc --out-dir "$DH_DC"
submit_l4 dh_dc_rdm_flex      01:30:00 bauer.scripts.fit_dehollander2024 rdm --flex --task dotcloud --n-subjects all --prior-estimate full --no-ppc --out-dir "$DH_DC"

# ---------- Dehollander symbolic 58 ----------
DH_SYM="$ROOT/dehollander_symbolic"
submit_l4 dh_sym_choice       00:30:00 bauer.scripts.fit_dehollander2024 choice  --task symbolic --n-subjects all --prior-estimate full --no-ppc --out-dir "$DH_SYM"
submit_l4 dh_sym_ddm           02:00:00 bauer.scripts.fit_dehollander2024 ddm    --task symbolic --n-subjects all --prior-estimate full --v-scale free --no-ppc --out-dir "$DH_SYM"
submit_l4 dh_sym_rdm           01:30:00 bauer.scripts.fit_dehollander2024 rdm    --task symbolic --n-subjects all --prior-estimate full --no-ppc --out-dir "$DH_SYM"
submit_l4 dh_sym_rdm_flex      02:00:00 bauer.scripts.fit_dehollander2024 rdm --flex --task symbolic --n-subjects all --prior-estimate full --no-ppc --out-dir "$DH_SYM"

# ---------- TMS 35 ----------
TMS="$ROOT/tms"
submit_l4 tms_choice          00:30:00 bauer.scripts.fit_dehollander_tms choice  --prior-estimate full --no-ppc --out-dir "$TMS"
submit_l4 tms_ddm              01:30:00 bauer.scripts.fit_dehollander_tms ddm    --prior-estimate full --v-scale free --no-ppc --out-dir "$TMS"
submit_l4 tms_rdm              01:00:00 bauer.scripts.fit_dehollander_tms rdm    --prior-estimate full --no-ppc --out-dir "$TMS"
submit_l4 tms_rdm_flex         01:30:00 bauer.scripts.fit_dehollander_tms rdm --flex --prior-estimate full --no-ppc --out-dir "$TMS"
submit_l4 tms_choice_reg       00:45:00 bauer.scripts.fit_dehollander_tms choice --regression --prior-estimate full --no-ppc --out-dir "$TMS"
submit_l4 tms_ddm_reg          02:00:00 bauer.scripts.fit_dehollander_tms ddm    --regression --prior-estimate full --v-scale free --no-ppc --out-dir "$TMS"
submit_l4 tms_rdm_reg          01:30:00 bauer.scripts.fit_dehollander_tms rdm    --regression --prior-estimate full --no-ppc --out-dir "$TMS"

echo "submitted; squeue -u \$USER  to watch."
