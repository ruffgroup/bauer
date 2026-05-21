#!/bin/bash
# Benchmarks 3 backends on Garcia DDM at TWO scales (N=8 and N=64). 6 jobs
# total. GPU jobs pinned to L4 for reproducible timings — the cluster's L4
# nodes have 24 GB which is plenty for these models.
#
#   N=8:                          N=64:
#     pymc CPU                      pymc CPU (~3-4 hr expected)
#     numpyro CPU                   numpyro CPU
#     numpyro GPU (L4)              numpyro GPU (L4)
#
# Timings auto-appended to ~/logs/bauer_runtimes.tsv.

set -e
SLURM=$HOME/git/bauer/bauer/scripts/slurm_jobs
OUT=/shares/zne.uzh/gdehol/bauer_results/jax_experiment

mkdir -p "$OUT"

# ---------- N=8 (cheap, runs in minutes) ----------
sbatch --job-name=jax_exp_pymc_cpu_n8 --time=00:45:00 \
       "$SLURM/run_fit.sh" bauer bauer.scripts.fit_garcia ddm \
       --n-subjects 8 --backend pymc --v-scale free \
       --no-ppc --out-dir "$OUT/pymc_cpu_n8"

sbatch --job-name=jax_exp_numpyro_cpu_n8 --time=00:30:00 \
       "$SLURM/run_fit.sh" bauer bauer.scripts.fit_garcia ddm \
       --n-subjects 8 --backend numpyro --v-scale free \
       --no-ppc --out-dir "$OUT/numpyro_cpu_n8"

sbatch --job-name=jax_exp_numpyro_gpu_n8 --gres=gpu:L4:1 --time=00:20:00 \
       "$SLURM/run_fit.sh" bauer_cuda bauer.scripts.fit_garcia ddm \
       --n-subjects 8 --backend numpyro --v-scale free \
       --no-ppc --out-dir "$OUT/numpyro_gpu_n8"

# ---------- N=64 (production scale) ----------
sbatch --job-name=jax_exp_pymc_cpu_n64 --time=06:00:00 \
       "$SLURM/run_fit.sh" bauer bauer.scripts.fit_garcia ddm \
       --n-subjects all --backend pymc --v-scale free \
       --no-ppc --out-dir "$OUT/pymc_cpu_n64"

sbatch --job-name=jax_exp_numpyro_cpu_n64 --time=04:00:00 \
       "$SLURM/run_fit.sh" bauer bauer.scripts.fit_garcia ddm \
       --n-subjects all --backend numpyro --v-scale free \
       --no-ppc --out-dir "$OUT/numpyro_cpu_n64"

sbatch --job-name=jax_exp_numpyro_gpu_n64 --gres=gpu:L4:1 --time=01:30:00 \
       "$SLURM/run_fit.sh" bauer_cuda bauer.scripts.fit_garcia ddm \
       --n-subjects all --backend numpyro --v-scale free \
       --no-ppc --out-dir "$OUT/numpyro_gpu_n64"

echo "submitted 6 jobs; squeue -u \$USER  to watch."
