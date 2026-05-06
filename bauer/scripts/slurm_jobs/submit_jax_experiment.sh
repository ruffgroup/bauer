#!/bin/bash
# Submits 3 small fits to compare backends on Garcia 8-subj DDM:
#   - pymc-NUTS (CPU)
#   - numpyro NUTS (CPU JAX, bauer env)
#   - numpyro NUTS (GPU JAX, bauer_cuda env, --gres=gpu:1)
#
# Inspect timings via:  grep -E "Sampling|saved|exit" ~/logs/jax_exp_*

set -e
SLURM=$HOME/git/bauer/bauer/scripts/slurm_jobs
OUT=/shares/zne.uzh/gdehol/bauer_results/jax_experiment

mkdir -p "$OUT"

# 1) pymc on CPU (baseline)
sbatch --job-name=jax_exp_pymc_cpu --time=00:45:00 \
       "$SLURM/run_fit.sh" bauer bauer.scripts.fit_garcia ddm \
       --n-subjects 8 --backend pymc --v-scale free \
       --no-ppc --out-dir "$OUT/pymc_cpu"

# 2) numpyro on CPU (JAX, no GPU)
sbatch --job-name=jax_exp_numpyro_cpu --time=00:30:00 \
       "$SLURM/run_fit.sh" bauer bauer.scripts.fit_garcia ddm \
       --n-subjects 8 --backend numpyro --v-scale free \
       --no-ppc --out-dir "$OUT/numpyro_cpu"

# 3) numpyro on GPU (JAX with CUDA, bauer_cuda env)
sbatch --job-name=jax_exp_numpyro_gpu --gres=gpu:1 --time=00:20:00 \
       "$SLURM/run_fit.sh" bauer_cuda bauer.scripts.fit_garcia ddm \
       --n-subjects 8 --backend numpyro --v-scale free \
       --no-ppc --out-dir "$OUT/numpyro_gpu"

echo "submitted; squeue -u \$USER  to watch."
