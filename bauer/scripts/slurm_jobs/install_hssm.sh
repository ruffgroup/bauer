#!/bin/bash
#SBATCH --job-name=bauer_install_hssm
#SBATCH --account=zne.uzh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/dev/null

# Adds hssm + jax + numpyro to the existing bauer (CPU) env. Quick fix
# for clusters where bauer was installed without the optional DDM extras.
set -e
LOGFILE="$HOME/logs/bauer_install_hssm_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "================================================================"
echo "install started at $(date)  on $(hostname)"
echo "================================================================"
PIP="$HOME/data/conda/envs/bauer/bin/pip"
$PIP install --quiet hssm "jax" numpyro blackjax
$HOME/data/conda/envs/bauer/bin/python -c "import hssm, jax, numpyro, blackjax; print('hssm', hssm.__version__); print('jax', jax.__version__); print('numpyro', numpyro.__version__); print('blackjax', blackjax.__version__); print('OK')"
echo "================================================================"
echo "install finished at $(date)"
