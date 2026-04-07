#!/bin/bash
#SBATCH --job-name=test_num_gpu
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/test_mcmc_numerosity_gpu_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Test MCMC on GPU with JAX backend, grid=101.

export PYTHONUNBUFFERED=1
module load gpu
module load cuda/12.2.1

PYTHON=$HOME/data/conda/envs/bauer/bin/python
REPO=$HOME/git/bauer

echo "=== Test MCMC: numerosity (GPU, grid=101) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# Verify GPU
$PYTHON -u -c "import jax; print('JAX devices:', jax.devices())"

# Benchmark gradient speed at grid=101
$PYTHON -u -c "
import numpy as np, pandas as pd, time
from bauer.utils.data import load_neuralpriors
from bauer.numerosity import LogEncodingEstimationModel

df = load_neuralpriors()
sub1 = df.xs(1, level='subject').reset_index()
paradigm = sub1[['n', 'response', 'range']].dropna(subset=['response']).copy()
paradigm['n'] = paradigm['n'].astype(float)
paradigm.index = pd.MultiIndex.from_arrays(
    [np.ones(len(paradigm), dtype=int), range(len(paradigm))],
    names=['subject', 'trial'])

model = LogEncodingEstimationModel(paradigm, grid_resolution=101,
                                    n_min=10, n_max=40, response_bin_width=1.0)
model.build_estimation_model(paradigm, hierarchical=False, flat_prior=True)
dlogp = model.estimation_model.compile_dlogp()
ip = model.estimation_model.initial_point()
dlogp(ip)  # warmup
t0 = time.time()
for _ in range(10):
    dlogp(ip)
ms = (time.time() - t0) / 10 * 1000
print(f'grid=101: {ms:.0f} ms/gradient eval')
"

# Run short MCMC
$PYTHON -u "$REPO/bauer/fitting/fit_numerosity.py" 1 \
    --model log_encoding \
    --grid-resolution 101 \
    --draws 100 \
    --tune 100 \
    --chains 2 \
    --output-dir /shares/zne.uzh/gdehol/ds-neuralpriors/derivatives/bauer

echo "=== Done: $(date) ==="
