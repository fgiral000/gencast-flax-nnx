#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fa.giral@alumnos.upm.es
#SBATCH --export=ALL

set -e -o pipefail  # avoid -u because Anaconda activate scripts use unset vars

module --force purge
module load apps/2021
module load Anaconda3

# init conda for non-interactive shell
eval "$($(which conda) shell.bash hook)"
conda activate graphcast_env

# sanity checks (no importlib.util)
echo ">>> Using Python: $(which python)"
python -V
python - <<'PY'
try:
    import sys
    import jax, jaxlib
    print("PY:", sys.executable)
    print("jax OK:", jax.__version__, "jaxlib:", jaxlib.__version__)
except Exception as e:
    print("jax import failed:", repr(e))
PY

# headless plotting + XLA memory
export MPLBACKEND=Agg

# run with the exact interpreter from the env
srun -u "$(which python)" -m training.download_era5_earthkit \
    --out-dir data_era5 \
    --start-year 2002 \
    --end-year 2003 \
    --resolution 2.5 \
    --season JJA \