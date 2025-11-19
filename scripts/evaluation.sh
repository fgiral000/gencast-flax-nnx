#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

nvidia-smi || true

# run with the exact interpreter from the env
srun -u "$(which python)" -m training.evaluation \
 --load_checkpoint /checkpoints/gencast_model/checkpoint_step_28000 \
 --clean_sst_nans \
 --apply_normalization \
 --max_rollout_steps 30 \
 --variables 2m_temperature,mean_sea_level_pressure,10m_u_component_of_wind,10m_v_component_of_wind \
 --domain global \
 --output_dir ./inference_plots_30steps \
 --gif_variable 2m_temperature \
 --gif_out rollout.gif \
 --gif_fps 3 \
 --save_rollout_nc \
 --rollout_nc_out rollout_30steps.nc \
 --netcdf_compression_level 4 \