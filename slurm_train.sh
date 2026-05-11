#!/bin/bash
#SBATCH --job-name=chess_eval
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --partition=batch          # Partition to submit to
#SBATCH --gpus=1                   # GPUs per node
#SBATCH --cpus-per-task=24          # CPU cores allocated per task (for multiprocessing MCTS)
#SBATCH --mem=64G                  # Memory allocation
#SBATCH --time=4-00:00:00

# Change to the job submission directory
cd $SLURM_SUBMIT_DIR

# Ensure uv is available in PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Redirect UV cache to the global scratch area
export UV_CACHE_DIR="$GLOBALSCRATCH/.cache/uv"

# Set threading/environment variables early so children inherit them
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Disable jemalloc background threads to avoid spawn issues
export MALLOC_CONF="background_thread:false"

# (Optional) increase process limits if permitted by cluster admin
# ulimit -u 4096

# Load CUDA environment modules (specify version if required)
module purge
module load CUDA

# Synchronize project environment using uv
uv sync

echo "=========================================================="
echo " JOB START   : $(date)"
echo " Nodes       : $SLURM_JOB_NODELIST"
echo "=========================================================="

echo "Launching evaluation..."
uv run main.py evaluate_mcts_params

echo "=========================================================="
echo " JOB FINISHED: $(date)"
echo "=========================================================="
