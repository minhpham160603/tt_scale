#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --job-name=apertus-lora
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=raw_results/hybrid_sampling_%x-%j.out
#SBATCH --no-requeue

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

set -x

# 1. Setup Directories
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"
mkdir -p raw_results

# 2. Configure Caches (Crucial for offline/scratch usage)
export HF_HOME="/iopsstor/scratch/cscs/$USER/.cache/huggingface"
export TRITON_CACHE_DIR="/iopsstor/scratch/cscs/$USER/.cache/triton"
mkdir -p "$HF_HOME"
mkdir -p "$TRITON_CACHE_DIR"

# 3. Optimization Flags
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"

export PYTHONUNBUFFERED=1

# 4. Run Execution
srun --environment="${PROJECT_DIR}/tt_scale.toml" bash -c "
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
    cd ${PROJECT_DIR}

    pip install -r requirements.txt

    pip install -e . --no-deps

    echo '----------------------------------'
    echo 'Environment Check:'
    which python
    python3 -c 'import vllm; print(f\"vLLM: {vllm.__version__}\")'
    python3 -c 'import torch; print(f\"Torch: {torch.__version__}, CUDA: {torch.version.cuda}\")'
    echo '----------------------------------'

    # Run the Parallel Baseline
    python3 -m tt_scale.vllm_parallel_hybrid_engine
"

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
