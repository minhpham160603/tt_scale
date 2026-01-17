#!/bin/bash
#SBATCH -A deep_learning
#SBATCH -t 20:00:00
#SBATCH -J sweep_vllm_hybrid
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Note: GPU request syntax varies by cluster.
# If your cluster supports GPU types via Slurm, keep this:
#SBATCH --gpus=5060ti:1
# Otherwise you may need one of these instead (ask admin / check docs):
# #SBATCH --gpus=1
# #SBATCH --gres=gpu:1

set -euo pipefail

cd /home/sichma/tt_scale

# If you want to force venv usage, uncomment:
# if [[ -f .env/bin/activate ]]; then
#   source .env/bin/activate
# fi

bash scripts/sweep_vllm_hybrid.sh
