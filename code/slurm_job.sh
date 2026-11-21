#!/usr/bin/env bash
# SLURM template (edit partition/account/time). Each phase is a separate short job;
# re-submitting the same job resumes where it stopped.
#   sbatch slurm_job.sh configs/default.yaml sft
#SBATCH --job-name=pii-rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --requeue

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# source ~/venvs/pii/bin/activate      # <- your environment
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TOKENIZERS_PARALLELISM=false
bash run_all.sh "${1:-configs/default.yaml}" "${2:-all}"
