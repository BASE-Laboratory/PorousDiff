#!/bin/bash
#SBATCH --job-name=carbon-sample-96
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/u6uc/asif/logs/carbon-sample-96-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/carbon-sample-96-%j.err

set -euo pipefail

cd /projects/u6uc/asif/code/PorousDiff
source /projects/u6uc/asif/.venv/bin/activate

echo "Started: $(date -u)"
echo "Job ID: ${SLURM_JOB_ID}"
nvidia-smi

python -u scripts/evaluation/sample_carbon_conditional_96.py

echo "Finished: $(date -u)"
