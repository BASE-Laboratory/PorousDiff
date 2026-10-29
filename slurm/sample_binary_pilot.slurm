#!/bin/bash
#SBATCH --job-name=binary-sample
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/projects/u6uc/asif/logs/binary-sample-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/binary-sample-%j.err

set -euo pipefail

cd /projects/u6uc/asif/code/PorousDiff
source /projects/u6uc/asif/.venv/bin/activate

echo "Starting binary DDPM sampling"
echo "Node: $(hostname)"

python -u scripts/evaluation/sample_binary_diffusion.py

echo "Sampling completed"
