#!/bin/bash
#SBATCH --job-name=op001-patches
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6uc/asif/logs/patches-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/patches-%j.err

set -euo pipefail

cd /projects/u6uc/asif/code/PorousDiff

source /projects/u6uc/asif/.venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(which python)"

python scripts/preprocessing/create_binary_patches.py

echo "Patch extraction completed."
