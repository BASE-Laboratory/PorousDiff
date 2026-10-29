#!/bin/bash
#SBATCH --job-name=binary-v2-test
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/projects/u6uc/asif/logs/binary-v2-test-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/binary-v2-test-%j.err

set -euo pipefail

cd /projects/u6uc/asif/code/PorousDiff
source /projects/u6uc/asif/.venv/bin/activate

python -u - <<'PY'
import torch
from generative.networks.nets import DiffusionModelUNet

assert torch.cuda.is_available(), "CUDA unavailable"

model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 128),
    attention_levels=(False, False, True),
    num_res_blocks=1,
).cuda()

x = torch.randn(1, 1, 64, 64, 64, device="cuda")
t = torch.tensor([500], device="cuda")

output = model(x, t)
loss = output.square().mean()
loss.backward()

print("GPU:", torch.cuda.get_device_name(0))
print("Output shape:", tuple(output.shape))
print("Loss:", loss.item())
print("Forward/backward smoke test PASSED")
PY
