#!/bin/bash
#SBATCH --job-name=test-binary-96
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/u6uc/asif/logs/test-binary-96-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/test-binary-96-%j.err

set -euo pipefail

source /projects/u6uc/asif/.venv/bin/activate

python -u - <<'PY'
import torch
from generative.networks.nets import DiffusionModelUNet

assert torch.cuda.is_available(), "GPU unavailable"

device = torch.device("cuda")

model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 128),
    attention_levels=(False, False, True),
    num_res_blocks=1,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

x = torch.randn(1, 1, 96, 96, 96, device=device)
timesteps = torch.randint(0, 1000, (1,), device=device)
target = torch.randn_like(x)

optimizer.zero_grad(set_to_none=True)

prediction = model(x, timesteps=timesteps)
loss = torch.nn.functional.mse_loss(prediction, target)

loss.backward()
optimizer.step()
torch.cuda.synchronize()

print("GPU:", torch.cuda.get_device_name(0))
print("Output shape:", tuple(prediction.shape))
print("Loss:", loss.item())
print("Peak GPU memory (GB):",
      round(torch.cuda.max_memory_allocated() / 1024**3, 2))
print("96³ GPU SMOKE TEST PASSED")
PY
