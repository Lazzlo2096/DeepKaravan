"""Verify that ROCm can execute work on the AMD GPU."""

import torch

if not torch.version.hip or not torch.cuda.is_available():
    raise RuntimeError("A ROCm build of PyTorch and an accessible AMD GPU are required")
print(f"PyTorch {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}", flush=True)
x = torch.ones(32, 32, device="cuda")
assert (x @ x).sum().item() == 32768
print("PASS AMD GPU matrix multiplication", flush=True)
