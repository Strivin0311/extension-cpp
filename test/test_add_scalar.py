import torch

import extension_cpp

device = "cuda"
# dtype = torch.float32
# dtype = torch.bfloat16
dtype = torch.float16

a = torch.randn(2, 3, dtype=dtype, device=device)
print(f"{a=}\n")

b = extension_cpp.ops.add_scalar(a, 1.0)
print(f"{b=}\n")
