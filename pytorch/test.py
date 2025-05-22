import torch
import my_kernel


x = torch.arange(10.0, device="cuda")
out = torch.empty_like(x)

my_kernel.square(x, out)

print("Input:", x)
print("Output:", out)

