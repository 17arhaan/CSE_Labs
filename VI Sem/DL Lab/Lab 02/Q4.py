import torch
import math
x = torch.tensor(1.0, requires_grad=True)

f_value = torch.exp(-x**2 - 2*x - torch.sin(x))

f_value.backward()

computed_gradient = x.grad

def analytical_gradient(x):
    return math.exp(-x**2 - 2*x - math.sin(x)) * (-2*x - 2 - math.cos(x))

analytical_grad = analytical_gradient(1.0)

print(f"PyTorch computed gradient: {computed_gradient.item()}")
print(f"Analytical gradient: {analytical_grad}")
