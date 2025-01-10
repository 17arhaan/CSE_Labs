import torch
x = torch.tensor([2.0])
x.requires_grad_(True) #indicate we will need the gradients with respect to this variable
y = x**2 + 5
print(y)
y.backward()
print('PyTorch gradient:', x.grad)
with torch.no_grad():
    dy_dx = 2 * x
print('Analytical gradient:',dy_dx)