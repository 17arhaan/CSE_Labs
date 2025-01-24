import torch

a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
x = 2 * a + 3 * b
y = 5 * a**2 + 3 * b**3
z = 2 * x + 3 * y

z.backward()

print(f"The gradient of z with respect to a is: {a.grad}")
