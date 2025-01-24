import torch
num1 = float(input("Enter 1st Number : "))
num2 = float(input("Enter 2nd Number : "))
num3 = float(input("Enter 3rd Number : "))
x = torch.tensor(num1, requires_grad=True)
y = torch.tensor(num2, requires_grad=True)
z = torch.tensor(num3, requires_grad=True)
a = 2 * x
b = torch.sin(y)
c = a / b if b != 0 else torch.tensor(0.0, requires_grad=True)
d = c * z
e = torch.log(1 + d) if (1 + d) > 0 else torch.tensor(0.0, requires_grad=True)
f = torch.tanh(e)
print(f"Intermediate values: a = {a.item()}, b = {b.item()}, c = {c.item()}, d = {d.item()}, e = {e.item()}")
f.backward()
print(f"Gradient of f with respect to y (PyTorch): {y.grad.item()}")
f_value = torch.tanh(torch.log(1 + (2 * x / torch.sin(y)) * z))
grad_f = (1 - f_value.item()**2) * (1 / (1 + (2 * x * z / torch.sin(y)))) * z * (-2 * x / torch.sin(y)**2) * torch.cos(y)
print(f"Manual gradient of f with respect to y: {grad_f}")