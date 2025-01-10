import torch
num1 = float(input("Enter 1st Number : "))
num2 = float(input("Enter 2nd Number : "))
num3 = float(input("Enter 3rd Number : "))
w = torch.tensor(num1, requires_grad=True)
x = torch.tensor(num2)
b = torch.tensor(num3, requires_grad=True)
u = w * x
v = u + b
a = torch.relu(v)
a.backward()
if v.item() > 0:
    da_dw_manual = x.item()
else:
    da_dw_manual = 0.0
print(f"Gradient computed by PyTorch: {w.grad.item()}")
print(f"Manual gradient calculation: {da_dw_manual}")

a = torch.sigmoid(v)
a.backward()
if v.item() > 0:
    da_dw_manual = x.item() * a.item() * (1 - a.item())
else:
    da_dw_manual = 0.0
print(f"Gradient computed by PyTorch: {w.grad.item()}")
print(f"Manual gradient calculation: {da_dw_manual}")