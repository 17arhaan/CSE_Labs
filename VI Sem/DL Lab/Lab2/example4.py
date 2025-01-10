import torch
import numpy as np
def grad_sigmoid_manual(x):
    a = -x
    b = np.exp(a)
    c=1+b
    s = 1.0 / c
    dsdc = (-1.0 / (c**2))
    dsdb = dsdc * 1
    dsda = dsdb * np.exp(a)
    dsdx = dsda * (-1)
    return dsdx
def sigmoid(x):
    y = 1.0 / (1.0 + torch.exp(-x))
    return y
input_x = 2.0
x = torch.tensor(input_x).requires_grad_(True)
y = sigmoid(x)
y.backward()
print('autograd:', x.grad.item())
print('manual:', grad_sigmoid_manual(input_x))