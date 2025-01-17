import torch

x = torch.tensor([2.0, 4.0])
y = torch.tensor([20.0, 40.0])

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
learning_rate = 0.001

for epoch in range(2):
    y_pred = w * x + b
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()

    print(f"Epoch {epoch + 1}:")
    print(f"w.grad: {w.grad.item()}, b.grad: {b.grad.item()}")

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    w.grad.zero_()
    b.grad.zero_()

    print(f"Updated w: {w.item()}, Updated b: {b.item()}\n")
