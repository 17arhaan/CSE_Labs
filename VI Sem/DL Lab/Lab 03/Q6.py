import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

X = torch.tensor([
    [3.0, 8.0],
    [4.0, 5.0],
    [5.0, 7.0],
    [6.0, 3.0],
    [2.0, 1.0]
], dtype=torch.float32)

y = torch.tensor([-3.7, 3.5, 2.5, 11.5, 5.7], dtype=torch.float32).view(-1, 1)

model = nn.Linear(2, 1)

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs = 1000
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

plt.plot(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.show()

x_input = torch.tensor([[3.0, 2.0]], dtype=torch.float32)
y_pred = model(x_input)
print(f"Predicted Y for X1=3 and X2=2: {y_pred.item():.4f}")
