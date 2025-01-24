import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

x = torch.tensor([1, 5, 10, 10, 25, 50, 70, 75, 100], dtype=torch.float32).view(-1, 1)
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32).view(-1, 1)

model = nn.Sequential(
    nn.Linear(1, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1000
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()

    y_pred = model(x)

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

with torch.no_grad():
    predicted = model(x)

predicted_class = (predicted > 0.5).float()
print(f"Predictions: {predicted_class.view(-1).numpy()}")
