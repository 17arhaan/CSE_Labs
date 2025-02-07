import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class ReducedCNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(32 * 3 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features.view(x.size(0), -1))

mnist_trainset = datasets.MNIST(root="./data", download=True, train=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_trainset, batch_size=50, shuffle=True)
mnist_testset = datasets.MNIST(root="./data", download=True, train=False, transform=transforms.ToTensor())
test_loader = DataLoader(mnist_testset, batch_size=50, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ReducedCNNClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

initial_params = count_parameters(model)
accuracies = []
param_drops = []

epochs = 6
for epoch in range(epochs):
    running_loss = 0.0
    correct, total = 0, 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100
    accuracies.append(accuracy)

    current_params = count_parameters(model)
    param_drop = ((initial_params - current_params) / initial_params) * 100
    param_drops.append(param_drop)

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.3f}, Accuracy: {accuracy:.2f}%")

plt.figure(figsize=(10, 5))
plt.plot(param_drops, accuracies, marker='o', color='b')
plt.xlabel('Percentage Drop in Parameters (%)')
plt.ylabel('Accuracy (%)')
plt.title('Parameter Drop vs Accuracy')
plt.grid(True)
plt.show()

correct, total = 0, 0
for data in test_loader:
    inputs, labels = data[0].to(device), data[1].to(device)
    outputs = model(inputs)

    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
