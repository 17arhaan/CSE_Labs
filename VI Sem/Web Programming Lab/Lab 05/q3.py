import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import os

class CheckpointHandler:
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
    
    def save_best_model(self, state, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if state['loss'] < self.best_valid_loss:
            self.best_valid_loss = state['loss']
            torch.save(state, filename)
            print('Checkpoint file updated')

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.classify_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 20, bias=True),
            nn.Linear(20, 10, bias=True)
        )
    
    def forward(self, x):
        return self.classify_head(self.net(x))

model = CNN().to('cuda')
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
checkpoint_handler = CheckpointHandler()

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train = datasets.MNIST('.', train=True, download=True, transform=transforms)
test = datasets.MNIST('.', download=True, train=False, transform=transforms)
train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=64)

for epoch in range(10):
    model.train()
    running_loss = 0
    for input, target in train_loader:
        input, target = input.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch - {epoch}, loss = {running_loss}')
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss
    }
    checkpoint_handler.save_best_model(state, './q3_checkpoints/checkpoint_best.pth')

model = CNN()
checkpoint = torch.load('./q3_checkpoints/checkpoint_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
criterion = nn.CrossEntropyLoss()

model.to('cuda')
for epoch in range(10):
    model.train()
    running_loss = 0
    for input, target in train_loader:
        input, target = input.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch - {epoch}, loss = {running_loss}')
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss
    }
    checkpoint_handler.save_best_model(state, './q3_checkpoints/checkpoint_best.pth')

all_preds, all_target = [], []
model.eval()
with torch.no_grad():
    for input, target in test_loader:
        input, target = input.to('cuda'), target.to('cuda')
        output = model(input)
        _, index = torch.max(output, dim=1)
        all_preds.extend(index.to('cpu').numpy())
        all_target.extend(target.to('cpu').numpy())

print(f'Accuracy: {accuracy_score(all_preds, all_target) * 100:.2f}%')
