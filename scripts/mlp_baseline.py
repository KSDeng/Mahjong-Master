import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

class MLP(nn.Module):
  # Define Multilayer Perceptron
   def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 768)
        self.fc2 = nn.Linear(768, 84)
        self.fc3 = nn.Linear(84, 34)

   def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
  
if __name__ == '__main__':
  
  torch.manual_seed(34)
  mlp = MLP()
  
  # Define loss function and optimizer
  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  a = 0
  b = 0
  # Run 20 epoches
  for epoch in range(0, 20): 

    print(f'Starting epoch {epoch+1}')
    current_loss = 0.0

    for i, data in enumerate(train_data_loader, 0):
      inputs, targets = data[0].to(device), data[1].to(device)
      optimizer.zero_grad()
      outputs = mlp(inputs)
      _,train_label = torch.max(outputs.data, 1)
      a += targets.size(0)
      b += (train_label == targets).sum().item()
      loss = loss_function(outputs, targets)
      loss.backward()
      optimizer.step()
      current_loss += loss.item()
      if i % 470 == 469:
          print('Loss after mini-batch %5d: %.3f' %
                (i + 1, current_loss / 470))
          current_loss = 0.0

  print(f'MLP training process done.')
  
  # Run test data set
  all_labels = 0
  good_labels = 0
  for data in test_data_loader:
      inputs, targets = data[0].to(device), data[1].to(device)
      optimizer.zero_grad()
      outputs = mlp(inputs)
      a, pred_labels = torch.max(outputs.data, 1)
      all_labels += targets.size(0)
      good_labels += (pred_labels == targets).sum().item()

  accuracy = (100 * good_labels) // all_labels

  print(f'MLP test accuracy: {accuracy} %')