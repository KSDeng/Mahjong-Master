import zipfile
import shutil

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import datetime

def getTimeStamp():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d%H%M%S")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, down_sample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(stride, stride), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.down_sample = down_sample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.down_sample:
            residual = self.down_sample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7,7), stride=(2,2), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pool = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=1)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, block, planes, blocks, stride = 1):
        down_sample = None
        if stride != 1 or self.in_planes != planes:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=(1,1), stride=(stride,stride)),
                nn.BatchNorm2d(planes)
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, down_sample))
        self.in_planes = planes
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    image_size = (64, 64)
    weight_save_path = 'weight-resnet50-' + getTimeStamp()
    path = '/tmp/dataset_cs5242'
    validation_set_ratio = 0.2
    batch_size = 32
    num_epoch = 20
    lr = 0.001
    lr_decay_gamma = 0.8

    shutil.rmtree(path, ignore_errors=True)
    with zipfile.ZipFile("./dataset_preprocessed.zip", 'r') as zip_ref:
        zip_ref.extractall(path)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.ImageFolder(root = path + "/dataset_preprocessed", transform=transform)
    print(dataset.classes)  # classes names
    print(dataset.class_to_idx) # index of classes

    n = len(dataset)  # total number of examples
    n_test = int(validation_set_ratio * n)
    subsets = torch.utils.data.random_split(dataset, [n - n_test, n_test], generator=torch.Generator().manual_seed(42))
    train_set = subsets[0]
    test_set = subsets[1]
    print(train_set.__len__(), test_set.__len__()) # [train_set, validation_set]
    print(type(train_set.dataset))

    train_data_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
    test_data_loader = DataLoader(test_set, batch_size, shuffle=True, num_workers=0)

    # training model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ResNet(ResidualBlock, [3,4,6,3], num_classes=34).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay_gamma, verbose=True)

    from torch_snippets.torch_loader import Report
    log = Report(num_epoch)

    cross_entropy = nn.CrossEntropyLoss()
    def resnet_criterion(predictions, targets):
        loss = cross_entropy(predictions, targets)
        acc = (torch.max(predictions, dim=1)[1] == targets).float().mean()
        return loss, acc

    def train_batch(model, data, optimizer, criterion):
        model.train()
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss, acc = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss, acc

    @torch.no_grad()
    def validate_batch(model, data, criterion):
        model.eval()
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss, acc = criterion(outputs, labels)
        return loss, acc

    for epoch in range(num_epoch):
        N = len(train_data_loader)
        for i, data in enumerate(train_data_loader, 0):
            loss, acc = train_batch(model, data, optimizer, resnet_criterion)
            log.record(pos=epoch+(i+1)/N, train_loss = loss, train_acc = acc, end = '\r')
        print()

        N = len(test_data_loader)
        for i, data in enumerate(test_data_loader, 0):
            loss, acc = validate_batch(model, data, resnet_criterion)
            log.record(pos=epoch+(i+1)/N, test_loss = loss, test_acc = acc, end = '\r')
        print()
        scheduler.step()

    log.plot(['train_acc', 'test_acc'])
    plt.show()
    log.plot(['train_loss', 'test_loss'])
    plt.show()

