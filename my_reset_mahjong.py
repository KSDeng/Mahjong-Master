import zipfile
import shutil
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

if __name__ == "__main__":
    image_size = (64, 64)
    weight_save_path = 'weight-resnet50-' + getTimeStamp()

    path = '/tmp/dataset_cs5242'

    shutil.rmtree(path, ignore_errors=True)
    with zipfile.ZipFile("./dataset_preprocessed.zip", 'r') as zip_ref:
        zip_ref.extractall(path)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.ImageFolder(root = path + "/dataset_preprocessed", transform=transform)

    print(dataset.classes)  # classes names
    print(dataset.class_to_idx) # index of classes

    validation_set_ratio = 0.2
    batch_size = 32

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
    num_epoch = 20
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = resnet50()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.8, verbose=True)

    for epoch in range(num_epoch):
        running_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print('epoch {:3d} | {:5d} batches loss: {:.7f}'.format(epoch, i + 1, running_loss / 128))
            running_loss = 0.0
            if (i + 1) % 128 == 0:
                torch.save(model.state_dict(), weight_save_path)
        scheduler.step()


    print('Finished Training')

