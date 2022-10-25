import zipfile
import shutil
import torch
from torchvision.models import resnet50
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# load model weights
path = 'weight-20221012124841'
model = resnet50()
model.load_state_dict(torch.load(path))

image_size = (64, 64)
path = '/tmp/dataset_cs5242'
shutil.rmtree(path, ignore_errors=True)
with zipfile.ZipFile("./dataset_preprocessed.zip", 'r') as zip_ref:
    zip_ref.extractall(path)

transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = torchvision.datasets.ImageFolder(root=path + "/dataset_preprocessed", transform=transform)

print(dataset.classes)  # classes names
print(dataset.class_to_idx)  # index of classes

validation_set_ratio = 0.2
batch_size = 32

n = len(dataset)  # total number of examples
n_test = int(validation_set_ratio * n)
subsets = torch.utils.data.random_split(dataset, [n - n_test, n_test], generator=torch.Generator().manual_seed(42))
train_set = subsets[0]
test_set = subsets[1]
print(train_set.__len__(), test_set.__len__())  # [train_set, validation_set]
print(type(train_set.dataset))

train_data_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
test_data_loader = DataLoader(test_set, batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataiter = iter(test_data_loader)
images, labels = dataiter.next()
correct = 0
total = 0
with torch.no_grad():
    for data in test_data_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')