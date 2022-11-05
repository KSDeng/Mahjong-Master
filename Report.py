# import required modules
import os
import random
import skimage.io as io
from skimage.transform import resize, rotate
from skimage.util import random_noise, img_as_ubyte
from skimage.draw import rectangle
import zipfile
import shutil
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch_snippets.torch_loader import Report
from torch.optim.lr_scheduler import ExponentialLR
import datetime
import gc
import torch.nn.functional as F

# assign directory
directory_dataset = './dataset/'
directory_train = './dataset/train'
directory_test = './dataset/test'
output_directory_dataset = './dataset_preprocessed'
output_directory_train = './dataset_preprocessed/train'
output_directory_test = './dataset_preprocessed/test'
# This may cause some error in some operation systems.
skip_list = ['.DS_Store']
k = 2

shutil.rmtree(directory_dataset, ignore_errors=True)
shutil.rmtree(output_directory_dataset, ignore_errors=True)

with zipfile.ZipFile("./dataset_raw.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

def load_data():
    image_size = (64, 64)
    batch_size = 32

    path = "."

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset_train = torchvision.datasets.ImageFolder(root = path + "/dataset_preprocessed/train", transform=transform)
    dataset_test = torchvision.datasets.ImageFolder(root = path + "/dataset_preprocessed/test", transform=transform)
    return dataset_train, dataset_test

def data_loader(dataset_train, dataset_test):
    batch_size = 32
    train_data_loader = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=0)
    test_data_loader = DataLoader(dataset_test, batch_size, shuffle=True, num_workers=0)
    return train_data_loader, test_data_loader


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset_raw = torchvision.datasets.ImageFolder(root=directory_train, transform=transform)

    # random sample and split train/test dataset.
    enable_split = True

    random.seed(0)

    # split before data enhancement to avoid data leak
    if enable_split:
        for filename in os.listdir(directory_train):
            f = os.path.join(directory_train, filename)
            f_out = os.path.join(directory_test, filename)
            if not os.path.exists(f_out):
                os.makedirs(f_out)
            if os.path.isdir(f):
                samples = random.sample(os.listdir(f), k)
                for p in samples:
                    from_name = os.path.join(f, p)
                    to_name = os.path.join(f_out, p)
                    shutil.move(from_name, to_name)

    # Debug switch
    is_debug = False
    # we only do preprocessing and augmentation for 1 time because it takes lot of time.
    # If want to reproduce, set `enable_preprocessing = True`.
    # If you only want to see what it generates, set `enable_preprocessing = True` and `is_debug = True`
    # Then samples are generated into "./dataset_preprocessed"
    enable_preprocessing = not os.path.exists("./dataset_preprocessed.zip")
    # output image size
    fix_size = 64


    # save image with noise augmentation
    def noise_save(name, content):
        io.imsave(name + '-1.png', img_as_ubyte(content))
        image_noised_gaussian = random_noise(content, mode='gaussian', mean=0, var=0.01, clip=True)
        io.imsave(name + '-2.png', img_as_ubyte(image_noised_gaussian))
        image_noised_gaussian_2 = random_noise(content, mode='gaussian', mean=0, var=0.02, clip=True)
        io.imsave(name + '-3.png', img_as_ubyte(image_noised_gaussian_2))
        image_noised_s_p = random_noise(content, mode='s&p', salt_vs_pepper=0.5, clip=True)
        io.imsave(name + '-4.png', img_as_ubyte(image_noised_s_p))
        image_noised_s_p_2 = random_noise(content, mode='s&p', salt_vs_pepper=0.2, clip=True)
        io.imsave(name + '-5.png', img_as_ubyte(image_noised_s_p_2))


    # save image with random occlusion
    def occlusion_save(name, content):
        image_occlusion = content.copy()
        size_x = random.randrange(fix_size // 3, fix_size // 2)
        size_y = random.randrange(fix_size // 3, fix_size // 2)
        rr, cc = rectangle((random.randrange(0, fix_size - size_x),
                            random.randrange(0, fix_size - size_y)),
                           extent=(size_x, size_y))
        image_occlusion[rr, cc] = 1
        io.imsave(name + '-1.png', img_as_ubyte(image_occlusion))


    # save image with rotation augmentation
    def rotate_save(name, content):
        for i in range(6, 7):
            occlusion_save(name + '-' + str(i), content)
        noise_save(name + '-1', content)
        content = rotate(content, 90)
        for i in range(7, 8):
            occlusion_save(name + '-' + str(i), content)
        noise_save(name + '-2', content)
        content = rotate(content, 90)
        for i in range(8, 9):
            occlusion_save(name + '-' + str(i), content)
        noise_save(name + '-3', content)
        content = rotate(content, 90)
        for i in range(9, 10):
            occlusion_save(name + '-' + str(i), content)
        noise_save(name + '-4', content)


    map_dirs = [
        [directory_train, output_directory_train],
        [directory_test, output_directory_test],
    ]

    if enable_preprocessing:
        # iterate over files in that directory
        for dir in map_dirs:
            for filename in os.listdir(dir[0]):
                f = os.path.join(dir[0], filename)
                f_out = os.path.join(dir[1], filename)
                if not os.path.exists(f_out):
                    os.makedirs(f_out)
                if os.path.isdir(f):
                    for imgname in os.listdir(f):
                        if (imgname in skip_list):
                            continue
                        img = os.path.join(f, imgname)
                        img_raw = io.imread(img)
                        # Resize all data to fix size
                        image_resized = resize(img_raw, (fix_size, fix_size), anti_aliasing=True)
                        rotate_save(os.path.join(f_out, imgname[0: imgname.find('.')]), image_resized)
                        if is_debug:
                            io.imshow(image_resized)
                            break
                if is_debug:
                    break
    else:
        shutil.rmtree("./dataset_preprocessed", ignore_errors=True)
        with zipfile.ZipFile("./dataset_preprocessed.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

    shutil.make_archive("./dataset_preprocessed", "zip", '.', './dataset_preprocessed')

    dataset_train, dataset_test = load_data()
    train_data_loader, test_data_loader = data_loader(dataset_train, dataset_test)
    # Sample for using the above data loaders
    for i, data in enumerate(train_data_loader, 0):
        # iteration index, torch.Size([32, 64, 64, 3]) torch.Size([32])
        print(i, data[0].permute(0, 2, 3, 1).shape, data[1].shape)
        break



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    loss_function = nn.CrossEntropyLoss()
    def criterion(predictions, targets):
        loss = loss_function(predictions, targets)
        acc = (torch.max(predictions, dim=1)[1] == targets).float().mean()
        return loss, acc


    num_epoch = 5
    lr_decay_gamma = 0.9

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
    def test_batch(model, data, criterion):
        model.eval()
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss, acc = criterion(outputs, labels)
        return loss, acc


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


    model = MLP().to(device)
    log_mlp = Report(num_epoch)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay_gamma, verbose=True)

    for epoch in range(num_epoch):
        N = len(train_data_loader)
        for i, data in enumerate(train_data_loader, 0):
            loss, acc = train_batch(model, data, optimizer, criterion)
            if i % 5 == 0:
                log_mlp.record(pos=epoch + (i + 1) / N, train_loss=loss, train_acc=acc, end='\r')
        print()

        N = len(test_data_loader)
        for i, data in enumerate(test_data_loader, 0):
            loss, acc = test_batch(model, data, criterion)
            if i % 5 == 0:
                log_mlp.record(pos=epoch + (i + 1) / N, test_loss=loss, test_acc=acc, end='\r')
        print()
        scheduler.step()

    print("train_acc:", [v for pos, v in log_mlp.train_acc][-1], "|test_acc:", [v for pos, v in log_mlp.test_acc][-1])


    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
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

    def getTimeStamp():
        now = datetime.datetime.now()
        return now.strftime("%Y%m%d%H%M%S")

    class ResNet(nn.Module):
        def __init__(self, block, layers, num_classes=10):
            super(ResNet, self).__init__()
            self.in_planes = 64
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
            self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
            self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer2 = self._make_layer(block, 256, layers[2], stride=1)
            self.layer3 = self._make_layer(block, 512, layers[3], stride=1)
            self.avg_pool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(2048, num_classes)

        def _make_layer(self, block, planes, blocks, stride=1):
            down_sample = None
            if stride != 1 or self.in_planes != planes:
                down_sample = nn.Sequential(
                    nn.Conv2d(self.in_planes, planes, kernel_size=(1, 1), stride=(stride, stride)),
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

    weight_save_path = 'weight-resnet34-' + getTimeStamp()
    model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=34).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay_gamma, verbose=True)


    log_resnet = Report(num_epoch)
    for epoch in range(num_epoch):
        N = len(train_data_loader)
        for i, data in enumerate(train_data_loader, 0):
            loss, acc = train_batch(model, data, optimizer, criterion)
            log_resnet.record(pos=epoch + (i + 1) / N, train_loss=loss, train_acc=acc, end='\r')
        print()
        torch.save(model.state_dict(), weight_save_path)

        N = len(test_data_loader)
        for i, data in enumerate(test_data_loader, 0):
            loss, acc = test_batch(model, data, criterion)
            log_resnet.record(pos=epoch + (i + 1) / N, test_loss=loss, test_acc=acc, end='\r')
        print()
        scheduler.step()
        gc.collect()

    print("train_acc:", [v for pos, v in log_resnet.train_acc][-1], "|test_acc:",
          [v for pos, v in log_resnet.test_acc][-1])



