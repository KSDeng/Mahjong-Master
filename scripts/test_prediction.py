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

transform = transforms.Compose([
    transforms.ToTensor()
])

if __name__ == "__main__":
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
        # shutil.rmtree("./dataset_preprocessed")       # TODO: check if file exists, if exists then delete
        with zipfile.ZipFile("./dataset_preprocessed.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

    shutil.make_archive("./dataset_preprocessed", "zip", '.', './dataset_preprocessed')
    dataset_train, dataset_test = load_data()

    from random import randint

    num_images = 4
    random.seed(12)
    fig, axes = plt.subplots(1, num_images)
    for i in range(num_images):
        value = randint(0, len(dataset_test))
        axes[i].imshow(dataset_test[value][0].permute(1, 2, 0))
        axes[i].set_title(f"Testing example #{i + 1} ")
    plt.show()

    
