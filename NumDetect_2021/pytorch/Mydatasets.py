import torch.nn.functional as F
import torch
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import random
import os

def default_loader(path):
    img = Image.open(path)
    image_transforms = transforms.Compose([
        transforms.Grayscale(1)
    ])
    img = image_transforms(img)
    # out = img.resize((28, 28), Image.ANTIALIAS)
    return img

class Mydatasets(Dataset):
    def __init__(self, txt, transform = None, target_transform = None, loader = default_loader):
        super(Mydatasets, self).__init__()

        files = open(txt, 'r')
        imgs = []

        for file in files:
            file = file.strip('\n')
            file = file.rstrip('\n')

            words = file.split()

            if(len(words) == 2):
                imgs.append((words[0], int(words[1])))
            else:
                continue

        random.shuffle(imgs)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def imgShuffle(self):
        random.shuffle(self.imgs)


