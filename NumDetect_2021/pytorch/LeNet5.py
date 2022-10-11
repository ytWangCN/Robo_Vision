import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm

class LeNet5(nn.Module):
    def __init__(self, num_class):
        super(LeNet5, self).__init__()
        #卷积层
        self.Conv = nn.Sequential(
            nn.Conv2d(1, 6, 5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            #池化   
            nn.MaxPool2d(4, 4),
            nn.Conv2d(6, 16, 5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            #池化
            nn.MaxPool2d(4, 4)
        )

        #全连接层
        self.FC = nn.Sequential(
            nn.Linear(576, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_class)
        )

    def forward(self, x):
        #卷积层
        out = self.Conv(x) 
        out = out.view(out.size(0), -1)
        #全连接层
        out = self.FC(out)
        return out


