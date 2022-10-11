import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import cv2
from tqdm import tqdm
from LeNet5 import LeNet5
from PIL import Image
from Mydatasets import Mydatasets

BATCH_SIZE = 128

train_dataset = Mydatasets(txt = './TrainNum.txt', transform = transforms.ToTensor())
test_dataset = Mydatasets(txt = './TestNum.txt', transform = transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# # 下载训练集 MNIST分类训练集
# train_dataset = datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#实例化
model = LeNet5(6)

#加载模型参数
model = torch.load('../model/Vision_NumDetect.pth')
# model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

criterion = nn.CrossEntropyLoss()

img = cv2.imread("../image/1.jpg", 0)
img  = cv2.resize(img, (28, 28))
img = Image.fromarray(img)
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

img = trans(img)
img = img.unsqueeze(0)  # 图片扩展多一维,[batch_size,通道,长，宽]

with torch.no_grad(): 
    model.eval()
    trace_script_modile=torch.jit.trace(model, img)
    trace_script_modile.save(r"../model/NumDetect.pt") #压缩好的模型存出来

# # 测试模型
# eval_loss = 0
# eval_acc = 0
# for data in test_loader:  
#     img, label = data

#     with torch.no_grad():
#             img = Variable(img)
#             label = Variable(label)

#     out = model(img)

#     loss = criterion(out, label)
#     eval_loss += loss.item() * label.size(0)

#     _, pred = torch.max(out, 1)

#     num_correct = (pred == label).sum()
#     eval_acc += num_correct.item()

# print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset))))
