import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms.transforms import Resize
from tqdm import tqdm
from LeNet5 import LeNet5
from Mydatasets import Mydatasets

# 定义学习速率
BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHES = 20

train_dataset = Mydatasets(txt='./TrainNum.txt', transform=transforms.ToTensor())
test_dataset = Mydatasets(txt='./TestNum.txt', transform=transforms.ToTensor())


# #加载MNIST数据集
# train_dataset = datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = LeNet5(5)

# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# 训练
for epoch in range(EPOCHES):
    print('*' * 25, 'epoch{}'.format(epoch + 1), '*' * 25)
    running_loss = 0.0
    running_acc = 0.0

    train_dataset.imgShuffle()
    test_dataset.imgShuffle()
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE, shuffle=True)

    for i, data in tqdm(enumerate(train_loader, 0)):
        img, label = data
        img = Variable(img)
        label = Variable(label)

        # 向前传播
        out = model(img)

        # 计算损失函数
        loss = criterion(out, label)
        running_loss = loss.item() * label.size(0)
        _, pred = torch.max(out, dim=1)

        num_correct = (pred == label).sum()

        running_acc += num_correct.item()

        # 手动清空梯度
        optimizer.zero_grad()

        # 向后传播
        loss.backward()
        optimizer.step()

    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))

    # 模型评估
    model.eval()
    eval_loss = 0
    eval_acc = 0
    # 测试模型

    for data in test_loader:
        try:
            img, label = data

            with torch.no_grad():
                img = Variable(img)
                label = Variable(label)

            # 向前传播
            out = model(img)

            loss = criterion(out, label)
            eval_loss += loss.item() * label.size(0)

            _, pred = torch.max(out, 1)

            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()

            # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            print("ERROR")

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print()

# 保存模型
torch.save(model, '../model/Vision_NumDetect.pth')
