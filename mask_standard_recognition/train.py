"""
1. 预处理 tranforms
2. 构造数据集
3. 划分数据集
4. 加载训练集，测试集
5. 加载模型
6. 训练
7. 评估
8. 保存模型

acc = 0.837109614206981
"""
import os.path

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from model import LeNet5
from tools import MyDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torchvision.models import resnet34


# 训练
def train(model, train_loader, loss_function, optimizer, lr_scheduler, device):
    model.train()
    loss = 0
    correct = 0

    for x_train, y_train in train_loader:
        x_train, y_train = x_train.to(device), y_train.to(device)
        # print(x_train.shape)  # shape [64, 3, 100, 100]
        output = model(x_train)
        loss = loss_function(output, y_train.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        loss += loss.item()
        y_pred = output.argmax(axis=1)
        correct += y_pred.eq(y_train).sum().item()

    acc = correct / len(train_loader.dataset)
    return loss, acc

# 测试
def test(model,test_loader,loss_function,device):
    model.eval()
    loss=0
    correct = 0

    with torch.no_grad():
        for x_test,y_test in test_loader:
            x_test,y_test = x_test.to(device), y_test.to(device)
            output=model(x_test)
            loss=loss_function(output,y_test.long())

            loss+=loss.item()
            y_pred=output.argmax(axis=1)
            correct += y_pred.eq(y_test).sum().item()

    acc = correct / len(test_loader.dataset)
    return loss, acc


# 预训练
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100, 100)),  # transforms 只处理 PIL Image格式
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_list = np.load('./data/img_list.npy', allow_pickle=True)
label_list = np.load('./data/label_list.npy', allow_pickle=True)
dataset = MyDataset(img_list, label_list, transform=transform)
print(len(dataset))

# 划分数据集
train_size = int(len(dataset) * 0.7)
test_size = (len(dataset) - train_size)
train_set, test_set = random_split(dataset, [train_size, test_size])
print(len(train_set))
print(len(test_set))

# 加载训练集， 测试集
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
print(len(train_loader))
print(len(test_loader))
print(len(test_loader.dataset))


# 加载模型
model = LeNet5()

# 训练 & 评估 & 保存模型
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Use {} to training'.format(device))

model = model.to(device)
loss_function = nn.CrossEntropyLoss()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.0001, weight_decay=0.1) # L2 正则化 (权重衰减）
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  # 学习率衰减
nums_epoch = 20

best_acc = 0
if not os.path.exists('best_model'):
    os.makedirs('./best_model')

save_path = './best_model/LeNet5.pth'
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

for epoch in range(nums_epoch):
    print('-'*10)
    print('The epoch:',epoch)
    train_loss, train_acc = train(model, train_loader, loss_function, optimizer, lr_scheduler, device)
    test_loss, test_acc = test(model, test_loader, loss_function, device)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    if test_acc>best_acc:
        best_acc=test_acc
        torch.save(model, save_path) # 只保存模型和其参数
print(best_acc)

# 可视化 loss, acc
all_train_loss=np.array(torch.tensor(train_loss_list, device='cpu'))
all_train_acc=np.array(torch.tensor(train_acc_list, device='cpu'))
all_test_loss=np.array(torch.tensor(test_loss_list, device='cpu'))
all_test_acc=np.array(torch.tensor(test_acc_list, device='cpu'))

plt.figure(figsize=(10,5))
plt.title("Train and Test Loss")
plt.plot(all_train_loss,label="train loss")
plt.plot(all_test_loss,label="test loss")
plt.xlabel("nums_epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Train and Test Loss')

plt.figure(figsize=(10,5))
plt.title("Train and Test Acc")
plt.plot(all_train_acc,label="train acc")
plt.plot(all_test_acc,label="test acc")
plt.xlabel("nums_epoch")
plt.ylabel("acc")
plt.legend()
plt.savefig('Train_Test_Acc')


