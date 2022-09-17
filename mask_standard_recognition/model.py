'''
搭建模型 LeNet-5
'''

import torch.nn as nn
import torch.nn.functional as F
import torch

# 搭建模型---
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(13*13*64, 166)
        self.fc2 = nn.Linear(166, 3)

        # 初始化参数 kaiming_normal_
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.maxpool2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.maxpool3(out)

        out = torch.flatten(out,1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)


        return out



'''

# 测试
input = torch.rand(1, 3, 100, 100)
model = LeNet5()
print(model)
output = model(input)
print(output.shape)

for param in model.named_parameters():
    print(param)

'''
