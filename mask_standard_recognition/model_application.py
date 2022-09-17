"""
1. 加载模型和其最优参数
2. 调用摄像头，并使用模型进行预测

"""
import cv2
import torch
from model import LeNet5
from tools import face_detect
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Use {} to execute".format(device))

# 加载最优模型
model = torch.load('./best_model/LeNet5.pth')
model = model.to(device)

# 调用摄像头
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()

    # 镜像翻转
    img = cv2.flip(img, 1)  #

    # 人脸截取
    face_region = face_detect(img)

    # 预训练
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 100)),  # transforms 只处理 PIL Image格式
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if face_region is not None:
        face_region = transform(face_region)
        # print(face_region.shape)
        face_region = torch.unsqueeze(face_region, 0) # 添加batch_size
        # print(face_region.shape)
        face_region = face_region.to(device)

        # 模型测试
        model.eval()
        output = model(face_region)
        output = F.softmax(output)
        # print('output',output)
        y_pred = output.argmax(axis=1)

        text ='%s' %y_pred
        cv2.putText(img,text,(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # 显示图片
    cv2.imshow('Image',img)
    # 关闭条件
    if cv2.waitKey(10) & 0xFF == 27:
        break
# 释放
cap.release()
cv2.destroyWindow()

"""

# 预测
model.eval()
with torch.no_grad():
    x_test
    x_test.to(device)
    output = model(x_test)
    y_pred = output.argmax(axis=1)
"""