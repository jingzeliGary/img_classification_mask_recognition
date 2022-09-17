"""
Name: 口罩识别
Image: 正常佩戴口罩， 没有佩戴口罩, 不规范佩戴口罩

1. 读取图像
2. 预处理 （ 人脸截取)
3. 保存 img_list, label_list 到 npy文件
4. 构建 Dataset

"""
import os
import glob
import cv2
import numpy as np
import tqdm
from tools import face_detect



img_list = []
label_list = []

# 读取图片
dir_list = os.listdir('./images')
print(dir_list)

label_dict={}
id = 0

for dir_name in dir_list:
    print(dir_name)
    if dir_name not in label_dict:
        label_dict['{}'.format(dir_name)] = id
        id += 1

    file_list = glob.glob(f'./images/{dir_name}/*.jpg'.format(dir_name))
    for file_name in tqdm.tqdm(file_list, desc=f'处理{dir_name}'.format(dir_name=dir_name)):
        img = cv2.imread(file_name)

        # 预处理
        face_region = face_detect(img)
        if face_region is not None:
            img_list.append(face_region)
            label = label_dict['{}'.format(dir_name)]
            label_list.append(label)

print(label_dict)
print(label_list)

# 保存 img_list, label_list
img_list = np.asarray(img_list)
label_list = np.asarray(label_list)
img_list = np.save('./data/img_list.npy', img_list)
label_list = np.save('./data/label_list.npy', label_list)
