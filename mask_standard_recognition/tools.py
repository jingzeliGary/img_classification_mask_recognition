from torch.utils.data import Dataset
import cv2
import numpy as np

# 构建数据集
class MyDataset(Dataset):
    def __init__(self, img_list, label_list, transform=None):
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, item):
        image = self.img_list[item]
        label = self.label_list[item]
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.img_list)



# 人脸截取
def face_detect(img):
    # 人脸检测 SSD + opencv.dnn
    face_detector = cv2.dnn.readNetFromCaffe('./weights/deploy.prototxt.txt',
                                             './weights/res10_300x300_ssd_iter_140000.caffemodel')

    # 检测
    height, width = img.shape[:2]
    img_blob = cv2.dnn.blobFromImage(img, 1, (300, 300), (104.0, 177.0, 123.0))  # Blob：减均值， 缩放

    face_detector.setInput(img_blob)
    detections = face_detector.forward()

    # 解析坐标
    # 人脸数量
    nums_face = detections.shape[2]
    # print('检测的人脸数量', nums_face)

    for item in range(nums_face):
        # 置信度
        confidence = detections[0, 0, item, 2]
        # 选择置信度
        if confidence > 0.5:
            locations = detections[0, 0, item, 3:7] * np.array([width, height, width, height])
            # print(confidence * 100)

            px1_x, px1_y, px2_x, px2_y = locations.astype('int')
            cv2.rectangle(img, (px1_x, px1_y), (px2_x, px2_y), (0, 255, 0), 5)

            return img[px1_y:px2_y, px1_x:px2_x]

    return None
