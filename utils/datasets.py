import os
import cv2
import random
import numpy as np

import torch

def contrast_and_brightness(img):
    """对比度和亮度调整"""
    alpha = random.uniform(0.75, 1.25)  # 对比度系数
    beta = random.uniform(0.75, 1.25)   # 亮度系数
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst

def motion_blur(image):
    """运动模糊增强"""
    if random.randint(1,5) == 1:  # 20%概率应用运动模糊
        degree = 2  # 模糊程度
        angle = random.uniform(-360, 360)  # 模糊角度
        image = np.array(image)
    
        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    
        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred
    else:
        return image

def augment_hsv(img, hgain = 0.0138, sgain = 0.678, vgain = 0.36):
    """HSV色彩空间增强"""
    r = np.random.uniform(-0.5, 0.5, 3) * [hgain, sgain, vgain] + 1  # 随机增益
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))  # 转换到HSV空间
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)  # 色调查找表
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)  # 饱和度查找表
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)  # 明度查找表

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # 转换回BGR空间
    return img

def img_aug(img):
    """图像增强组合"""
    img = contrast_and_brightness(img)  # 对比度和亮度调整
    img = motion_blur(img)             # 运动模糊
    img = augment_hsv(img)             # HSV增强
    return img

def collate_fn(batch):
    """数据批处理函数"""
    img, label = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i  # 设置批次索引
    return torch.stack(img), torch.cat(label, 0)

class TensorDataset():
    """YOLO数据集类"""
    def __init__(self, path, img_size_width = 352, img_size_height = 352, imgaug = False):
        assert os.path.exists(path), "%s文件路径错误或不存在" % path

        self.path = path
        self.data_list = []
        self.img_size_width = img_size_width    # 图像宽度
        self.img_size_height = img_size_height  # 图像高度
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png']  # 支持的图像格式
        self.imgaug = imgaug  # 是否启用数据增强

        # 数据检查
        with open(self.path, 'r') as f:
            for line in f.readlines():
                data_path = line.strip()
                if os.path.exists(data_path):
                    img_type = data_path.split(".")[-1]
                    if img_type not in self.img_formats:
                        raise Exception("img type error:%s" % img_type)
                    else:
                        self.data_list.append(data_path)
                else:
                    raise Exception("%s is not exist" % data_path)

    def __getitem__(self, index):
        """获取单个数据样本"""
        img_path = self.data_list[index]
        label_path = img_path.split(".")[0] + ".txt"

        # 图像预处理
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size_width, self.img_size_height), interpolation = cv2.INTER_LINEAR) 
        # 数据增强
        if self.imgaug == True:
            img = img_aug(img)
        img = img.transpose(2,0,1)  # HWC转CHW格式

        # 加载标签文件
        if os.path.exists(label_path):
            label = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    l = line.strip().split(" ")
                    label.append([0, l[0], l[1], l[2], l[3], l[4]])  # [batch_idx, class, x, y, w, h]
            label = np.array(label, dtype=np.float32)

            if label.shape[0]:
                assert label.shape[1] == 6, '> 5 label columns: %s' % label_path
                #assert (label >= 0).all(), 'negative labels: %s'%label_path
                #assert (label[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s'%label_path
        else:
            raise Exception("%s is not exist" % label_path)  
        
        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        """返回数据集大小"""
        return len(self.data_list)