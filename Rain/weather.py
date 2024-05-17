# coding=utf-8
# @FileName:weather.py
# @Time:2024/5/8 
# @Author: CZH
import random

import cv2
import numpy as np
import os
import cv2
import numpy as np
from tqdm import tqdm


def get_noise(img, value=10):
    '''
    #生成噪声图像

    '''

    noise = np.random.uniform(0, 256, img.shape[0:2])
    # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    # 噪声做初次模糊
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)

    return noise

def rain_blur(noise, length=10, angle=0, w=1):
    '''
    将噪声加上运动模糊,模仿雨滴

    '''

    # 这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  # 生成对焦矩阵
    k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
    k = cv2.GaussianBlur(k, (w, w), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度

    # k = k / length                         #是否归一化

    blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波

    # 转换到0-255区间
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    return blurred

def alpha_rain(rain, img, beta=0.8):
    # 输入雨滴噪声和图像
    # beta = 0.8   #results weight
    # 显示下雨效果

    # expand dimensin
    # 将二维雨噪声扩张为三维单通道
    # 并与图像合成在一起形成带有alpha通道的4通道图像
    rain = np.expand_dims(rain, 2)
    rain_effect = np.concatenate((img, rain), axis=2)  # add alpha channel

    rain_result = img.copy()  # 拷贝一个掩膜
    rain = np.array(rain, dtype=np.float32)  # 数据类型变为浮点数，后面要叠加，防止数组越界要用32位
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    # 对每个通道先保留雨滴噪声图对应的黑色（透明）部分，再叠加白色的雨滴噪声部分（有比例因子）

    return rain_result

def weather(output_dir, w=3):
    """
    noise：输入噪声图，shape = img.shape[0:2]
    length: 对角矩阵大小，表示雨滴的长度
    angle： 倾斜的角度，逆时针为正
    w:      雨滴大小
    :return:
    """
    # Define the directory paths
    test_dir = r"G:\dataset\Traffic-Sign\GTSRB_128x128\test"
    # output_dir = "path_to_output_directory"
    os.makedirs(output_dir, exist_ok=True)
    # Iterate through each class directory in the test set
    for class_folder in tqdm(os.listdir(test_dir), desc="Processing"):
        class_path = os.path.join(test_dir, class_folder)
        output_class_path = os.path.join(output_dir, class_folder)
        os.makedirs(output_class_path, exist_ok=True)  # Create class folder in the output directory if it doesn't exist

        # Iterate through each image in the class folder
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            img = cv2.imread(image_path)

            # Apply rain effect
            noise = get_noise(img, value=500)
            rain = rain_blur(noise, length=random.randint(40, 70), angle=random.randint(-60, 60), w=w)
            rain_result = alpha_rain(rain, img, beta=0.6)()

            # Save processed image
            output_image_path = os.path.join(output_class_path, image_name)
            cv2.imwrite(output_image_path, rain_result)


if __name__ == '__main__':
    weather(r"F:\acm\pythonProject\new_dataset\weather_rain_1", 1)
    weather(r"F:\acm\pythonProject\new_dataset\weather_rain_3", 3)
    weather(r"F:\acm\pythonProject\new_dataset\weather_rain_5", 5)
    weather(r"F:\acm\pythonProject\new_dataset\weather_rain_7", 7)
