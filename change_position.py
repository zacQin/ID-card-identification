import cv2
import numpy as np

#从文件路径中读入图片。

file_path = '/path to your image'
img = cv2.imread(file_path, 1)

#对图片做灰度化转换。

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#再进行图片标准化,将图片数组的数值统一到一定范围内。函数的参数
#依次是：输入数组，输出数组，最小值，最大值，标准化模式。

cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

#使用投影算法对图像投影。

horizontal_map = np.mean(img, axis=1)

import imutils
import numpy as np


# 定义 Radon 变换函数，检测范围-90 至 90,间隔为 0.5：

def radon_angle(img, angle_split=0.5):
    angles_list = list(np.arange(-90., 90. + angle_split,
                                 angle_split))

    # 创建一个列表 angles_map_max，存放各个方向上投影的积分最大
    # 值。我们对每个旋转角度进行计算，获得每个角度下图像的投影，
    # 然后计算当前指定角度投影值积分的最大值。最大积分值对应的角度
    # 即为偏转角度。

    angles_map_max = []
    for current_angle in angles_list:
        rotated_img = imutils.rotate_bound(img, current_angle)
        current_map = np.sum(rotated_img, axis=1)
        angles_map_max.append(np.max(current_map))

    adjust_angle = angles_list[np.argmax(angles_map_max)]

    return adjust_angle