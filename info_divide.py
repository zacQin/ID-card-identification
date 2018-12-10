import cv2
import numpy as np
import pytesseract
from PIL import Image,ImageGrab
import smooth_sharpen as ss

img_Noface = cv2.imread('img.jpg')

def info_divide(img):

    img_Noface = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    index_x_y = []
    for ii in range(6):

        sum_horizon = np.sum(img_Noface, axis=1)
        point_ad = []

        for i in range(1,len(sum_horizon)-1):
            if sum_horizon[i] < sum_horizon [i+1] and sum_horizon [i] < sum_horizon [i-1]:
                point_ad.append ( sum_horizon [i] )

        themin_ad = np.where(sum_horizon == min(point_ad))
        # print(themin_ad)
        themin_ad = themin_ad[0][0]

        flag_x = 0
        flag_y = 0

        x = themin_ad-1
        y = themin_ad+1

        while flag_x == 0:
            tag1 = sum_horizon [x] >= ( max (sum_horizon) - sum_horizon[themin_ad]) / 7 * 6 + sum_horizon [themin_ad]
            tag2 = x <= themin_ad - int(len(sum_horizon)*0.095)
            if tag1 or tag2:
                leftpoint_ad = x
                flag_x+=1
            else:
                x-=1

        while flag_y == 0:
            tag1 = sum_horizon [y] >= ( max (sum_horizon) - sum_horizon [themin_ad]) / 7 * 6 + sum_horizon [themin_ad]
            tag2 = y >= themin_ad + int(len(sum_horizon)*0.095)
            if tag1 or tag2:
                rightpoint_ad = y
                flag_y+=1
            else:
                y+=1

        index_x_y.append((x,y))

        img_Noface[x:y,:]=255

        cv2.imshow('show', img_Noface)
        cv2.waitKey(0)

    def takeSecond(elem):
        return elem[1]
    # 指定第二个元素排序
    index_x_y.sort(key=takeSecond)

    return index_x_y

s = info_divide(img_Noface)


print()