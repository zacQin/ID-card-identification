import cv2
import numpy as np

img = cv2.imread('timg-4.jpeg',0)

def threshold_demo(gray):
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    print("threshold value %s"%ret)

    return binary

img = threshold_demo(img)

ret,thresh = cv2.threshold(img,127,255,0)

contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.waitKey(0)
cnt = contours[0]
M = cv2.moments(cnt)
print()