import cv2
import matplotlib.pyplot as plt
from skimage import io
import numpy as np

def smooth (img,show=0):
    img_smooth = cv2.GaussianBlur(img,(3,3),0)
    if show == 1:
        cv2.imshow(img_smooth)
        cv2.waitKey(0)
    return img_smooth

def sharpen (img,show=0,kernel=0):
    if kernel == 0 :
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)
    img_sharpen  = cv2.filter2D(img,-1,kernel = kernel)
    if show == 1:
        cv2.imshow(img_sharpen)
        cv2.waitKey(0)

    return img_sharpen

def defind_kernel(UDLR,center):
    kernel = np.array([[0, UDLR, 0], [UDLR, center, UDLR], [0, UDLR, 0]], np.float32)
    return kernel






