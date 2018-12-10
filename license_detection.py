import cv2
from skimage import io
import matplotlib.pyplot as plt

img = io.imread('image/timg-1.jpeg',0)
(B,G,R) = cv2.split(img)
# cv2.imshow('red_way',R)
# cv2.waitKey(0)

ret, img_2value = cv2.threshold(R, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# img_2value = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# ret, img_2value = cv2.threshold(R, 0, 255,  cv2.THRESH_TOZERO)
cv2.imshow('red_way',img_2value)
cv2.waitKey(0)

gray_edges = cv2.Canny(img_2value,100,200)
cv2.imshow('figure 4',gray_edges)
cv2.waitKey(0)

# ret, img_2value = cv2.threshold(R, 0, 255,  cv2.THRESH_OTSU)
img_edge, contours, hierarchy = cv2.findContours(img_2value,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# epsilon = 0.1*cv2.arcLength(cnt,True)
# approx = cv2.approxPolyDP(cnt,epsilon,True)

st=[]
for i in range(len(contours)):
    st.append( cv2.arcLength(contours[i],True) )
themax = st.index(max(st))

img_edge1 = cv2.drawContours(img, contours, -1, (0,0,255), 3)
cv2.imshow('figure 4',img_edge1)
cv2.waitKey(0)

print()