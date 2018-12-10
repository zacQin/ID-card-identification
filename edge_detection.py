import cv2
import matplotlib.pyplot as plt
from skimage import io
import numpy as np

img = io.imread('image/timg-1.jpeg',0)
# io.imshow(img)
# cv2.imshow('before_smooth',img)
img = cv2.GaussianBlur(img,(3,3),0)
cv2.imshow('img1',img)
cv2.waitKey(0)

core=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)
img_new = cv2.filter2D(img,-1,kernel=core)
cv2.imshow('img2',img_new)
cv2.waitKey(0)
img =img_new

# img [40:190,240:370] = img [30,200]
# cv2.imshow('smooth',img)
# io.imshow(img)

img_gray = cv2.cvtColor(img_new,cv2.COLOR_BGR2GRAY)
plt.figure('img_gray')
io.imshow(img_gray)

ret, img_2value = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.figure('img_2value')

io.imshow(img_2value)

gray_edges = cv2.Canny(img_gray,100,200)
plt.figure('gray_edges')

io.imshow(gray_edges)


img_edge, contours, hierarchy = cv2.findContours(img_2value,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# for i in range(len(contours)):
#     img_edge1 = cv2.drawContours(img, contours, i, (0,0,255), 3)
#     cv2.imshow('figure 4',img_edge1)
    # cv2.waitKey(0)

st=[]
for i in range(len(contours)):
    st.append( cv2.arcLength(contours[i],True) )
themax = st.index(max(st))

img_edge1 = cv2.drawContours(img, contours, themax, (0,0,255), 3)
cv2.imshow('maxedge',img_edge1)
# cv2.waitKey(0)

epsilon = .1*cv2.arcLength(contours[themax],True)
approx = cv2.approxPolyDP(contours[themax],epsilon,True)
img1=cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
cv2.imshow('i', img1)


rect = cv2.minAreaRect(contours[themax])
box = np.int0(cv2.boxPoints(rect))

# cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
# cv2.imshow("Image", img)

Xs = [i[0] for i in box]
Ys = [i[1] for i in box]
x1 = min(Xs)
x2 = max(Xs)
y1 = min(Ys)
y2 = max(Ys)
hight = y2 - y1
width = x2 - x1

img = io.imread('timg-1.jpeg',0)
# cropImg = img_gray[y1:y1+hight, x1:x1+width]
cropImg = img_2value[y1:y1+hight, x1:x1+width]
cv2.imshow('copy',cropImg)
cv2.imwrite('detection.jpeg',cropImg)

print()