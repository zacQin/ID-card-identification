import face_base as fb
import smooth_sharpen as ss
import numpy as np
import cv2
import io

img = cv2.imread('image/timg-1.jpeg')

face,facepoint = fb.find_face(img)

license = fb.license_area_onface(img,facepoint)

license = ss.smooth(license)

license = ss.sharpen(license)

license_gray = cv2.cvtColor(license,cv2.COLOR_BGR2GRAY)

ret, license_2value = cv2.threshold(license_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

license_2value_no_face = fb.face_wipeoff(license_2value,facepoint)

cv2.imwrite('img1.jpg',license_2value_no_face)
cv2.imshow('two_values',license_2value_no_face)
cv2.waitKey(0)



print()

