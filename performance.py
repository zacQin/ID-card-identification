import cv2
from face_alignment import face_alignment
from face_base import find_face
from face_base import license_detection
from smooth_sharpen import smooth
from smooth_sharpen import sharpen
from face_base import face_wipeoff
import pytesseract

img = cv2.imread('/Users/qinfeiyu/PycharmProjects/image_recognition/valuable/Obama_1.jpg')

img = face_alignment(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face,face_plus = find_face(img_gray)

face_plus,lincese = license_detection(img,face_plus)

lincese = smooth(lincese)
lincese = sharpen(lincese)
cv2.imshow('license',lincese)
cv2.waitKey(0)

# cv2.imshow('license',lincese)
# cv2.waitKey(0)

lincese_gray = cv2.cvtColor(lincese, cv2.COLOR_BGR2GRAY)

# face,face_plus = find_face(lincese_gray)
# lincese_gray_noface = face_wipeoff(lincese_gray,face_plus)

# lincese_gray_noface = smooth(lincese_gray)
# lincese_gray_noface = sharpen(lincese_gray_noface)
# lincese_gray_noface = sharpen(lincese_gray_noface)

# cc = cv2.imread('/Users/qinfeiyu/PycharmProjects/image_recognition/valuable/id_num.png')
# text = pytesseract.image_to_string(cc,lang='chi_sim')
text = pytesseract.image_to_string(lincese,lang='chi_sim')
print(text)


cv2.imshow('license',lincese_gray)
cv2.waitKey(0)





print()