import cv2
from skimage import io
import numpy as np

# def find_face(img):
# #     #输入彩色图像
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     face_detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
# #     eye_detector = cv2.CascadeClassifier('models/haarcascade_eye.xml')
# #     faces = face_detector.detectMultiScale(img, 1.1, 6)
# #     for (x, y, w, h) in faces:
# #         candidate = img[y:y+h,x:x+w]
# #         eyes = eye_detector.detectMultiScale(candidate, 1.2, 6)
# #         if eyes !=[]:
# #             upper = y - int(h/2)
# #             lower = y + int(h/2*3)
# #             left = x - int(w/7*2)
# #             right =x + int(w/5*6)
# #             facepoint=[upper,lower,left,right]
# #             #返回面部信息
# #             return faces,facepoint
# #
# # def wipeoff_onface(img,facepoint,reservation=0):
# #     #擦出面部，输入灰度图（彩图也可以，但是希望是灰度的）
# #     img_face_wipeoff = img
# #     img_face_wipeoff[facepoint[0]:facepoint[1], facepoint[2]:facepoint[3]] = 255
# #     if reservation == 0:
# #         return img_face_wipeoff
# #     else:
# #         return img,img_face_wipeoff
# #
# # def license_area_onface(img,facepoint):
# #     #检测出证件区域，输入彩图
# #     upper = facepoint[0]
# #     lower = facepoint[1]
# #     left = facepoint[2]
# #     right = facepoint[3]
# #
# #     w = abs(left - right)
# #     h = abs(upper - lower)
# #     size = img.shape
# #
# #     area_left = left - 2*w
# #     area_right = right
# #     area_upper = upper - int(h/10)
# #     area_lower = lower + int(h/3)
# #
# #     if area_left < 0:
# #         area_left = 0
# #
# #     if area_right > size[1]:
# #         area_right = size[1]
# #
# #     if area_upper < 0:
# #         area_upper = 0
# #
# #     if area_lower > size[0]:
# #         area_lower = size[0]
# #
# #     license_area = img[area_upper:area_lower,area_left:area_right]
# #     # cv2.imshow('area', img)
# #     # cv2.waitKey(0)
# #     # cv2.imshow('area', license_area)
# #     # cv2.waitKey(0)
# #     return license_area
# #
# #
# # ####暂时别用
# # def license_detection(img):
# #     facepoint = find_face(img)
# #     if facepoint :
# #         license_area = license_area_onface(img,facepoint)
# #         return facepoint,license_area
# #     else:
# #         license_area =np.zeros([50,50])
# #         print('NO FACE')
# #         return license_area
# #
# # def face_wipeoff(img,facepoint):
# #     # facepoint = find_face(img)
# #     img = wipeoff_onface(img,facepoint)
# #     return img
# # ####################
# #
# # img = cv2.imread('image/timg-1.jpeg')
# #
# # a,b = find_face(img)
# # # # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # #
# # # img = license_detection(img)
# # # cv2.imshow('area',img)
# # # cv2.waitKey(0)
# # #
# # # img = face_wipeoff(img)
# # #
# # cv2.imshow('area',img)
# # cv2.waitKey(0)
#

def find_face(img_gray):
    #输入灰色图像
    img = img_gray
    face_detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    eye_detector = cv2.CascadeClassifier('models/haarcascade_eye.xml')
    faces = face_detector.detectMultiScale(img, 1.1, 6)
    for (x, y, w, h) in faces:
        candidate = img[y:y+h,x:x+w]
        eyes = eye_detector.detectMultiScale(candidate, 1.2, 6)
        if eyes !=[]:
            upper = y - int(h/2)
            lower = y + int(h/2*3)
            left = x - int(w/7*2)
            right =x + int(w/5*6)
            face_plus=[upper,lower,left,right]
            #返回面部信息
            return faces,face_plus

def wipeoff_onface(img_gray,face_plus,reservation=0):
    #擦出面部，输入灰度图（彩图也可以，但是希望是灰度的）
    img = img_gray
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow('license_2', img)
    cv2.waitKey(0)

    img_face_wipeoff = img
    img_face_wipeoff[face_plus[0]:face_plus[1], face_plus[2]:face_plus[3]] = 255
    if reservation == 0:
        return img_face_wipeoff
    else:
        return img,img_face_wipeoff

def license_area_onface(img,face_plus):
    #检测出证件区域，输入彩图
    upper = face_plus[0]
    lower = face_plus[1]
    left = face_plus[2]
    right = face_plus[3]

    w = abs(left - right)
    h = abs(upper - lower)
    size = img.shape

    area_left = left - 2*w
    area_right = right
    area_upper = upper - int(h/10)
    area_lower = lower + int(h/3)

    if area_left < 0:
        area_left = 0

    if area_right > size[1]:
        area_right = size[1]

    if area_upper < 0:
        area_upper = 0

    if area_lower > size[0]:
        area_lower = size[0]

    license_area = img[area_upper:area_lower,area_left:area_right]
    # cv2.imshow('area', img)
    # cv2.waitKey(0)
    # cv2.imshow('area', license_area)
    # cv2.waitKey(0)
    return license_area

def license_detection(img,face_plus):
    if face_plus :
        license_area = license_area_onface(img,face_plus)
        return face_plus,license_area
    else:
        license_area =np.zeros([50,50])
        print('NO FACE')
        return license_area

def face_wipeoff(img_gray,face_plus):
    img_gray = wipeoff_onface(img_gray,face_plus)
    return img_gray
