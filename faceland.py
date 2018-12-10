# def rect_to_bb(rect):
#     # take a bounding predicted by dlib and convert it
#     # to the format (x, y, w, h) as we would normally do
#     # with OpenCV
#     x = rect.left()
#     y = rect.top()
#     w = rect.right() - x
#     h = rect.bottom() - y
#
#     # return a tuple of (x, y, w, h)
#     return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
#                 help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
##### predictor = dlib.shape_predictor(args["shape_predictor"])
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load the input image, resize it, and convert it to grayscale
##### image = cv2.imread(args["image"])
image = cv2.imread('image/timg-13.jpeg')

# image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image

    for (x, y) in shape:
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

markpoint = []
for ii in [30,36,45,48,54]:
    markpoint .append((shape[ii][0],shape[ii][1]))


# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)


#coding=utf-8
import os,cv2,numpy
import logging
logging.basicConfig(
	level=logging.DEBUG,
	format='%(asctime)s %(levelname)s: %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

imgSize = [112, 96];

coord5point = [
               [48.0252, 71.7366],
               [30.2946, 51.6963],
               [65.5318, 51.6963],
               [33.5493, 92.3655],
               [62.7299, 92.3655],
]

face_landmarks = markpoint

def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),numpy.matrix([0., 0., 1.])])

def warp_im(img_im, orgi_landmarks,tar_landmarks):
    pts1 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    return dst

def main():
    pic_path = r'image/timg-13.jpeg'
    img_im = cv2.imread(pic_path)
    cv2.imshow('affine_img_im', img_im)
    dst = warp_im(img_im, face_landmarks, coord5point)
    cv2.imshow('affine', dst)
    crop_im = dst[0:imgSize[0], 0:imgSize[1]]
    cv2.imshow('affine_crop_im', crop_im)

if __name__=='__main__':
    main()
    cv2.waitKey()
    pass
