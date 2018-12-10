from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),np.matrix([0., 0., 1.])])

def warp_im(img_im, orgi_landmarks,tar_landmarks):
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    return dst

def face_alignment(image):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # image = cv2.imread('image/timg-13.jpeg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

    markpoint = []
    for ii in [30,36,45,48,54]:
        markpoint .append((shape[ii][0],shape[ii][1]))

    coord5point = [
                   [48.0252+100*5, 71.7366+200],
                   [30.2946+100*5, 51.6963+200],
                   [65.5318+100*5, 51.6963+200],
                   [33.5493+100*5, 92.3655+200],
                   [62.7299+100*5, 92.3655+200],
    ]

    face_landmarks = markpoint

    dst = warp_im(image, face_landmarks, coord5point)
    # cv2.imshow('affine', dst)
    # cv2.waitKey()

    return dst

# image = cv2.imread('image/timg-13.jpeg')
# image = face_alignment(image)
# cv2.imshow('affine', image)
# cv2.waitKey()