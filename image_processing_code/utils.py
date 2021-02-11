import os
import numpy as np
import cv2
import scipy.io as sio
from math import cos, sin
import dlib


def draw_axis(img, yaw, pitch, ptx=None, pty=None, size=100):
    # pi / 180 = 1 degree, multiplying this by the euler angels
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)

    # ptx and pty are center points, setting it to center of frame if the inputs are null
    if ptx != None and pty != None:
        ptx = ptx
        pty = pty
    else:
        height, width = img.shape[:2]
        ptx = width / 2
        pty = height / 2

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + ptx
    y3 = size * (-cos(yaw) * sin(pitch)) + pty
    ctr_pnt = (int(ptx), int(pty))
    pt2 = (int(x3), int(y3))

    cv2.line(img, ctr_pnt, pt2, (255, 0, 0), 2)

    return img


#   function to retrieve a face from an image and return the area surrounding it
def get_face(frame, input_size, face_cascade):
    roi = np.full((1, 1, 3), 255, dtype=np.uint8)  # setting the region of interest to a 1x1 black square
    face_rects = face_cascade.detectMultiScale(image=frame, scaleFactor=1.2, minNeighbors=5)  # detecting faces
    loc = [0, 0]

    for (x, y, w, h) in face_rects:  # iterating through each face detected
        roi = frame[y - 10:y + h + 10, x - 10:x + w + 10]  # setting roi to be the detected face
        loc = [x + (w / 2), y + (h / 2)]

    roi = cv2.resize(roi, (input_size, input_size))
    input_img = np.asarray(roi, dtype=np.float32)
    normed_img = (input_img - input_img.mean()) / input_img.std()

    return normed_img, loc


def crop_face_loosely(shape, img, input_size):
    x = []
    y = []
    for (_x, _y) in shape:
        x.append(_x)
        y.append(_y)

    max_x = min(max(x), img.shape[1])
    min_x = max(min(x), 0)
    max_y = min(max(y), img.shape[0])
    min_y = max(min(y), 0)

    Lx = max_x - min_x
    Ly = max_y - min_y

    Lmax = int(max(Lx, Ly) * 2.0)

    delta = Lmax // 2

    center_x = (max(x) + min(x)) // 2
    center_y = (max(y) + min(y)) // 2
    start_x = int(center_x - delta)
    start_y = int(center_y - delta - 30)
    end_x = int(center_x + delta)
    end_y = int(center_y + delta - 30)

    if start_y < 0:
        start_y = 0
    if start_x < 0:
        start_x = 0
    if end_x > img.shape[1]:
        end_x = img.shape[1]
    if end_y > img.shape[0]:
        end_y = img.shape[0]

    crop_face = img[start_y:end_y, start_x:end_x]

    cv2.imshow('crop_face', crop_face)

    crop_face = cv2.resize(crop_face, (input_size, input_size))
    input_img = np.asarray(crop_face, dtype=np.float32)
    normed_img = (input_img - input_img.mean()) / input_img.std()

    return normed_img


def create_mask(img, yaw, pitch, ptx=None, pty=None, size=100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)

    if ptx != None and pty != None:
        ptx = ptx
        pty = pty
    else:
        height, width = img.shape[:2]
        ptx = width / 2
        pty = height / 2

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + ptx
    y3 = size * (-cos(yaw) * sin(pitch)) + pty
    ctr_pnt = (int(ptx), int(pty))
    pt2 = (int(x3), int(y3))

    cv2.line(img, ctr_pnt, pt2, (255, 255, 255), 2)
