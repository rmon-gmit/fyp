import os
import numpy as np
import cv2
import scipy.io as sio
import math

from math import cos, sin
import dlib
import matplotlib.pyplot as plt

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

    # cv2.imshow("Cropped Face", crop_face)

    crop_face = cv2.resize(crop_face, (input_size, input_size))
    input_img = np.asarray(crop_face, dtype=np.float32)
    cropped_face = (input_img - input_img.mean()) / input_img.std()

    return cropped_face

def create_mask(img, pitch, yaw, ptx=None, pty=None, size=1000, theta=15):
    # drawing a line
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)

    if ptx != None and pty != None:
        ptx = ptx
        pty = pty
    else:
        height, width = img.shape[:2]
        ptx = width / 2
        pty = height / 2

    head_ctr = (int(ptx), int(pty))

    x1 = size * (sin(yaw)) + ptx
    y1 = size * (-cos(yaw) * sin(pitch)) + pty

    pt2 = (int(x1), int(y1))

    # cv2.line(img, head_ctr, pt2, (255, 0, 0), 2)

    ####### Getting ellipse points ######
    if pitch > 0:
        dir_x = head_ctr[0] - pt2[0]
        dir_y = head_ctr[1] - pt2[1]
    else:
        dir_x = pt2[0]-head_ctr[0]
        dir_y = pt2[1]-head_ctr[1]

    dir_len = math.sqrt(dir_x*dir_x + dir_y*dir_y)  # Length of gaze direction line (in white)
    dir_angle = math.degrees(math.acos(dir_x/dir_len))

    # Ellipse point 1
    edge1_len = dir_len / math.cos(math.radians(theta))   # Length of line from head to ellipse point 1
    edge1_angle = dir_angle - theta
    if pitch > 0:
        edge1_x = head_ctr[0] - math.cos(math.radians(edge1_angle)) * edge1_len
        edge1_y = head_ctr[1] - math.cos(math.radians(90 - edge1_angle)) * edge1_len
    else:
        edge1_x = head_ctr[0] + math.cos(math.radians(edge1_angle)) * edge1_len
        edge1_y = head_ctr[1] + math.cos(math.radians(90-edge1_angle)) * edge1_len
    edge1 = (round(edge1_x), round(edge1_y))     # Point 1 of ellipse

    # Ellipse point 2
    edge2_len = dir_len / math.cos(math.radians(theta))   # Length of line from head to ellipse point 2
    edge2_angle = dir_angle + theta
    if pitch > 0:
        edge2_x = head_ctr[0] - math.cos(math.radians(edge2_angle)) * edge2_len
        edge2_y = head_ctr[1] - math.cos(math.radians(90 - edge2_angle)) * edge2_len
    else:
        edge2_x = head_ctr[0] + math.cos(math.radians(edge2_angle)) * edge2_len
        edge2_y = head_ctr[1] + math.cos(math.radians(90-edge2_angle)) * edge2_len
    edge2 = (round(edge2_x), round(edge2_y))     # Point 2 of ellipse

    ellipse_angle = dir_angle  # Ellipse angle of rotation

    # cv2.line(img, head_ctr, edge1, (0, 255, 0), 2)
    # cv2.line(img, head_ctr, edge2, (0, 255, 0), 2)

    x_max = img.shape[1] / 2

    # ellipse_x = x_max * 1/dir_len  # Ellipse x axis
    ellipse_x = 0
    ellipse_y = math.tan(math.radians(theta)) * dir_len    # Ellipse y axis

    if ellipse_x < 0:
        ellipse_x *= -1
    if ellipse_y < 0:
        ellipse_y *= -1

    # cv2.ellipse(img,
    #             center=pt2,
    #             axes=(int(ellipse_x), int(ellipse_y)),
    #             angle=ellipse_angle,
    #             startAngle=0,
    #             endAngle=360,
    #             color=(0, 0, 255),
    #             thickness=2)

    # mask
    mask = np.full(img.shape, 0, dtype=np.uint8)
    roi = np.array([[head_ctr, (edge2_x, edge2_y), (edge1_x, edge1_y)]], dtype=np.int32)
    channel_count = img.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi, ignore_mask_color)

    # apply the mask
    mask_image = cv2.bitwise_and(img, mask)
    return mask_image
