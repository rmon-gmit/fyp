"""
    Name: Ross Monaghan
    File: utils.py
    Description: File containing utility methods
    Date: 15/05/21

    ** THE FOLLOWING CODE CONTAINS SECTIONS FROM THE TENSORFLOW ADAPTATION OF THE HOPENET HEAD POSE ESTIMATION MODEL **
    ** URL: https://github.com/Oreobird/tf-keras-deep-head-pose **

    @InProceedings{Ruiz_2018_CVPR_Workshops,
    author = {Ruiz, Nataniel and Chong, Eunji and Rehg, James M.},
    title = {Fine-Grained Head Pose Estimation Without Keypoints},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2018}
    }

"""

import numpy as np
import cv2
import imutils
from skimage import measure
import math
from math import cos, sin


# Method to draw a line from a persons head in the direction of the estimated gaze direction
def draw_axis(img, yaw, pitch, ptx=None, pty=None, size=300):
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


# Method to crop a face and return the cropped face and its location in the original image
def crop_face(frame, face_rects, input_size):
    (x, y, w, h) = face_rects[0]  # getting first face detected

    roi = frame[y - 55:y + h + 55, x - 55:x + w + 55]  # setting roi to be the detected face
    roi = cv2.resize(roi, (input_size, input_size))
    loc = [x + (w / 2), y + (h / 2)]

    input_img = np.asarray(roi, dtype=np.float32)
    normed_img = (input_img - input_img.mean()) / input_img.std()

    return normed_img, loc


# Method to crop a face loosely for use in dataset processing
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

    crop_face = cv2.resize(crop_face, (input_size, input_size))
    input_img = np.asarray(crop_face, dtype=np.float32)
    cropped_face = (input_img - input_img.mean()) / input_img.std()

    return cropped_face


# Method to create a gaze mask from the estimated gaze direction
def create_mask(img, pitch, yaw, ptx=None, pty=None, size=1000, theta=20):
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

    ####### Getting ellipse points ######
    if pitch > 0:
        dir_x = head_ctr[0] - pt2[0]
        dir_y = head_ctr[1] - pt2[1]
    else:
        dir_x = pt2[0] - head_ctr[0]
        dir_y = pt2[1] - head_ctr[1]

    dir_len = math.sqrt(dir_x * dir_x + dir_y * dir_y)  # Length of gaze direction line
    dir_angle = math.degrees(math.acos(dir_x / dir_len))

    # Ellipse point 1
    edge1_len = dir_len / math.cos(math.radians(theta))  # Length of line from head to ellipse point 1
    edge1_angle = dir_angle - theta
    if pitch > 0:
        edge1_x = head_ctr[0] - math.cos(math.radians(edge1_angle)) * edge1_len
        edge1_y = head_ctr[1] - math.cos(math.radians(90 - edge1_angle)) * edge1_len
    else:
        edge1_x = head_ctr[0] + math.cos(math.radians(edge1_angle)) * edge1_len
        edge1_y = head_ctr[1] + math.cos(math.radians(90 - edge1_angle)) * edge1_len

    # Ellipse point 2
    edge2_len = dir_len / math.cos(math.radians(theta))  # Length of line from head to ellipse point 2
    edge2_angle = dir_angle + theta
    if pitch > 0:
        edge2_x = head_ctr[0] - math.cos(math.radians(edge2_angle)) * edge2_len
        edge2_y = head_ctr[1] - math.cos(math.radians(90 - edge2_angle)) * edge2_len
    else:
        edge2_x = head_ctr[0] + math.cos(math.radians(edge2_angle)) * edge2_len
        edge2_y = head_ctr[1] + math.cos(math.radians(90 - edge2_angle)) * edge2_len

    # mask
    mask = np.full(img.shape, 0, dtype=np.uint8)
    roi = np.array([[head_ctr, (edge2_x, edge2_y), (edge1_x, edge1_y)]], dtype=np.int32)
    channel_count = img.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi, ignore_mask_color)

    # apply the mask
    mask_image = cv2.bitwise_and(img, mask)
    return mask_image, dir_len, head_ctr, pt2


# Method to return the most salient area in an image
def most_salient_area(dir_len, head_ctr, saliency_img):
    max_mean = 0
    most_salient = []

    gray = cv2.cvtColor(saliency_img, cv2.COLOR_BGR2GRAY)

    # Blurring to reduce noise
    blurred = cv2.GaussianBlur(src=gray, ksize=(41, 41), sigmaX=0)
    thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    # Perform connected component analysis on the image
    labels = measure.label(thresh)

    # Loop over unique labels
    for label in np.unique(labels):

        # If not background label
        if label != 0:

            label_mask = np.zeros(thresh.shape, dtype="uint8")
            label_mask[labels == label] = 255
            num_pixels = cv2.countNonZero(label_mask)

            # If area is not too small (eliminating noise)
            if num_pixels > 300:
                contour = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = imutils.grab_contours(contour)
                (contour_ctr, radius) = cv2.minEnclosingCircle(contour[0])  # contour_ctr = (x, y)

                if contour_ctr[0] > head_ctr[0]:
                    cnt_len_x = contour_ctr[0] - head_ctr[0]
                else:
                    cnt_len_x = head_ctr[0] - contour_ctr[0]

                if contour_ctr[1] > head_ctr[1]:
                    cnt_len_y = contour_ctr[1] - head_ctr[1]
                else:
                    cnt_len_y = head_ctr[1] - contour_ctr[1]

                cnt_len = math.sqrt(cnt_len_x * cnt_len_x + cnt_len_y * cnt_len_y)

                angle = math.degrees(math.acos(cnt_len / dir_len))

                mean = cv2.mean(gray, label_mask)
                current_mean = mean[0] * (1 / angle)
                if max_mean < current_mean:
                    max_mean = current_mean  # calculating average value of pixels within mask
                    most_salient = cv2.bitwise_and(gray, label_mask)
                    most_salient = cv2.cvtColor(most_salient, cv2.COLOR_GRAY2BGR)

    return most_salient
