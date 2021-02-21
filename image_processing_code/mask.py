import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
# from math import cos, sin, sqrt, degrees, radians, pow


def create_mask(img, pitch, yaw, ptx=None, pty=None, size=300, theta=20):
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

    x1 = size * (math.sin(yaw)) + ptx
    y1 = size * (-math.cos(yaw) * math.sin(pitch)) + pty

    pt2 = (int(x1), int(y1))

    cv2.line(img, head_ctr, pt2, (255, 0, 0), 2)

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

    cv2.line(img, head_ctr, edge1, (0, 255, 0), 2)
    cv2.line(img, head_ctr, edge2, (0, 255, 0), 2)

    x_max = img.shape[1] / 2

    # ellipse_x = x_max * 1/dir_len  # Ellipse x axis
    ellipse_x = 20
    ellipse_y = math.tan(math.radians(theta)) * dir_len   # Ellipse y axis

    if ellipse_x < 0:
        ellipse_x *= -1
    if ellipse_y < 0:
        ellipse_y *= -1

    cv2.ellipse(img,
                center=pt2,
                axes=(int(ellipse_x), int(ellipse_y)),  # axes need to change as length of line increases
                angle=ellipse_angle,  # angle needs to change with line direction
                startAngle=0,
                endAngle=360,
                color=(0, 0, 255),
                thickness=2)

    plt.imshow(img)
    plt.show()

    # mask
    mask = np.full(img1.shape, 0, dtype=np.uint8)
    roi = np.array([[(0, 400), (200, 200), (100, 640), (0, 640)]], dtype=np.int32)
    channel_count = img1.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi, ignore_mask_color)

    # apply the mask
    mask_image = cv2.bitwise_and(img1, mask)
    return mask_image


img1 = cv2.imread('images/00000003.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
print(img1.shape)

masked_image = create_mask(img1, pitch=-12, yaw=-20, theta=20)
# plt.imshow(masked_image)
# plt.show()
