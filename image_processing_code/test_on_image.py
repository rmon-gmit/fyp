from time import sleep

import cv2
import dlib
import numpy as np

import utils
from imutils import face_utils
from keras_preprocessing import image

import hopenet

PROJECT_DIR = "C:\\Users\\rossm\\OneDrive - GMIT\\Year 4\\Final Year Project\\image_processing_code\\"
FACE_LANDMARK_PATH = PROJECT_DIR + "haar_features\\shape_predictor_68_face_landmarks.dat"
MODEL_PATH = "C:\\Users\\rossm\\OneDrive - GMIT\\Year 4\\Final Year Project\\models\\biwi_model_pretrained.h5"
face_cascade = cv2.CascadeClassifier('./haar_features/haarcascade_frontalface_default.xml')

INPUT_SIZE = 64
BIN_NUM = 66
IMAGE_FILE = "./images/frame_00139_rgb.png"

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(FACE_LANDMARK_PATH)

net = hopenet.HopeNet(input_size=INPUT_SIZE, num_bins=BIN_NUM, model_path=MODEL_PATH, new=False)

img = cv2.imread(IMAGE_FILE)

print(str(img.shape[0]) + " " + str(img.shape[1]) + " " + str(img.shape[2]))

faces = []
angles = []

# while True:
#     face_rects = detector(img)
#     if len(face_rects) > 0:
#         shape = predictor(img, face_rects[0])
#         shape = face_utils.shape_to_np(shape)
#         face_crop = utils.crop_face_loosely(shape, img, INPUT_SIZE)
#         frames.append(face_crop)
#         if len(frames) == 1:
#             pred_cont_yaw, pred_cont_pitch, pred_cont_roll = net.test(frames)
#             cv2_img = utils.draw_axis(img, pred_cont_yaw, pred_cont_pitch, pred_cont_roll, ptx=shape[30][0],pty=shape[30][1], size=100)
#
#             cv2.imshow("cv2_img", cv2_img)
#             frames = []
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
while True:
    face_crop, loc = utils.get_face(frame=img, input_size=INPUT_SIZE, face_cascade=face_cascade)
    faces.append(face_crop)
    if len(faces) > 0:
        ptx = loc[0]
        pty = loc[1]
        pred_cont_yaw, pred_cont_pitch, pred_cont_roll = net.test(faces)
        angles.append(pred_cont_yaw)
        if len(angles) < 20:
            cv2_img = utils.draw_axis(img, pred_cont_yaw, pred_cont_pitch, pred_cont_roll, ptx=ptx, pty=pty, size=300)
            cv2.imshow("cv2_img", cv2_img)
        faces = []
    sleep(0.5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
