"""
    Name: Ross Monaghan
    File: gaze_direction_test.py
    Description: File containing functionality to test head pose estimation model on webcam
    Date: 15/05/21
"""

import cv2
import numpy as np
import hopenet
import utils

PROJECT_DIR = "C:/Users/rossm/OneDrive - GMIT/Year 4/Final Year Project/image_processing_code/"
BIWI_DATA_DIR = "C:/Users/rossm/kinect_head_pose_db/hpdb/"
AFLW2000_DATA_DIR = "C:/Users/rossm/AFLW2000/"
MODEL_PATH = PROJECT_DIR + "models/biwi_model_pretrained.h5"

face_cascade = cv2.CascadeClassifier('./haar_features/haarcascade_frontalface_default.xml')

BIN_NUM = 66
INPUT_SIZE = 64
BATCH_SIZE = 16

net = hopenet.HopeNet(input_size=INPUT_SIZE, num_bins=BIN_NUM, batch_size=BATCH_SIZE, model_path=MODEL_PATH, new=False)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read(0)
    if ret:
        face_rects = face_cascade.detectMultiScale(image=frame, scaleFactor=1.2, minNeighbors=5)  # detecting faces

        if len(face_rects) > 0:

            cropped_face, loc = utils.crop_face(frame, face_rects, INPUT_SIZE)
            cropped_face = np.expand_dims(cropped_face, 0)

            pred_cont_yaw, pred_cont_pitch = net.test(cropped_face)
            v2_img = utils.draw_axis(frame, pred_cont_yaw, pred_cont_pitch, ptx=loc[0], pty=loc[1])
            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
