import cv2
import numpy as np


#   function to retrieve a face from an image and return the area surrounding it
def get_face(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_cascade = cv2.CascadeClassifier('./haar_features/haarcascade_frontalface_default.xml')

    roi = np.full((1, 1, 3), 255, dtype=np.uint8)  # setting the region of interest to a 1x1 black square
    loc = [0, 0]
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)  # detecting faces

    for (x, y, w, h) in face_rects:  # iterating through each face detected
        roi = frame[y:y + h, x:x + w]  # setting roi to be the detected face
        loc = [x + (w / 2), y + (h / 2)]

    return roi, loc


