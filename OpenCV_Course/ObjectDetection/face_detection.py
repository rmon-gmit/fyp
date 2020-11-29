import cv2
import numpy as np
import matplotlib.pyplot as plt


def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()

# these are basic face and eye detection functions based on the viola-jones algorithm with haar cascades
# the face detection works reasonably well for all test images, but notice that the eye detection will
# not work with the photo of denis or the group photo


denis = cv2.imread('../DATA/Denis_Mukwege.jpg', 0)
display(denis)

nadia = cv2.imread('../DATA/Nadia_Murad.jpg', 0)
display(nadia)

group = cv2.imread('../DATA/solvay_conference.jpg', 0)
display(group)

face_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_frontalface_default.xml')


def detect_face(img):

    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)

    for(x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (255,255,255), 10)

    return face_img


result = detect_face(group)
display(result)


def adj_detect_face(img):

    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)

    for(x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (255,255,255), 10)

    return face_img


adj_result = adj_detect_face(group)
display(adj_result)

eye_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_eye.xml')


def detect_eyes(img):

    face_img = img.copy()
    eye_rects = eye_cascade.detectMultiScale(face_img)

    for(x,y,w,h) in eye_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (255,255,255), 10)

    return face_img


eye_result = detect_eyes(group)
display(eye_result)

# cap = cv2.VideoCapture(0)
#
# while True:
#
#     ret, frame = cap.read(0)
#
#     frame = adj_detect_face(frame)
#
#     cv2.imshow('Video Face Detect', frame)
#
#     k = cv2.waitKey(1)
#     if k == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()