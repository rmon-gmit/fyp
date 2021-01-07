from time import sleep
import cv2
import dlib
import gaze_direction

cap = cv2.VideoCapture(0)  # capturing video, will be replaced by a function get_frames which will get frames from a camera
detector = dlib.get_frontal_face_detector()
face_cascade = cv2.CascadeClassifier('./haar_features/haarcascade_frontalface_default.xml')


def detect_face(img):

    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=6)

    for(x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (0, 0, 255))

    return face_img


while True:
    ret, frame = cap.read(0)  # getting the frame from the video, this will be replaced by the return contents of get_frames
    # face, head_loc = gaze_direction.get_face(frame)
    frame2 = detect_face(frame)

    faces = detector(frame)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))


    if cv2.waitKey(1) == 27:
        break

    cv2.imshow('Dlib', frame)
    cv2.imshow('ViolaJones', frame2)

    # if face.shape != (1, 1, 3):
    #     cv2.imshow('Face', face)
    #     print(head_loc)

    # sleep(0.33)

cap.release()
cv2.destroyAllWindows()
