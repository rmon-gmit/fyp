from time import sleep
import cv2
import gaze_direction

cap = cv2.VideoCapture(0)  # capturing video, will be replaced by a function get_frames which will get frames from a camera

while True:
    ret, frame = cap.read(0)  # getting the frame from the video, this will be replaced by the return contents of get_frames
    face, head_loc = gaze_direction.get_face(frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

    cv2.imshow('Frame', frame)

    if face.shape != (1, 1, 3):
        cv2.imshow('Face', face)
        print(head_loc)

    sleep(0.33)

cap.release()
cv2.destroyAllWindows()
