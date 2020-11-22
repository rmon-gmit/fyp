import matplotlib.pyplot as plt
import cv2
import numpy as np

# the 0 denotes the default video
cap = cv2.VideoCapture(0)

# just assigning these variables
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# the fourcc parameter is important to get right for your specific operation system, for windows DIVX is used
# he calls this something like 'codac' but not sure if thats correct. check opencv docs for more info
writer = cv2.VideoWriter('../DATA/opencvPracticeVideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

# keep capturing
while True:
    # cap.read returns a tuple, we want the frame value from this
    ret, frame = cap.read()
    writer.write(frame)

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # change frame to gray and uncomment above line for grayscale image
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

writer.release()
cap.release()
cv2.destroyAllWindows()


