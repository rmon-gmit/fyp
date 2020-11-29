import cv2
import numpy as np
import matplotlib.pyplot as plt


flat_chess = cv2.imread('../DATA/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)

real_chess = cv2.imread('../DATA/real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(image=gray_flat_chess, maxCorners=64, qualityLevel=0.01, minDistance=10)

corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(flat_chess, (x, y), 3, (255, 0, 0), -1)
    plt.imshow(flat_chess)

plt.show()

corners2 = cv2.goodFeaturesToTrack(image=gray_real_chess, maxCorners=100, qualityLevel=0.01, minDistance=10)

corners2 = np.int0(corners2)

for i in corners2:
    x, y = i.ravel()
    cv2.circle(real_chess, (x, y), 3, (255, 0, 0), -1)
    plt.imshow(real_chess)

plt.show()