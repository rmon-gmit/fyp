import cv2
import numpy as np
import matplotlib.pyplot as plt

# finding grids is used commonly for camera calibration, opencv has two handy built in functions
# that can find chessboard style grid and a dots style grid

flat_chess = cv2.imread('../DATA/flat_chessboard.png')

found, corners = cv2.findChessboardCorners(image=flat_chess, patternSize=(7,7))

cv2.drawChessboardCorners(image=flat_chess, patternSize=(7, 7), corners=corners, patternWasFound=found)

plt.imshow(flat_chess)
plt.show()

dots = cv2.imread('../DATA/dot_grid.png')

found2, corners2 = cv2.findCirclesGrid(image=dots, patternSize=(10,10), flags=cv2.CALIB_CB_SYMMETRIC_GRID)

cv2.drawChessboardCorners(image=dots, patternSize=(10, 10), corners=corners2, patternWasFound=found2)
plt.imshow(dots)
plt.show()