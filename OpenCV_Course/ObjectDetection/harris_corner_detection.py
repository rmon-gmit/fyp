import cv2
import numpy as np
import matplotlib.pyplot as plt

flat_chess = cv2.imread('../DATA/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)

gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)

real_chess = cv2.imread('../DATA/real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)

gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray_real_chess, cmap='gray')

# applying harris corner detection to these images

# harris corner detection requires floating points
# converting grayscale image to floats
gray = np.float32(gray_flat_chess)

# blockSize is the neighbourhood size, used to calculate the eigenvalues and eigenvectors
# ksize is the aperture parameter for the sobel operator (ie. the kernel size for the sobel operator)
# k is the harris detector free parameter (0.04 is good default val)
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)

# dilate is a morphological operator that we are using to show results on image below
# not necessary for harris algorithm
dst = cv2.dilate(dst, None)

# same
flat_chess[dst>0.01*dst.max()] = [255, 0, 0]
plt.imshow(flat_chess)
plt.show()

# now for the real chess board

gray2 = np.float32(gray_real_chess)
dst2 = cv2.cornerHarris(src=gray2, blockSize=2, ksize=3, k=0.04)

dst2 = cv2.dilate(dst2, None)

real_chess[dst2>0.01*dst2.max()] = [255, 0, 0]
plt.imshow(real_chess)
plt.show()

