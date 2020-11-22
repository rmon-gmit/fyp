import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_img(img):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


img = cv2.imread('../sudoku.jpg', 0)   # remember: 0 puts image to grayscale
display_img(img)

# calculating x gradient sobel
sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
display_img(sobelx)

# calculating y gradient sobel
sobely = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
display_img(sobely)

# blending these two images
blended = cv2.addWeighted(src1=sobelx, alpha=0.5, src2=sobely, beta=0.5, gamma=0)
display_img(blended)

######################################################################################
# NOTE ON PIXEL DEPTH:
#
# Depth is the "precision" of each pixel.
# Typically it can be 8/24/32 bit for displaying, but any precision for computations.
#
# Instead of precision you can also call it the data type of the pixel.
# The more bits per element, the better to represent different colors or intensities.
######################################################################################

# laplacian gradient
laplacian = cv2.Laplacian(src=img, ddepth=cv2.CV_64F)
display_img(laplacian)

# carrying out a gradient morphological operator on 'blended'
kernel = np.ones((4, 4), np.uint8)
gradient = cv2.morphologyEx(src=blended, op=cv2.MORPH_GRADIENT, kernel=kernel)
display_img(gradient)

# these processes are often combined when carrying out edge detection, although this is a basic example

