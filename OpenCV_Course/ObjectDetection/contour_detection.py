import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../DATA/internal_external.png', 0)
plt.imshow(img, cmap='gray')
plt.show()

# RETR_CCOMP will detect both 'internal' and 'external' contours
# this function will return a list of contours and a numpy array called hierarchy
contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)

# creating a black image the same shape as the image, we will then draw the contours onto that black image.
external_contours = np.zeros(img.shape)
internal_contours = np.zeros(img.shape)

# to do this we will use the contours and hierarchy objects that we retrieved from the findcontours method

# drawing only external contours
for i in range(len(contours)):

    # iterating through last column(3) of each row of the hierarchy nparray.
    # -1 denotes an external contour, anything else is internal
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(image=external_contours,   # drawing on this image
                         contours=contours,         # choosing the contour list to draw from
                         contourIdx=i,              # selecting specific contour to draw by its indec
                         color=255,                 # colour is white
                         thickness=-1)              # thickness of -1 means we fill the shape

plt.imshow(external_contours, cmap='gray')
plt.show()

# drawing only internal contours
for i in range(len(contours)):

    # iterating through last column(3) of each row of the hierarchy nparray.
    # this time getting all internal contours
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(image=internal_contours,   # drawing on this image
                         contours=contours,         # choosing the contour list to draw from
                         contourIdx=i,              # selecting specific contour to draw by its index
                         color=255,                 # colour is white
                         thickness=-1)              # thickness of -1 means we fill the shape

    # note on internal contours
    # the internal contours are assigned different numbers in the hierarchy,
    # depending on what external contours they belong to
    # in this case the contours in the pizza are given a val of 4, and the smiley face a val of 0


plt.imshow(internal_contours, cmap='gray')
plt.show()