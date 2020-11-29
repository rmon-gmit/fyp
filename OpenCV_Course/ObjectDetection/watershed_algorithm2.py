import cv2
import numpy as np
import matplotlib.pyplot as plt


def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


img = cv2.imread('../DATA/pennies.jpg')
img = cv2.medianBlur(img, 35)
display(img)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# otsu's method
ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
display(thresh)

# noise removal (for more complex images)

kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)
display(sure_bg)

# distance transform

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
display(dist_transform)

ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

display(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

display(unknown)

# creating label markers for watershed algorithm

ret, markers = cv2.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0
display(markers)

# apply algorithm

markers = cv2.watershed(img, markers)
display(markers)

contours, heirarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):

    if heirarchy[0][i][3] == -1:
        cv2.drawContours(img, contours, i, (255,0,0),10)

display(img)