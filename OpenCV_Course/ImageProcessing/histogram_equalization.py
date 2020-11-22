import matplotlib.pyplot as plt
import cv2
import numpy as np


def display_img(img):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


gorilla = cv2.imread('../DATA/gorilla.jpg', 0)

display_img(gorilla)

# calculating histogram of grayscale image

hist_values = cv2.calcHist(images=[gorilla], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plt.plot(hist_values)
plt.show()

eq_gorilla = cv2.equalizeHist(src=gorilla)
display_img(eq_gorilla)

# calculating histogram of equalized image
eq_hist_values = cv2.calcHist(images=[eq_gorilla], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plt.plot(eq_hist_values)
plt.show()

# doing the same for the color image

color_gorilla = cv2.imread('../DATA/gorilla.jpg')
show_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2RGB)
display_img(show_gorilla)

# equalizing a histogram of a color image means
# we first need to translate the image to a hsv colourspace

hsv_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2HSV)

hsv_gorilla[:, :, 2] = cv2.equalizeHist(hsv_gorilla[:, :, 2])
eq_color_gorilla = cv2.cvtColor(hsv_gorilla, cv2.COLOR_HSV2RGB)
display_img(eq_color_gorilla)