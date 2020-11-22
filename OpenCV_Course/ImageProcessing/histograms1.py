import matplotlib.pyplot as plt
import cv2
import numpy as np


def display_img(image):
    plt.imshow(image, cmap='gray')
    plt.show()


# Creating 2 variables for each image.
# The calculations we do will be done using opencv so we need rgb colour values (top variable) for thesse
# the bottom (show_) variable is for relating it to a picture for our viewing
dark_horse = cv2.imread('../DATA/horse.jpg')    # original bgr for opencv
show_horse = cv2.cvtColor(dark_horse, cv2.COLOR_BGR2RGB)    # converted to rgb for viewing

rainbow = cv2.imread('../DATA/rainbow.jpg')     # original bgr for opencv
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)     # converted to rgb for viewing

blue_bricks = cv2.imread('../DATA/bricks.jpg')      # original bgr for opencv
show_bricks = cv2.cvtColor(blue_bricks, cv2.COLOR_BGR2RGB)      # converted to rgb for viewing

display_img(show_horse)
display_img(show_rainbow)
display_img(show_bricks)

# calculating histogram values
# calcHist takes in the image as a list, channels decides what bgr channel you want,
# mask is for if you want to apply a mask to part of the image to exclude it form the histogram
# histSize gives you the upper limit, ranges gives you the range of values for the histogram
# change source and channel to see histogram change
hist_values = cv2.calcHist([dark_horse], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
print(hist_values.shape)
plt.plot(hist_values)
plt.show()


# now showing all 3 colour channels on one histogram
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([dark_horse], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 50])
    plt.ylim([0, 500000])

plt.title('HISTOGRAM FOR HORSE')
plt.show()
