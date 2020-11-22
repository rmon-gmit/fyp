import cv2
import matplotlib.pyplot as plt
import numpy as np

# creating a 512 x 512 rgb image and setting the colour to all black
blank_img = np.zeros(shape=(512, 512, 3), dtype=np.int16)

# creating shapes in the image, giving it co-ordinates, setting the colour, setting the thickness
# try other shapes!
cv2.rectangle(blank_img, pt1=(100, 100), pt2=(400, 400), color=(0, 255, 0), thickness=3)
cv2.circle(blank_img, center=(256, 256), radius=100, color=(255, 0, 0), thickness=-1)
plt.imshow(blank_img)
plt.show()

