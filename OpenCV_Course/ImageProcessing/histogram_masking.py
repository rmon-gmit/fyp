import matplotlib.pyplot as plt
import cv2
import numpy as np


def display_img(image):
    plt.imshow(image, cmap='gray')
    plt.show()


rainbow = cv2.imread('../DATA/rainbow.jpg')     # original bgr for opencv
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)     # converted to rgb for viewing

display_img(show_rainbow)

img = rainbow

# creating a mask
mask = np.zeros(shape=img.shape[:2], dtype=np.uint8)
mask[300:400, 100:400] = 255
display_img(mask)

masked_img = cv2.bitwise_and(src1=img, src2=img, mask=mask)     # for calculations
show_masked_img = cv2.bitwise_and(show_rainbow, show_rainbow, None, mask)       # for visualization

display_img(show_masked_img)

hist_mask_values_red = cv2.calcHist(images=[rainbow],
                                    channels=[2],
                                    mask=mask,
                                    histSize=[256],
                                    ranges=[0, 256])

hist_values_red = cv2.calcHist(images=[rainbow],
                               channels=[2],
                               mask=None,
                               histSize=[256],
                               ranges=[0, 256])

plt.plot(hist_mask_values_red)
plt.title('RED HISTOGRAM FOR MASKED RAINBOW')
plt.show()

plt.plot(hist_values_red)
plt.title('RED HISTOGRAM FOR UNMASKED RAINBOW')
plt.show()