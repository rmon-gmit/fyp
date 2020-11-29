import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../DATA/sammy_face.jpg')

# initially, a lot of edges will be found and the result looks very noisy
edges = cv2.Canny(image=img, threshold1=100, threshold2=100)
plt.imshow(edges)
plt.show()

# to counteract this, we can change the threshold values
# here is a formula to determine the best threshold values

med_val = np.median(img)

# lower threshold to either 0, or 70% of the median value, whichever is higher
lower_thresh = int(max(0, 0.7*med_val))

# upper threshold to either 255, or 130% of the median value, whichever is smaller
upper_thresh = int(max(255, 1.3*med_val))

edges2 = cv2.Canny(image=img, threshold1=lower_thresh, threshold2=upper_thresh)
plt.imshow(edges2)
plt.show()

# this wont really make that much of a difference,
# what will make a difference is blurring the image before processing it

blurred = cv2.blur(src=img, ksize=(5, 5))

edges3 = cv2.Canny(image=blurred, threshold1=lower_thresh, threshold2=upper_thresh)
plt.imshow(edges3)
plt.show()