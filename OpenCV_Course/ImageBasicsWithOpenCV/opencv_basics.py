import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../00000003.jpg')    # opencv image read
print(img.shape)
plt.imshow(img)
# below image shown will be a strange colour because opencv and matplot lib
# expect different orders of the rgb channels, matplotlib -> rgb opencv -> bgr
plt.show()

# below line converts the colour channel order BGR 'to' RGB
fixed_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fixed_img)
plt.show()

# importing it as a grayscale image
# check out other colour mappings too!
img_gray = cv2.imread('../00000003.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img_gray, cmap='gray')
plt.show()

# resizing images
rsz_img = cv2.resize(fixed_img, (150, 320))     # note: array dimensions goes (vSize, hSize) instead of (hSize, vSize)
plt.imshow(rsz_img)
plt.show()

# resizing by ratio
# below will resize it to half its original dimensions
rsz_img2 = cv2.resize(fixed_img, (0, 0), fixed_img, 0.5, 0.5)
plt.imshow(rsz_img2)
plt.show()

# flipping image across an axis
# cv2.flip(fixed_img, [] ) 0 = along horizontal axis, 1 = along vertical axis, -1 = along horizontal and vertical axis
flipped_img = cv2.flip(fixed_img, 1)
plt.imshow(flipped_img)
plt.show()

# saving an image file
flipped_img2 = cv2.cvtColor(flipped_img, cv2.COLOR_RGB2BGR)
cv2.imwrite('../newimg.jpg', flipped_img2)

