import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('../00000003.jpg')
img2 = cv2.imread('../copyright2.png')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img2 = cv2.resize(img2, (300, 300))

plt.imshow(img1)
plt.show()

print(img1.shape)

# tuple unpacking to get values for rows and columns
rows, cols, channels = img2.shape

# offsetting by size of rows and columns
x_offset = 428-cols
y_offset = 640-rows

# roi region of interest is just a small selection of the picture
roi = img1[y_offset:640, x_offset:428]
plt.imshow(roi)
plt.show()

# next step is creating the mask which will allow us to grab only red portion of copyright logo

# getting grayscale version of the copyright image
img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
# plt.imshow(img2_gray, cmap='gray')
# plt.show()

# you want to put in white what you want shown so we need to invert the greyscale image so only the logo is white
mask_inv = cv2.bitwise_not(img2_gray) # bitwise_not gets inverse of each bit
# plt.imshow(mask_inv, cmap='gray')
# plt.show()

# at the moment the shape of this image does not include rgb values
# we need to reintroduce this so that the original image and the mask shapes are compatible

# creating a full white background
white_background = np.full(img2.shape, 255, dtype=np.uint8)
# print(white_background.shape)

# or operation which is basically saying combine a few images and if there is black and not black, the not black is shown
# actually not sure if i am describing this correctly
bk = cv2.bitwise_or(src1=white_background, src2=white_background, mask=mask_inv)
# print(bk.shape)
# plt.imshow(bk, cmap='gray')
# plt.show()

# applying the or operation so that the red is printed ontop of the white
fg = cv2.bitwise_or(src1=img2, src2=img2, mask=mask_inv)
# plt.imshow(fg)
# plt.show()

# final region of interest is the regio of interest with the copyright logo
final_roi = cv2.bitwise_or(roi, fg)
# plt.imshow(final_roi)
# plt.show()

large_img = img1
small_img = final_roi

# just inputting the final roi where the location of the original roi was on the larger image
large_img[y_offset:y_offset+small_img.shape[0], x_offset:x_offset+small_img.shape[1]] = small_img
plt.imshow(large_img)
plt.show()
