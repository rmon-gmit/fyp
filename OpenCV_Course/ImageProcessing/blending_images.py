import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('../00000003.jpg')
img2 = cv2.imread('../copyright.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.imshow(img1)
plt.show()

plt.imshow(img2)
plt.show()

# Need to make images the same size in this case
img1 = cv2.resize(img1, (700, 450))
img2 = cv2.resize(img2, (700, 450))

plt.imshow(img2)
plt.show()

# blending the images
# formula = (src1*alpha) + (src2*beta) + gamma
# alpha and beta values will determine how prominent an image is
blended = cv2.addWeighted(src1=img1, alpha=0.5, src2=img2, beta=0.5, gamma=0)
plt.imshow(blended)
plt.show()

# overlaying images of different sizes

img_large = cv2.imread('../00000003.jpg')
img_small = cv2.imread('../copyright.jpg')

img_large = cv2.cvtColor(img_large, cv2.COLOR_BGR2RGB)
img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

# small image on top of smaller image with no blending
# numpy reassignment
img_small = cv2.resize(img2, (300, 300))

# offset is point image overlay will start
x_offset = 40
y_offset = 40

# shape[1] is horizontal, shape[0] is vertical
x_end = x_offset + img_small.shape[1]
y_end = y_offset + img_small.shape[0]

# choosing what x and y co ordinates of img_large to replace with img_small
# numpy has co-ordinates a little different so y is first here.
img_large[y_offset:y_end, x_offset:x_end] = img_small

plt.imshow(img_large)
plt.show()
