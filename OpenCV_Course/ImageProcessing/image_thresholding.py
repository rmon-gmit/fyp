import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_pic(image):
    plt.imshow(image, cmap='gray')
    plt.show()


img = cv2.imread('../00000003.jpg', 0)
show_pic(img)

print(img.max())
thresh_val = round(img.max()/2)
print(thresh_val)

# thresholding takes an image and assigns its pixels a value based on certain parameters.
# thresh is the value that you want to be the point where values are changed
# type is the type of thresholding you want to do on the image
ret, thresh1 = cv2.threshold(img, thresh=thresh_val, maxval=img.max(), type=cv2.THRESH_TOZERO)

show_pic(thresh1)

cword = cv2.imread('../crossword.jpg', 0)
show_pic(cword)

cword_thresh = round(cword.max()/2)

ret2, thresh2 = cv2.threshold(cword, thresh=cword_thresh, maxval=cword.max(), type=cv2.THRESH_BINARY)
show_pic(thresh2)

thresh3 = cv2.adaptiveThreshold(cword, img.max(), cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
show_pic(thresh3)

blended = cv2.addWeighted(src1=thresh2, alpha=0.5, src2=thresh3, beta=0.5, gamma=0)
show_pic(blended)
