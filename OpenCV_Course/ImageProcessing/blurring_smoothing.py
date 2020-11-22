import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_img():
    image = cv2.imread('../00000003.jpg').astype(np.float32) / 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def show_img(image):
    plt.imshow(image, cmap='gray')
    plt.show()


img1 = load_img()
show_img(img1)

# gamma correction allows you to alter what is perceived ad brightness ina m image
gamma = 0.5

# np.power takes the pixel values and raises them by the power of gamma
bright_img = np.power(img1, gamma)
show_img(bright_img)


# low pass filter blur with 2d convolution

img2 = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img=img2,
            text="TEST",
            org=(50, 400),
            fontFace=font,
            fontScale=4,
            color=(255, 0, 0),
            thickness=4)

show_img(img2)

# setting up kernel for low pass filter
kernel = np.ones(shape=(5, 5), dtype=np.float32)/25
print(kernel)

# second parameter is known as desired depth, third param is the kernel
destination = cv2.filter2D(img2, -1, kernel)
show_img(destination)

# we can also blur images using cv2's blur method
img3 = load_img()
cv2.putText(img=img3,
            text="TEST",
            org=(50, 400),
            fontFace=font,
            fontScale=4,
            color=(255, 0, 0),
            thickness=4)

# larger the kernel size, more extreme the blur
blurred_img = cv2.blur(img3, ksize=(50, 50))
show_img(blurred_img)

img4 = load_img()
cv2.putText(img=img4,
            text="TEST",
            org=(50, 400),
            fontFace=font,
            fontScale=4,
            color=(255, 0, 0),
            thickness=4)

# gaussian image blurring
blurred_img2 = cv2.GaussianBlur(src=img4, ksize=(5, 5), sigmaX=10)
show_img(blurred_img2)

# median blur, actually good for removing noise from an image while still keeping detail !!!
blurred_img3 = cv2.medianBlur(src=img4, ksize=5)
show_img(blurred_img3)
