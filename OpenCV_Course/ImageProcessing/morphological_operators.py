import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_img():
    blank_img = np.zeros((600, 600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img=blank_img,
                text='ABCDE',
                org=(50, 300),
                fontFace=font,
                fontScale=5,
                color=(255, 255, 255),
                thickness=25)
    return blank_img


def display_img(img):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


img = load_img()
display_img(img)

# morphological operators are essentially specialized kernels that receive a specific effect

# erosion morphological operator erodes the boundaries of foreground objects
# in the picture we have created, the text is considered foreground, the black considered background

kernel = np.ones((5, 5), dtype=np.uint8)

# more iterations will amplify the effect
erosion_result = cv2.erode(src=img, kernel=kernel, iterations=4)
display_img(erosion_result)

img2 = load_img()

# just creating an np array and filling it with random values between 0 and 2 (so 1 or 0)
white_noise = np.random.randint(low=0, high=2, size=(600, 600))
display_img(white_noise)

# now we combine with original image

# original image is scale 0 - 255, so we need to multiply our white noise by 255 so they are compatible
white_noise = white_noise * 255

# combining both images
noise_image = white_noise + img2

display_img(noise_image)

###############################################################
# now we want to get rid of this noise
# to do this we use a morphological operator called 'opening'

opening = cv2.morphologyEx(src=noise_image, op=cv2.MORPH_OPEN, kernel=kernel)
display_img(opening)

# here we are doing an operator called 'closing', which is useful for removing foreground noise

#############################################
# creating an image that has foreground noise

# creating normal image
img3 = load_img()

# assigning it random noise
black_noise = np.random.randint(low=0, high=2, size=(600, 600))

# changing values to -255 and 0
black_noise = black_noise * -255

# combining noise with original image
black_noise_img = black_noise + img3

# changing noisy image to have a val of 0 where val is -255
black_noise_img[black_noise_img == -255] = 0

# image with foreground noise only
display_img(black_noise_img)

# now we can do the morphological close
closing = cv2.morphologyEx(src=black_noise_img, op=cv2.MORPH_CLOSE, kernel=kernel)
display_img(closing)

# morphological gradient. Takes the difference between the dilation and erosion of an image
# it is essentially a basic form of edge detection

img4 = load_img()

gradient = cv2.morphologyEx(src=img4, op=cv2.MORPH_GRADIENT, kernel=kernel)

display_img(gradient)
