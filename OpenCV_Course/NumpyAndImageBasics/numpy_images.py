import numpy as np
import matplotlib.pyplot as plt
from PIL import Image   # python image library

stars = "*************"

pic = Image.open('../00000003.jpg')    # path needs to be accurate
# Image._show(pic)  # will open image with windows
pic_array = np.asarray(pic)     # setting the picture to a numpy array so numpy can access it

print(stars, " (vPixels, hPixels, RGB) ", stars)
print(pic_array.shape)  # will return array shape of (vertical pixels, horizontal pixels, colour depth)

plt.imshow(pic_array)   # show array as image
plt.show()      # display image

pic_red = pic_array.copy()  # making copy of image

# below we are accessing the shape of pic_red, which has the format [vPixels, hPixels, [R, G, B]]
# in this case we take all horizontal and vertical pixels, and access only element 0 of [R, G, B], which is red
plt.imshow(pic_red[:, :, 0])
plt.show()
# it actually displays kind of strangely because the colour scale is not RGB, but a matplotlib scale called 'viridis'
# refer to matplotlib documentation

# below we are accessing the red values instead as a greyscale image, which is more accurate to how the
# computer will interpret each colour.
plt.imshow(pic_red[:, :, 0], cmap='gray') # show red value in grayscale where 0(white) = no red and 255(black) = more red
plt.show()

# for green
plt.imshow(pic_red[:, :, 1], cmap='gray')
plt.show()

# for blue
plt.imshow(pic_red[:, :, 2], cmap='gray')
plt.show()

pic_red[:, :, 2] = 0    # removing all blue
pic_red[:, :, 1] = 0    # removing all green
plt.imshow(pic_red)     # showing image. result will be varying intensities of red!
plt.show()


