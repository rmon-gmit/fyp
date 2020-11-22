import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('../00000003.jpg')
plt.imshow(img)
plt.show()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()


# HSV and HLS are other colour mappings
# Here is an example of how to convert colour mappings

img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()

# they will appear weird looking, this is normal, for some reason

img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
plt.imshow(img)
plt.show()