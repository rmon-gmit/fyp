import cv2
import numpy as np
import matplotlib.pyplot as plt

full = cv2.imread('../DATA/sammy.jpg')
full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)

plt.imshow(full)
plt.show()


face = cv2.imread('../DATA/sammy_face.jpg')
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
plt.imshow(face)
plt.show()

# something to note about these two images is that 'face' is just a cutout of 'full'
# looking at the dimensions of 'face', we would see that it has the same dimensions of
# that area of the full image


# side note:
# if you create a string that is also the name of a built in function, the eval
# method will transform that string into the relevant function
# very cool!
myvals = [1, 2, 3, 4, 5]
mystring = 'sum'
myfunc = eval(mystring)
print(sum(myvals))
print(myfunc(myvals))

# list of methods that we will use for template merging, for ease
# we will use the above eval method to transform these strings into a function
methods = ['cv2.TM_CCOEFF',
           'cv2.TM_CCOEFF_NORMED',
           'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED',
           'cv2.TM_SQDIFF',
           'cv2.TM_SQDIFF_NORMED']

for m in methods:

    # create a copy  of full image
    full_copy = full.copy()

    method = eval(m)

    # template matching
    res = cv2.matchTemplate(image=full_copy, templ=face, method=method)

    # match template returns a heat map of values where the correlation has occurred
    # depending on the method, the heat map may be brighter the greater the correlation
    # or brighter the lesser the correlation
    # we will find the max and min values/locations of the heat map and use this to
    # draw a rectangle on the copy of the image

    min_val, max_val, min_location, max_location = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_location     # (x, y)
    else:
        top_left = max_location     # (x, y)

    # assigning the bottom right of the rectangle
    height, width, channels = face.shape
    bottom_right = (top_left[0]+width, top_left[1]+height)

    # drawing it
    cv2.rectangle(img=full_copy, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=10)

    # this is just a way of showing the plots side by side
    plt.subplot(121)
    plt.imshow(res)
    plt.title('HEATMAP OF TEMPLATE MATCHING')

    plt.subplot(122)
    plt.imshow(full_copy)
    plt.title('DETECTION OF TEMPLATE')
    plt.suptitle(m)
    plt.show()

    print('\n\n')
