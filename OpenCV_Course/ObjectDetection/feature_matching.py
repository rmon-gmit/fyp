import cv2
import numpy as np
import matplotlib.pyplot as plt


def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


reeses = cv2.imread('../DATA/reeses_puffs.png', 0)
display(reeses)


cereals = cv2.imread('../DATA/many_cereals.jpg', 0)
display(cereals)

# brute force detection with ORB (pronounced 'o r b') descriptors

# creating detector
orb = cv2.ORB_create()
# so what we have done here is create a detection object
# we are now going to run it on the two images
# this will return feature 'keypoint's' and feature 'descriptors'

# note on keypoint's and descriptors:
# keypoint's are essentially the general information of a feature (location, size etc.)
# whereas a descriptor is essentially describing the area surrounding the keypoint so that it may be compared
# good explanation here:
# https://answers.opencv.org/question/37985/meaning-of-keypoints-and-descriptors/?sort=oldest
# and more on features here:
# https://docs.opencv.org/3.4/df/d54/tutorial_py_features_meaning.html

kp1, des1 = orb.detectAndCompute(image=reeses, mask=None)
kp2, des2 = orb.detectAndCompute(image=cereals, mask=None)

# note on hamming distance:
# when comparing two binary strings of same length,
# the hamming distance is the sum of bits that are different between both strings
# this is easiest described by an XOR operation on the strings
# taking two binary strings a: 1001 0100 and b: 1101 0101
# 1001 0100 XOR 1101 0101 = 0100 0001
# the result shows that there are two times that the bits did not match up
# so the hamming distance is 2

# brute force matching for binary string based descriptors, takes in the hamming distance,
# crossCheck confuses me... its a default value
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
# creating a brute force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# this will return the a list of the best matches for each feature that is found
# (des1 & des2 contain many descriptors)
# the objects in the list actually have a few values, one of these is the hamming distance
matches = bf.match(des1, des2)

# sorting matches in order of hamming distance, more distance = worse match
matches = sorted(matches, key=lambda x:x.distance)

reeses_matches = cv2.drawMatches(img1=reeses,               # target image
                                 keypoints1=kp1,            # target keypoint
                                 img2=cereals,              # secondary image
                                 keypoints2=kp2,            # secondary keypoint
                                 matches1to2=matches[:25],  # choosing first 25 matches
                                 outImg=reeses,
                                 matchesMask=None,          # dont want a mask
                                 flags=2)

display(reeses_matches)

# result will be very bad, better methods explored in next example