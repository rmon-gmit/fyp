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

# sift descriptor
# https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(image=reeses, mask=None)
kp2, des2 = sift.detectAndCompute(image=cereals, mask=None)

bf = cv2.BFMatcher()

# knnMatch finds the k best matches for each descriptor from a query set, in this case first 2 matches
# this means that the result will be a 2d array of the 2 best matches for each descriptor
#
# each match has a distance value, which represents how close a descriptor matches another.
# in this case, we want to only use matches that are actually worthwhile.
# one way of doing this is by determining the difference between the best and 2nd best matches.
# first we need to make the assumption that only one keypoint will be equivalent in both images, therefore
# the second best match will always be some point that is not a good match.
# this brings us to the conclusion that if the distance value of the best match is actually quite close to the distance
# val of the 2nd best match, this best match may not be the most appropriate one to use for feature matching.
#
# lowes ratio test: https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
# more on feature matching: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
#
# we will apply a ratio test to the best and 2nd best matches of each descriptor to try to find the ones that are
# a good match, (further distance = better)
matches = bf.knnMatch(queryDescriptors=des1, trainDescriptors=des2, k=2)

# empty array of good matches
good = []

for match1, match2 in matches:
    # if the distance value of the best match is less than 50% of the distance value of the second best match
    # we decide the best match is good enough to add to the list
    if match1.distance < 0.50*match2.distance:
        good.append([match1])

print(len(matches))
print(len(good))

sift_matches = cv2.drawMatchesKnn(img1=reeses,
                                  keypoints1=kp1,
                                  img2=cereals,
                                  keypoints2=kp2,
                                  matches1to2=good,
                                  outImg=reeses,
                                  matchesMask=None,
                                  flags=2)

display(sift_matches)

######## FLANN based matcher ########

sift2 = cv2.SIFT_create()

kp3, des3 = sift2.detectAndCompute(image=reeses, mask=None)
kp4, des4 = sift2.detectAndCompute(image=cereals, mask=None)

# FLANN is faster than brute force matcher but only finds approximate nearest neighbours

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches2 = flann.knnMatch(des3, des4, k=2)

good2 = []

for match3, match4 in matches2:
    if match3.distance < 0.5 * match4.distance:
        good2.append([match3])

flann_matches = cv2.drawMatchesKnn(img1=reeses,
                                  keypoints1=kp3,
                                  img2=cereals,
                                  keypoints2=kp4,
                                  matches1to2=good2,
                                  outImg=reeses,
                                  matchesMask=None,
                                  flags=0)

display(flann_matches)

