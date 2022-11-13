#!/usr/bin/env python3

# Brute-Force matcher is simple. 
# It takes the descriptor of one feature in first set and is matched with all other features in second set using some distance calculation. 
# And the closest one is returned.

import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('images/eiffel2.jpg',cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread('images/eiffel.jpg',cv.IMREAD_GRAYSCALE) # trainImage

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

if cv.waitKey(0) == ord('q'):
        exit()


#The result of matches = bf.match(des1,des2) line is a list of DMatch objects. This DMatch object has following attributes:

    # DMatch.distance - Distance between descriptors. The lower, the better it is.
    # DMatch.trainIdx - Index of the descriptor in train descriptors
    # DMatch.queryIdx - Index of the descriptor in query descriptors
    # DMatch.imgIdx - Index of the train image.
