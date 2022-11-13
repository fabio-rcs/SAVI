#!/usr/bin/env python3

# All the other feature detection methods are good in some way. 
# But they are not fast enough to work in real-time applications like SLAM. 
# There comes the FAST algorithm, which is really "FAST".

import cv2 as cv

img = cv.imread('images/block.jpg',0)

# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

cv.imshow('fast_true.png', img2)

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)

print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )

img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
cv.imshow('fast_false.png', img3)

if cv.waitKey(0) == ord('q'):
        exit()