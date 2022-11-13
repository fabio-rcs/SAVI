#!/usr/bin/env python3

# -------------------------------------------------------------------------------
# You can try ORB (Oriented FAST and Rotated BRIEF) as an alternate to SURF in open cv. 
# It almost works as good as SURF and SIFT and it's free unlike SIFT and SURF which are patented and can't be used commercially.
# You can read about it more in openCV-python documentation
# -------------------------------------------------------------------------------

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('images/eiffel.jpg',0)
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()

if cv.waitKey(0) == ord('q'):
        exit()