#!/usr/bin/env python3

# You initiate a SURF object with some optional conditions like 64/128-dim descriptors, Upright/Normal SURF etc. 
# All the details are well explained in docs. 
# Then as we did in SIFT, we can use SURF.detect(), SURF.compute() etc for finding keypoints and descriptors.
#! SURF isn't free to use anymore. Substitute is ORB

import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('images/eiffel.jpg',0)

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400

surf = cv.xfeatures2d.SURF_create(400)

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)

# Check present Hessian threshold
# We set it to some 50000. Remember, it is just for representing in picture.
# In actual cases, it is better to have a value 300-500
surf.setHessianThreshold(50000)

# Again compute keypoints and check its number.
kp, des = surf.detectAndCompute(img,None)

img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()

#--------------------------------------------------------
#TODO U-SURF

# Check upright flag, if it False, set it to True
surf.setUpright(True)

# Recompute the feature points and draw it
kp = surf.detect(img,None)
img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()

#All the orientations are shown in same direction. 
# It is faster than previous. 
# If you are working on cases where orientation is not a problem (like panorama stitching) etc, this is better.