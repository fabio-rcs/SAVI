#!/usr/bin/env python3

# cv.goodFeaturesToTrack() finds N strongest corners in the image by Shi-Tomasi method (or Harris Corner Detection, if you specify it). 
# Image should be a grayscale image. Then you specify number of corners you want to find. 
# Then you specify the quality level, which is a value between 0-1, which denotes the minimum quality of corner below which everyone is rejected. 
# Then we provide the minimum euclidean distance between corners detected.
# With all this information, the function finds corners in the image. 
# All corners below quality level are rejected. 
# Then it sorts the remaining corners based on quality in the descending order. 
# Then function takes first strongest corner, throws away all the nearby corners in the range of minimum distance and returns N strongest corners.
#? This function is more appropriate for tracking

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#Load image and convert to gray
img = cv.imread('images/block.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Finds 25 best corners
corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

# Draws a circle for each corner detected
for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)

# Show image with the corners detected
plt.imshow(img),plt.show()

if cv.waitKey() == ord('q'):
    exit()