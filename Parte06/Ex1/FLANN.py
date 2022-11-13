#!/usr/bin/env python3

# FLANN stands for Fast Library for Approximate Nearest Neighbors. 
# It contains a collection of algorithms optimized for fast nearest neighbor search in large datasets and for high dimensional features. 
# It works faster than BFMatcher for large datasets. 

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('images/eiffel2.jpg',cv.COLOR_BGR2GRAY) # queryImage
img2 = cv.imread('images/eiffel.jpg',cv.COLOR_BGR2GRAY) # trainImage

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters

# For algorithms like SIFT, SURF etc. you can pass following: 
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

# While using ORB, you can pass the following. 
# The commented values are recommended as per the docs, but it didn't provide required results in some cases. 
# Other values worked fine.
# FLANN_INDEX_LSH = 6
# index_params= dict(algorithm = FLANN_INDEX_LSH,
                #    table_number = 6, # 12
                #    key_size = 12,     # 20
                #    multi_probe_level = 1) #2

search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
                   
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()

if cv.waitKey(0) == ord('q'):
        exit()