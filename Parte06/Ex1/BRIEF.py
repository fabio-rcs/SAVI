#!/usr/bin/env python3

# SIFT uses a feature descriptor with 128 floating point numbers. 
# Consider thousands of such features. 
# It takes lots of memory and more time for matching. 
# We can compress it to make it faster. 
# But still we have to calculate it first. 
# There comes BRIEF which gives the shortcut to find binary descriptors with less memory, faster matching, still higher recognition rate.

import cv2 as cv

img = cv.imread('images/eiffel.jpg',0)

# Initiate FAST detector
star = cv.xfeatures2d.StarDetector_create()

# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

# find the keypoints with STAR
kp = star.detect(img,None)

# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)

img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

print( brief.descriptorSize() )
print( des.shape )

cv.imshow('Results', img2)

if cv.waitKey(0) == ord('q'):
        exit()