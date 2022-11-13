#!/usr/bin/env python3

# sift.detect() function finds the keypoint in the images. 
# You can pass a mask if you want to search only a part of image. 
# Each keypoint is a special structure which has many attributes like its (x,y) coordinates, 
# size of the meaningful neighborhood, angle which specifies its orientation, response that specifies strength of keypoints etc.

# cv.drawKeyPoints() function draws the small circles on the locations of keypoints. 
# If you pass a flag, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS to it, it will draw a circle with size of keypoint and it will even show its orientation. 

import cv2 as cv

img = cv.imread('images/eiffel.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp = sift.detect(gray,None)

# Draws key points in the image
key_points = cv.drawKeypoints(gray,kp,img)

#Here kp will be a list of keypoints and des is a numpy array of shape (Number of Keypoints)×128.
sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)

cv.imshow('1st', key_points)

#Draws a circle with the size of the keypoint and shows it's orientation
key =cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('2nd', key)

if cv.waitKey(0) == ord('q'):
        exit()