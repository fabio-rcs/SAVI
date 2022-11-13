#!/usr/bin/env python3

# Imports
from copy import deepcopy
import cv2 as cv

def main():

    # Read images
    img_1 = cv.imread('../images/castle/1.png')
    img_2 = cv.imread('../images/castle/2.png')

    # Create work copies
    img_1_gui = deepcopy(img_1)
    img_2_gui = deepcopy(img_2)

    # Create work window
    cv.namedWindow('img', cv.WINDOW_NORMAL)

    # Convert images to gray
    gray_1 = cv.cvtColor(img_1_gui,cv.COLOR_BGR2GRAY)
    gray_2 = cv.cvtColor(img_2_gui,cv.COLOR_BGR2GRAY)

    # Create SIFT object and apply it to the images
    sift = cv.SIFT_create(500)
    kp1, des1 = sift.detectAndCompute(gray_1,None)
    kp2, des2 = sift.detectAndCompute(gray_2,None)

    # Draws key points in the image
    cv.drawKeypoints(img_1_gui, kp1, gray_1)
    cv.drawKeypoints(img_2_gui, kp2, gray_2)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])
            
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img_1_gui, kp1, img_2_gui, kp2, good, None, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show matching results
    cv.imshow('img', img3)

    # Exit
    if cv.waitKey(0) == ord('q'):
            exit()

if __name__ == '__main__':
    main()