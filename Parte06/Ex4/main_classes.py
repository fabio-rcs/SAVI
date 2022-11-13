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

    # FLANN parameters

    # For algorithms like SIFT, SURF etc. you can pass following: 
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

    search_params = dict(checks = 50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # Ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.4*n.distance: # The lower, the more exigent the test is 
            matchesMask[i]=[1,0]

    draw_params = dict(matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)
            
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img_1_gui, kp1, img_2_gui, kp2, matches, None, **draw_params)

    # Show matching results
    cv.imshow('img', img3)

    # Exit
    cv.waitKey(0)

if __name__ == '__main__':
    main()