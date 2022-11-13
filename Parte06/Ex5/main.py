#!/usr/bin/env python3

import numpy as np
import cv2 as cv
from copy import deepcopy
from random import randint

def main():

        # -------------------------------------
        #* Initialization
        # -------------------------------------

        # Read necessary images
        img_rgb_1 = cv.imread('../images/machu_pichu/2.png') # queryImage
        img_rgb_2 = cv.imread('../images/machu_pichu/1.png') # trainImage

        # Create work copies
        img_1_gui = deepcopy(img_rgb_1)
        img_2_gui = deepcopy(img_rgb_2)

        # Convert images to gray
        img_gray_1 = cv.cvtColor(img_1_gui,cv.COLOR_BGR2GRAY)
        img_gray_2 = cv.cvtColor(img_2_gui,cv.COLOR_BGR2GRAY)

        # Initiate SIFT detector
        sift = cv.SIFT_create(nfeatures=500)

        # Create normal sized windows
        cv.namedWindow('Stitched', cv.WINDOW_NORMAL)
        cv.namedWindow('Feature Matching', cv.WINDOW_NORMAL)
        # cv.namedWindow('Warped', cv.WINDOW_NORMAL)

        # -------------------------------------
        #* Execution
        # -------------------------------------

        # Find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img_gray_1,None)
        kp2, des2 = sift.detectAndCompute(img_gray_2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        # Store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
                if m.distance < 0.4*n.distance:
                        good.append(m)

        # Find homography if there's a minimum number of matches
        MIN_MATCH_COUNT = 10

        if len(good)>MIN_MATCH_COUNT:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()
                h,w, _ = img_1_gui.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv.perspectiveTransform(pts,M)
                img_2_gui = cv.polylines(img_2_gui,[np.int32(dst)],True,255,8, cv.LINE_AA)
        else:
                print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )

        # -------------------------------------
        #* Visualization
        # -------------------------------------

        color = (randint(0,255), randint(0,255), randint(0,255) )

        draw_params = dict(matchColor = color, # draw matches in random color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)

        img_3 = cv.drawMatches(img_1_gui, kp1, img_2_gui, kp2, good, None, **draw_params)
        cv.imshow('Feature Matching', img_3)

        #? Once you have the Homography matrix you need to transform one of the images to have the same perspective as the other. 
        #? This is done using the warpPerspective function in OpenCV. 

        # dst_1 = cv.warpPerspective(img_1_gui, M, ((img_2_gui.shape[1]), img_2_gui.shape[0])) # Warped image
        # cv.imshow('Warped', dst_1)
        
        # Once you've done the transformation, it's time to stich the images. 
        #* In this case there was no need to transform, they are already in the same orientation,
        #* Check main_classes.py for code with this feature

        # Gets vertice coordinates
        W = int(dst[0][0][0])
        H = int(dst[0][0][1])

        # Overlay one image onto the other
        img_2_gui[H:h+H, W:w+W] = img_1_gui

        cv.imshow('Stitched',img_2_gui)

        if cv.waitKey(0) == ord('q'):
                exit()

if __name__ == '__main__':
        main()