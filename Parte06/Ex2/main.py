#!/usr/bin/env python3

# Imports
import cv2 as cv
from random import randint
import numpy as np
from copy import deepcopy

def main():
        # Read images
        img_1 = cv.imread('../images/santorini/1.png')
        img_2 = cv.imread('../images/santorini/2.png')

        # Create working copies
        img_1_gui = deepcopy(img_1)
        img_2_gui = deepcopy(img_2) 

        # Create work window
        cv.namedWindow('Images', cv.WINDOW_NORMAL)

        # Resize image
        img_1_gui = cv.resize(img_1_gui, (1000, 1000), interpolation= cv.INTER_AREA)
        img_2_gui = cv.resize(img_2_gui, (1000, 1000), interpolation= cv.INTER_AREA)

        # --------------------------------------------------------------------------
        #? Image resizing using a scale
        # scale_percent = 20 # percent of original size
        # width_1 = int(img_1.shape[1] * scale_percent / 100)
        # height_1 = int(img_1.shape[0] * scale_percent / 100)
        # dim_1 = (width_1, height_1)

        # width_2 = int(img_2.shape[1] * scale_percent / 100)
        # height_2 = int(img_2.shape[0] * scale_percent / 100)
        # dim_2 = (width_2, height_2)

        # img_1 = cv.resize(img_1, dim_1, interpolation = cv.INTER_AREA)
        # img_2 = cv.resize(img_2, dim_2, interpolation = cv.INTER_AREA)
        # --------------------------------------------------------------------------

        # Create gray version of images
        gray_1 = cv.cvtColor(img_1_gui,cv.COLOR_BGR2GRAY)
        gray_2 = cv.cvtColor(img_2_gui,cv.COLOR_BGR2GRAY)

        # Create SIFT object and apply it to the images
        sift = cv.SIFT_create(nfeatures = 500)
        kp1 = sift.detect(gray_1, None)
        kp2 = sift.detect(gray_2, None)

        # Draws key points in the image 
        for idx, key_point in enumerate(kp1):
                x = int(key_point.pt[0])
                y = int(key_point.pt[1])
                color = (randint(0, 255), randint(0, 255), randint(0, 255))
                cv.circle(img_1_gui, (x, y), 10, color, 2)

        for idx, key_point in enumerate(kp2):
                x = int(key_point.pt[0])
                y = int(key_point.pt[1])
                color = (randint(0, 255), randint(0, 255), randint(0, 255))
                cv.circle(img_2_gui, (x, y), 10, color, 2)

        # Concatenate both images  
        img_3 = np.concatenate((img_1_gui, img_2_gui), axis = 1)

        # Show concatenated image in the created window
        cv.imshow('Images', img_3)

        cv.waitKey(0)

if __name__ == '__main__':
        main()