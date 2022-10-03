#!/usr/bin/env python3

import math
import cv2
import numpy as np

def main():

    print('Showing Image lake.jpg')

    img = cv2.imread("../SAVI/Parte02/images/scene.jpg")
    template = cv2.imread("../SAVI/Parte02/images/wally.png")
    H,W,_=img.shape
    h,w,_=template.shape
    #Load image

    method = cv2.TM_CC0EFF

    #Apply template matching
    result = cv2.matchTemplate(img, template, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    cv2.imshow('Resultado',result)