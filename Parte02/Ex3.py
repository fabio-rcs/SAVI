#!/usr/bin/env python3

import cv2
import numpy as np

def main():
    img = cv2.imread("../Parte02/images/scene.jpg")
    img2 = cv2.imread("../Parte02/images/beach.jpg")
    img3= cv2.imread("../Parte02/images/school.jpg")
    template = cv2.imread("../Parte02/images/wally.png")
    #Load image and template

    H,W,_=img.shape
    h,w,_=template.shape

    method = cv2.TM_CCOEFF

    #Apply template matching
    res = cv2.matchTemplate(img, template, method)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    top_left=max_loc
    bottom_right=(top_left[0] + w, top_left[1]+h)

    cv2.rectangle(img, top_left, bottom_right,255,2)

    cv2.imshow('Resultado',img)
    cv2.waitKey(0)

    """Para a segunda imagem:"""
    res2 = cv2.matchTemplate(img2, template, method)
    _, _, _, max_loc2 = cv2.minMaxLoc(res2)

    top_left2=max_loc2
    bottom_right2=(top_left2[0] + w, top_left2[1]+h)

    cv2.rectangle(img2, top_left2, bottom_right2,255,2)

    cv2.imshow('Resultado',img2)
    cv2.waitKey(0)
    
    """Para a terceira imagem:"""
    res3 = cv2.matchTemplate(img3, template, method)
    _, _, _, max_loc3 = cv2.minMaxLoc(res3)

    top_left3=max_loc3
    bottom_right3=(top_left3[0] + w, top_left3[1]+h)

    cv2.rectangle(img3, top_left3, bottom_right3,255,2)

    cv2.imshow('Resultado',img3)
    cv2.waitKey(0)
    
if __name__=='__main__':
    main()