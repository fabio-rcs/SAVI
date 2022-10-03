#!/usr/bin/env python3

import cv2
import numpy as np

def main():
    img = cv2.imread("../Parte02/images/scene.jpg")
    template = cv2.imread("../Parte02/images/wally.png")
    #Load image and template

    H,W,_=img.shape
    h,w,_=template.shape

    method = cv2.TM_CCOEFF
    #Quanto maior o coeficiente de correlação, melhor

    #Apply template matching
    res = cv2.matchTemplate(img, template, method)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    #Queremos os máximos de forma a ter o maior coef.

    top_left=max_loc
    bottom_right=(top_left[0] + w, top_left[1]+h)

    cv2.rectangle(img, top_left, bottom_right,255,2)

    cv2.imshow('Resultado',img)
    cv2.waitKey(0)

if __name__=='__main__':
    main()