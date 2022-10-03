#!/usr/bin/env python3

import math
import cv2
import numpy as np

def main():
    img = cv2.imread('../Parte02/images/lake.jpg')
    res= img.copy()
    h,w,_=img.shape[::]
    index=int(w/2)

    for i in np.arange(1,0.2,-0.01):
        res[:,index:w,:]=(img[:,index:w,:]*i).astype(np.uint8)
        cv2.imshow('NightFall',res)
        cv2.waitKey(30)

    cv2.waitKey(0)
if __name__== '__main__':
    main()