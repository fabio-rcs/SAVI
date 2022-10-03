#!/usr/bin/env python3

#Nighfall
#Carregar imagem lake.jpg do disco e mostrá-a

import numpy as np
import cv2

def main():

    print('Showing Image lake.jpg')

    image = cv2.imread("../SAVI/Parte02/images/lake.jpg")
    img = cv2.imread("../SAVI/Parte02/images/lake.jpg")

    #Load image

    img_dark=image
    img_dark[:,int(w/2):] = (image[:,int(w/2):]*0.5).astype(np.int8)     
    
    h,w,_= img.shape[::-1]

    #for y in range(0,h/2):
    #    for x in range(0,w):
     #       px=getPixel(img,x,y) 
      #      setBlue(px,y*(2.0/h)) 
       #     setRed(px,y*(2.0/h)) 
        #    setGreen(px,y*(2.0/h))
    
    cv2.imshow('Lake', img)
    cv2.imshow('Half Darkened',img_dark)
   # cv2.imshow('Nighfall',img_d)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

