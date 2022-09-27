#!/usr/bin/env python3

#Carregar imagem lake.jpg do disco e mostrá-a

import cv2
import numpy as np

def main():

    print('Showing Image lake.jpg')

    image = cv2.imread("../SAVI/Parte02/images/lake.jpg")
    #Load image

    cv2.imshow('Display window', image)
    cv2.waitKey(0)
    #Show image

    h,w,_=image.shape[::]
    img_dark=image
    img_dark[:,int(w/2):] = (image[:,int(w/2):]*0.5).astype(np.int8)     
    cv2.imshow('Darkened Image',img_dark)
    cv2.waitKey(0)

    for i in no.arange(1,0.2,-0.01):
        

if __name__ == '__main__':
    main()

