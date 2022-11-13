#!/usr/bin/env python3

# ------------------------------- 
# Corner Harris detection parameters
#? cv.cornerHarris(img, blockSize, ksize, k)
#   img - Input image. It should be grayscale and float32 type.
#   blockSize - It is the size of neighborhood considered for corner detection
#   ksize - Aperture parameter of the Sobel derivative used.
#   k - Harris detector free parameter in the equation.
# ------------------------------

import numpy as np
import cv2 as cv


filename = 'images/polygons.png'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv.imshow('dst',img)

if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()

