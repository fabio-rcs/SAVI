#!/usr/bin/env python3

# Sometimes, you may need to find the corners with maximum accuracy. 
# cv.cornerSubPix() further refines the corners detected with sub-pixel accuracy. 
# We need to find the Harris corners first. 
# Then we pass the centroids of these corners (There may be a bunch of pixels at a corner, we take their centroid) to refine them. 
# Harris corners are marked in red pixels and refined corners are marked in green pixels. 
# For this function, we have to define the criteria when to stop the iteration. 
# We stop it after a specified number of iterations or a certain accuracy is achieved, whichever occurs first. 
# We also need to define the size of the neighborhood it searches for corners. 

import numpy as np
import cv2 as cv

filename = 'images/polygons.png'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# First find Harris corners
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
dst = cv.dilate(dst,None)
ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

# Find centroids
ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

# Define the criteria to stop and refine the corners
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

# Now draw them
res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

cv.imshow('subpixel15.jpg',img)

if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()