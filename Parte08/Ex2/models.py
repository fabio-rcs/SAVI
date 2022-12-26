#!/usr/bin/env python3
# --------------------------------------------------
# Fábio Sousa.
# SAVI, December 2022.
# --------------------------------------------------

import cv2
import numpy as np
from copy import deepcopy
from colorama import Fore, Style

class ImageMosaic():
    # Defines the model of an image mosaic

    def __init__(self, q_image, t_image):

        self.q_image = q_image # Query Image
        self.t_image = t_image # Target Image

        # Makes sure we are using float images on a the scale of 0-1 
        # Let's assume the input is uint8
        self.q_image = self.q_image.astype(float) / 255.0
        self.t_image = self.t_image.astype(float) / 255.0

        self.q_height, self.q_width, _ = q_image.shape
        self.t_height, self.t_width, _ = t_image.shape

        self.mask = q_image[:,:,0] > 0 # Compute overlap mask
        self.Params()

        cv2.namedWindow('Stitched Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Stitched Image', 600, 400)


        stitched_image_f = deepcopy(self.t_image)
        stitched_image_f[self.mask] = (self.q_image[self.mask] + stitched_image_f[self.mask] ) / 2
        
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Original', 600, 400)
        
        self.drawFloatImage('Original', stitched_image_f)


    def Params(self):
        # Start with neutral values 
        self.q_scale = 1.0 
        self.q_bias =  0.0
        self.t_scale = 1.0
        self.t_bias =  0.0

    def correctImage(self, image, scale, bias):
        image_c = scale * image + bias # Correction
        image_c[image_c > 1] = 1 # Saturate at 1
        image_c[image_c < 0] = 0 # Under saturate at 0
        return image_c

    def objectiveFunction(self, params):
        # Assume order of q_scale, q_bias, t_scale, t_bias
        self.q_scale = params[0] 
        self.q_bias =  params[1]
        self.t_scale = params[2]
        self.t_bias =  params[3]

        print(Fore.GREEN + Style.BRIGHT + 'Query Image:' + Style.RESET_ALL + Style.BRIGHT + '\nScale' + Style.RESET_ALL 
        + ' = ' + str(self.q_scale) + ', ' + Style.BRIGHT + 'Bias' + Style.RESET_ALL +  ' = ' + str(self.q_bias)) 
        print(Fore.BLUE + Style.BRIGHT + 'Target Image:' + Style.RESET_ALL + Style.BRIGHT + '\nScale' + Style.RESET_ALL 
        + ' = ' + str(self.t_scale) + ', ' + Style.BRIGHT + 'Bias' + Style.RESET_ALL +  ' = ' + str(self.t_bias)) 

        # Correct images with the parameters
        self.q_image_c = self.correctImage(self.q_image, self.q_scale, self.q_bias)
        self.t_image_c = self.correctImage(self.t_image, self.t_scale, self.t_bias)

        residuals = [] # Each residual will be the difference of a pixel
        
        # Matricidal form
        diffs = np.abs(self.t_image_c - self.q_image_c)
        diffs_in_overlap =diffs[self.mask] 
        residuals = np.sum(diffs_in_overlap)

        # Error is the sum of the residuals
        print(Fore.RED + Style.BRIGHT + 'Residual error = ' + Style.RESET_ALL + str(residuals) + '\n')

        # Draw for visualization
        self.draw()
        
        return residuals

    def drawFloatImage(self, win_name, image_f):
        image_uint8 = (image_f*255).astype(np.uint8)
        cv2.imshow(win_name, image_uint8)

    def drawBooleanImage(self, win_name, image_f):
        image_uint8 = (image_f*255).astype(np.uint8)
        cv2.imshow(win_name, image_uint8)


    def draw(self):

        stitched_image_f = deepcopy(self.t_image_c)
        stitched_image_f[self.mask] = (self.q_image_c[self.mask] + stitched_image_f[self.mask] ) / 2

        self.drawFloatImage('Stitched Image', stitched_image_f)
        cv2.waitKey(2)