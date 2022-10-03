#!/usr/bin/env python3

import numpy as np  # shortcut or alias
import cv2


def main():

    image_rgb = cv2.imread('../Parte02/images/scene.jpg')
    H,W,_ = image_rgb.shape
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('../Parte02/images/wally.png')
    h,w,_ = template.shape

    result = cv2.matchTemplate(image_rgb, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    image_gui = image_rgb * 0

    top_left = (max_loc[0], max_loc[1])
    bottom_right = (max_loc[0] + w , max_loc[1] + h)
    color = (255,0,0) # BGR format
    cv2.rectangle(image_rgb, top_left, bottom_right, color, thickness=2)

    mask = np.zeros((H,W)).astype(np.uint8)
    cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    #cv2.imshow('Mask', mask)

    mask_bool = mask.astype(bool)
    image_gui[mask_bool] = image_rgb[mask_bool]

    negated_mask = np.logical_not(mask_bool)

    image_gray_3 = cv2.merge([image_gray, image_gray, image_gray])
    image_gui[negated_mask] = image_gray_3[negated_mask]

    cv2.imshow("Wally's there", image_rgb)
    #cv2.imshow('template', template)
    cv2.imshow('Highlight Wally', image_gui)
    cv2.waitKey(0)



if __name__ == "__main__": # checks if code was called from terminal
    main() # call main function