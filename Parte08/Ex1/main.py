#!/usr/bin/env python3
# --------------------------------------------------
# Fábio Sousa.
# SAVI, December 2022.
# --------------------------------------------------


import cv2
from scipy.optimize import least_squares
from models import ImageMosaic


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------

    # Two images, query (q) and target (t)
    q_path = '../images/machu_pichu/query_warped.png'
    q_image = cv2.imread(q_path)

    t_path = '../images/machu_pichu/target.png'
    t_image = cv2.imread(t_path)

    image_mosaic = ImageMosaic(q_image, t_image) # created the class

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    x0 = [image_mosaic.q_scale, image_mosaic.q_bias, image_mosaic.t_scale, image_mosaic.t_bias]
    _ = least_squares(image_mosaic.objectiveFunction, x0, verbose=2)

    image_mosaic.draw()
    cv2.waitKey(0)

    # ------------------------------------------
    # Termination
    # ------------------------------------------

if __name__ == "__main__":
    main()
