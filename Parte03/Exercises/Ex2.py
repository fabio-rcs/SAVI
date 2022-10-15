#!/usr/bin/env python3

import cv2 
import numpy as np


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    blackout_time = 0.5# secs
    threshold_difference = 20

    # Define rectangles (only once)
    rects = [{'name': 'r1', 'x1': 150, 'y1': 550, 'x2': 400, 'y2': 630, 'ncars': 0, 'tic_since_car_count': -500},
             {'name': 'r2', 'x1': 400, 'y1': 550, 'x2': 630, 'y2': 630, 'ncars': 0, 'tic_since_car_count': -500},
             {'name': 'r3', 'x1': 700, 'y1': 550, 'x2': 880, 'y2': 630, 'ncars': 0, 'tic_since_car_count': -500},
             {'name': 'r4', 'x1': 980, 'y1': 550, 'x2': 1150, 'y2': 630, 'ncars': 0, 'tic_since_car_count': -500}]

    cap = cv2.VideoCapture("../docs/traffic.mp4")
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    is_first_time = True
    while(cap.isOpened()): # this is an infinite loop

        # Step 1: get frame
        ret, image_rgb = cap.read() # get a frame, ret will be true or false if getting succeeds
        if ret == False:
            break
        stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
        
        # Step 2: convert to gray
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        # Step 3: get average color in rectangle


        for rect in rects:


            total = 0
            number_of_pixels = 0
            for row in range(rect['y1'],rect['y2']): # iterate all image pixels inside the rectangle
                for col in range(rect['x1'],rect['x2']):
                    number_of_pixels += 1
                    total += image_gray[row, col] # add pixel color to the total count
                
            # after computing the total we should divide to get the average
            rect['avg_color'] = int(total / number_of_pixels)

            # How to get the model average? We know that in the first frame there are no cars in the rectangles. The first measurement is the model average
            if is_first_time:
                rect['model_avg_color'] = rect['avg_color']


            # Compute the different in color and make a decision
            diff = abs(rect['avg_color'] - rect['model_avg_color'])


            if diff > 20 and (stamp - rect['tic_since_car_count']) > blackout_time:
                rect['ncars'] = rect['ncars'] + 1
                rect['tic_since_car_count'] = stamp


        is_first_time = False


        # Drawing --------------------------
       

        for rect in rects:
            # draw rectangles
            #cv2.rectangle(image_rgb, (rect['x1'],rect['y1']), (rect['x2'],rect['y2']), (0,255,0),2)
            
            # Add text with avg color
            # text = 'avg=' + str(rect['avg_color']) + ' m=' + str(rect['model_avg_color'])
            # image_rgb = cv2.putText(image_rgb, text, (rect['x1'], rect['y1']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

            # Add text with car count color
            text = 'No. of cars = ' + str(rect['ncars']) 
            image_rgb = cv2.putText(image_rgb, text, (rect['x1'], rect['y1']-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,230), 2, cv2.LINE_AA)

            # Add text time since last car count
            # text = 'Time since lcc=' + str(round(stamp - rect['tic_since_car_count'],1))  + ' secs'
            # image_rgb = cv2.putText(image_rgb, text, (rect['x1'], rect['y1']-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), , cv2.LINE_AA)

        cv2.imshow('image_rgb',image_rgb) # show the image
        # cv2.imshow('image_gray',image_gray) # show the image


        if cv2.waitKey(10) == ord('q'):
            break  


    # ------------------------------------------
    # Termination
    # ------------------------------------------
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
