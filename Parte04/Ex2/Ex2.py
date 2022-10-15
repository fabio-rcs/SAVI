#!/usr/bin/env python3

from collections import defaultdict
from copy import deepcopy
import numpy as np
import cv2
import csv

def main():
    #--------------------------
    # Initialization
    #--------------------------

    cap = cv2.VideoCapture('../docs/OxfordTownCentre/TownCentreXVID.mp4') #Gets a frame
    if (cap.isOpened() == False):
        print('Error opening the video') #Prints error message if getting fails

    cv2.namedWindow('Image GUI', cv2.WINDOW_NORMAL) # Creation of a window
    cv2.resizeWindow('Image GUI', 800, 500)    # Resize the image 
    file = '../docs/OxfordTownCentre/TownCentre-groundtruth.top' 

    number_of_persons = 0 
    csv_reader = csv.reader(open(file))
    for row in csv_reader: 
            
        if len(row) != 12:  # Skip badly formatted rows
            continue
        
        person_number, frame_number, _, _, _, _, _, _, body_left, body_top, body_right, body_bottom = row #Variables are strings #? They come from a text file
        person_number = int(person_number) # Convert strings to integers 
        if person_number >= number_of_persons:
            number_of_persons += 1
    
    colors = np.random.randint(0, 255, size = (number_of_persons ,3), dtype = int) 
    
    points_dict = defaultdict (list) # We use the default model for simplification
    points_list = [] # Point storage list
    #---------------------------
    #Execution
    #---------------------------
    # is_first_time == True

    frame_counter = 0

    while (cap.isOpened()):         #This is an infinite loop
        
        # Step 1: get a frame
        ret, image_rbg = cap.read() # ret will be true or false if getting a frame succeeds
        image_gui = deepcopy(image_rbg) # Image for graphic user interface. We create a copy so we don't interfere with the original one, where we have the pixels information
       
        if ret == False:
            break
        stamp  = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000

        # Read CSV with ground truth bounding boxes and draws them
        csv_reader = csv.reader(open(file))
        for row in csv_reader: 
           
            if len(row) != 12:  # Skip badly formatted rows
                continue

            person_number, frame_number, _, _, _, _, _, _, body_left, body_top, body_right, body_bottom = row #Variables are strings #? They come from a text file
            person_number = int(person_number) # Convert strings to integers 
            frame_number = int(frame_number)
            body_left = int(float(body_left))
            body_top = int(float(body_top))
            body_right = int(float(body_right))
            body_bottom = int(float(body_bottom))
            color = colors[person_number, :]

            if frame_number != frame_counter: # Doesn't draw bboxes of other frames
                continue
            
            # Coordinates of rectangles corners and the center of the point
            x1 = body_left
            y1 = body_top
            x2 = body_right
            y2 = body_bottom
            cx = int((x1 + x2)/2)
            cy = int(y2-20)

            cv2.rectangle(image_gui, (x1, y1), (x2, y2), (int(color[0]), int(color[1]), int(color[2])) , 4) # Draws the bounding box
            point = cv2.circle(image_gui, ((int(cx)),(int(cy))), 0, (int(color[0]), int(color[1]), int(color[2])), -1) # Draws the tracking point. Thickness -1 so that the circle is filled
            
            points_dict[person_number].append((cx, cy)) # Append the coordinates associated to a person

            if person_number not in points_list:# Checks if the point is already in list, or else the drawing of line will give error
                points_list.append(person_number) 
                continue
                # start_point = (cx, cy)
                # end_point = (cx, cy)
                # cv2.line(image_gui, start_point, end_point, (int(color[0]), int(color[1]), int(color[2])), 2) # Start point = end point, could also be continue

            else: #Draws a line between two points for each person
                length = len(points_dict[person_number]) # We iterate for each person 

                for pt in range(length): 

                    if not pt + 1 == length: # If not the last frame, draws line
                        start_point = (points_dict[person_number][pt][0], points_dict[person_number][pt][1]) 
                        end_point = (points_dict[person_number][pt+1][0], points_dict[person_number][pt+1][1]) 
                        cv2.line(image_gui, start_point, end_point, (int(color[0]), int(color[1]), int(color[2])), 2)

        cv2.imshow('Image GUI', image_gui)
        
        if cv2.waitKey(5) == ord('q'):
            break
        
        frame_counter += 1

    #---------------------------
    # Termination
    #--------------------------

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()