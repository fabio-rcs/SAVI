#!/usr/bin/env python3

from collections import defaultdict
from copy import deepcopy
import numpy as np
import cv2
import csv
from functions import Detection, Tracker
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

    # Count the number of persons in the dataset
    number_of_persons = 0 
    csv_reader = csv.reader(open(file))
    for row in csv_reader: 
        
        if len(row) != 12:  # Skip badly formatted rows
            continue
        
        person_number, frame_number, _, _, _, _, _, _, body_left, body_top, body_right, body_bottom = row #Variables are strings #? They come from a text file
        person_number = int(person_number) # Convert strings to integers 
        if person_number >= number_of_persons:
            number_of_persons += 1
    
    # Create the colors for each person
    colors = np.random.randint(0, 255, size = (number_of_persons ,3), dtype = int) 

    body_detector = cv2.CascadeClassifier('haarcascade_fullbody.xml')

    detection_counter = 0
    tracker_counter = 0
    trackers = []
    iou_threshold = 0.85

    #---------------------------
    #Execution
    #---------------------------

    frame_counter = 0

    while (cap.isOpened()):         #This is an infinite loop
        
        #? Step 1: get a frame
        # ret will be true or false if getting a frame succeeds
        ret, image_rbg = cap.read()

        # Image for graphic user interface. We create a copy so we don't interfere 
        # with the original one, where we have the pixels information
        image_gui = deepcopy(image_rbg) 
        image_gray = cv2.cvtColor(image_rbg, cv2.COLOR_BGR2GRAY)
        
        if ret == False:
            break
        stamp  = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000

       
        
        #* Detection of persons
       
        bboxes = body_detector.detectMultiScale(image_gray, scaleFactor = 1.2, 
                                                minNeighbors = 4, 
                                                minSize = (20, 40), 
                                                flags = cv2.CASCADE_SCALE_IMAGE)    
       
        #---------------------------
        #Create detections per haar cascade bbox
        #---------------------------
        detections = []

        for bbox in bboxes:
            x1, y1, w, h = bbox
            detection = Detection(x1, y1, w, h, image_gray, id=detection_counter)
            detection_counter += 1
            detection.draw(image_gui)
            detections.append(detection)
            # cv2.imshow('detection' + str(detection.id), detection.image)
        
        #For each detection see if there's a tracker to which it should be associated

        for detection in detections:
                    for tracker in trackers:
                        tracker_bbox = tracker.detections[-1]
                        iou = detection.computeIOU(tracker_bbox)
                        # print('IOU(T' + str(tracker.id) + 'D' + 
                        #         str(detection.id) + ') = ' + str(iou))

                        #Associate detection with tracker
                        if iou > iou_threshold:
                             tracker.addDetection(detection)
        #---------------------------    
        #Create tracker for each detection
        #---------------------------    
        if frame_counter == 0:
            for detection in detections:
                tracker = Tracker(detection, id=tracker_counter)
                tracker_counter += 1
                trackers.append(tracker)

        #Draw tracker
        for tracker in trackers:
            tracker.draw(image_gui)        
            # print(tracker)

        cv2.imshow('Image GUI',image_gui)
        
        if cv2.waitKey(10) == ord('q'):
            break
        
        frame_counter += 1

    #---------------------------
    # Termination
    #--------------------------

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()