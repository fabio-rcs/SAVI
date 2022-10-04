#!/usr/bin/env python3

from dataclasses import asdict
import cv2
import numpy as np

def main():

    #----------------------------------------------------
    #Initialization
    #----------------------------------------------------
    cap = cv2.VideoCapture("../docs/traffic.mp4")
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    
    #---------------------------------------------------
    #Execution
    #--------------------------------------------------
    while(cap.isOpened()): #infinite loop
        ret, frame = cap.read() #get a frame: ret will be true or false if getting succeeds

        if ret == True: #if we get a frame successfully
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Frame',frame) #show the image


            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 