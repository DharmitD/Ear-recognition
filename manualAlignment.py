#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.

# Manual keypoint selecton GUI and interfaces
# Code based on https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

# External imports
import numpy as np
import cv2
import glob
import os

# Internal imports
import earImage
import parameters as p



# Function to call on detecting a mouse click
def getClick(event, x, y, flags, param):
    global ikeypoints
    #global imagedisp
    
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = (x, y)
        print("Keypoint at: ", x, ", ", y)
        
        # Save click location
        keypointList.append(refPt)
        # Place point onto image
        #cv2.circle(imagedisp,(x,y),10,(0,255,255),-11)
        #cv2.imshow("image", imagedisp)
        ikeypoints += 1 # increment keypoint click counter
    




NUM_KEYPOINTS = 5
ikeypoints = 0;
keypointList = []
imagedisp = []

cv2.namedWindow("image")
cv2.setMouseCallback("image", getClick)


def manualAlignment():
    global ikeypoints
    global keypointList
    #global imagedisp
    
    # File to save keypoints to
    fp = open('myKeypointsTEMP.csv', 'w')
    
    # Loop through all images
    filelist = glob.glob(p.DATA_PATH + "/*jpg*")
    print("Reading images from file")
    
    quitnow = False
    
    for file in filelist:
        if quitnow:
            break
            
        image = earImage.earImage(file) # read and initialize the image
        imagedisp = cv2.imshow("image", image.rawImage)
        
        # Wait for a keypress
        stillGoing = True
        while stillGoing:
            
            key = cv2.waitKey(1) & 0xFF
             
 
            # if we have the desired number of keypoints
            # go to the next image
            if ikeypoints >= NUM_KEYPOINTS or key == ord("f"):
                stillGoing = False
                # Print keypoints to file
                keypointsStr = ""
                for keypoint in keypointList:
                    keypointsStr += str(keypoint[0]/image.nx) + \
                              "," + str(keypoint[1]/image.ny) + ","
                print(keypointsStr)
                fp.write(image.nameString+","+keypointsStr+"\n")
                
                # Reset keypoints list and counter
                ikeypoints = 0
                keypointList = []
                
            # Quit and continue later
            if key == ord("q"):
                stillGoing = False
                quitnow = True
        
    fp.close()


            
            
# Call manual alignment function
manualAlignment()