#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.

# All adjustable parameters are in this file. It is imported by all the code
# files using import parameters as p, so that all parameters are 
# prefixed with "p."

# Folder, relative to run location, where imagery is stored
DATA_PATH = "data"

# True to read in images with the ear "donut", false to read in images without it
DONUT = True

# Number of ears to use (of 195) (use smaller numbers to make faster for debugging)
NUM_TO_READ = 195

# Shrink images by this factor before doing anything. Set to 1 to do no shrinking.
SHRINK_FACTOR = 8

# True to convert to black and white (required for edge detection)
BLACK_AND_WHITE = False

# Run background removal algorithm (True) or not (False)
REMOVE_BACKGROUND = False

# Use keypoints from file for registration (True) or not (False)
USE_KEYPOINT_FILE = False

# csv file where manual keypoints are located
KEYPOINT_FILE = "myKeypoints.csv"

# Do automatic template alignment (True) or not (False) (different from keypoints)
DO_TEMPLATE_ALIGN = False

# Template image. This is a binary mask that is used for image alignment.
TEMPLATE_IMAGE = "customMeanStackTemplate8x.png"

# Do edge detection (True) or not (False)
DO_EDGE_DETECTION = True

# Edge threshold range
EDGE_RANGE = [100,200]  # default [50,250]

# Edge detection dilation radius as a fraction of the image (width+height)/2
EDGE_DILATION_RADIUS = 0.01

# Set to true to do a PCA decomposition before comparison
DO_PCA = False 

# Number of eigencomponents to use in PCA decomposition
NUM_COMPONENTS = 40

# Size of thumbnails for final display (pixels)
THUMBSIZE = (63,84)

# Display images at this size (pixels). (504,672) is 1/6 raw image size
DISPLAY_SHAPE = (504,672) 

def printParameters():
    print("DATA_PATH:            ", DATA_PATH)
    print("DONUT:                ", DONUT)
    print("NUM_TO_READ:          ", NUM_TO_READ)
    print("SHRINK_FACTOR:        ", SHRINK_FACTOR)
    print("BLACK_AND_WHITE:      ", BLACK_AND_WHITE)
    print("REMOVE_BACKGROUND:    ", REMOVE_BACKGROUND)
    print("USE_KEYPOINT_FILE:    ", USE_KEYPOINT_FILE)
    print("KEYPOINT_FILE:        ", KEYPOINT_FILE)
    print("DO_TEMPLATE_ALIGN:    ", DO_TEMPLATE_ALIGN)
    print("TEMPLATE_IMAGE:       ", TEMPLATE_IMAGE)
    print("DO_EDGE_DETECTION:    ", DO_EDGE_DETECTION)
    print("EDGE_RANGE:           ", EDGE_RANGE)
    print("EDGE_DILATION_RADIUS: ", EDGE_DILATION_RADIUS)
    print("DO_PCA:               ", DO_PCA)
    print("NUM_COMPONENTS:       ", NUM_COMPONENTS)
    print("THUMBSIZE:            ", THUMBSIZE)
    print("DISPLAY_SHAPE:        ", DISPLAY_SHAPE)
