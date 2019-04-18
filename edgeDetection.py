#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.

# External imports
import cv2
import numpy as np
import parameters as p

# Just wraps the cv2 Canny edge detection routine
def cannyEdges(image):
    #edges = cv2.Canny(image, 100,200).astype(np.uint8)
    #edges = cv2.Canny(image, p.EDGE_RANGE[0],p.EDGE_RANGE[1]).astype(np.uint8)
    
    # See https://www.pyimagesearch.com/2015/04/06/
    # zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    # Apply automatic Canny edge detection using the computed median
    if False:
        sigma = 0.5
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        print("AUTO-CANNY VALS: ", lower, ", ", upper)
    if True:
        lower = p.EDGE_RANGE[0]
        upper = p.EDGE_RANGE[1]
    
    edges = cv2.Canny(image, lower, upper)
    
    
    edges = np.repeat(np.reshape(edges,[edges.shape[0], edges.shape[1], 1]), 3, axis=2)
    
    return edges
