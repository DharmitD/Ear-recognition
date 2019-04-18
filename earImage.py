#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.

# Class to hold an ear image and some pertinent information about it. Also
# includes some functions to do simple operations.

# External library imports
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# Internal imports
import compare_sumsqdiff
import preprocess
import pca
import parameters as p
import edgeDetection
import template

# Image type enumeration
FIRST = 0  # no donut, first image (no extension)
SECOND = 1 # no donut, 2nd image ('t' extension)
DONUT1 = 2 # donut, first image ('d' extension)
DONUT2 = 3 # donut, 2nd image ('dt' extension)


# Class for holding an ear image
class earImage:
    def __init__(self, file):
        # Identifying information
        self.nameString = file.split(os.sep)[-1].split('.')[0]
        self.number = int(self.nameString.split('_')[0])
        self.typeStr = self.nameString.split('_')[-1]
        if self.typeStr == '':
            self.type = FIRST
        if self.typeStr == 't':
            self.type = SECOND
        if self.typeStr == 'd':
            self.type = DONUT1
        if self.typeStr == 'dt':
            self.type = DONUT2
        
        # Read the image. "rawImage" is a bit of a misnomer, as we use this
        # memory to hold the image after various transformations have been
        # done to it to save memory.
        self.rawImage = cv2.imread(file)
        
        # Shrink if desired. The shrinking is done for
        # two reasons:
        # 1. As a noise removal technique
        # 2. To speed up processing and save memory
        if not p.SHRINK_FACTOR == 1:
            self.rawImage = cv2.resize(self.rawImage, 
                (int(self.rawImage.shape[1]/p.SHRINK_FACTOR),
                int(self.rawImage.shape[0]/p.SHRINK_FACTOR)))
            
        self.ny, self.nx, self.ncolors = self.rawImage.shape
        
        # Read keypoints from file if desired
        if p.USE_KEYPOINT_FILE:
            self.readKeypoints()
            
                     
    # Displays the image using cv2 methods for a specified duration (time)
    # Time is in milliseconds.
    def displayRawRGB(self, shape=p.DISPLAY_SHAPE, time=0):
        #plt.imshow(self.rgbImage)
        forDisplay = cv2.resize(self.rawImage, shape)
        cv2.imshow(self.nameString, forDisplay)
        cv2.waitKey(time)
        cv2.destroyWindow(self.nameString)
        
    
    # Run alignment algorithm on the ear image.
    def align(self, templates):
        if p.DO_TEMPLATE_ALIGN:
            print("Doing template alignment...")
            self.rawImage, h = preprocess.alignViaTemplate(self.rawImage, templates)
        if p.USE_KEYPOINT_FILE:
            # Note templates here holds the earImage that we will align to
            print("Doing keypoint alignment...")
            #self.rawImage, h = preprocess.alignViaORB(self.rawImage, templates.rawImage)
            self.rawImage, h = preprocess.alignViaKeypoints(self, templates)
    
    # Remove (zero out) background
    def removeBackground(self):
        self.rawImage = preprocess.removeBackground(self.rawImage)
    
    # Run suite of preprocessing algorithms on the ear image
    def preprocess(self, templates=[]):
        # Run background removal algorithm
        if p.REMOVE_BACKGROUND:
            self.removeBackground()
        
        # Run template alignment algorithm
        if p.DO_TEMPLATE_ALIGN:
            self.align(templates)
            
        # Run pre-existing keypoint alignment algorithm
        if p.USE_KEYPOINT_FILE:
            self.align(templates)
        
        #cv2.imshow("aligned", self.rawImage)
        #cv2.waitKey(0)
        #cv2.destroyWindow("aligned")
        
        # Make image black and white if desired
        if p.BLACK_AND_WHITE:
            self.rawImage = np.mean(self.rawImage, axis=2).astype(np.uint8)
            self.rawImage = np.repeat(np.reshape(self.rawImage,
                [self.rawImage.shape[0],self.rawImage.shape[1],1]), 3, axis=2)
            
        self.ny, self.nx, self.ncolors = self.rawImage.shape
        
        #cv2.imshow("preprocessed", self.rawImage)
        #cv2.waitKey(0)
        #cv2.destroyWindow("preprocessed")
        
        # Write preprocessed image to file for later analysis
        cv2.imwrite("preprocessed"+os.sep+self.nameString + ".jpg", self.rawImage)       
        
        
    # Decompose iamge into eigencomponents using already-fit PCA object
    def pcaDecomposition(self, skl_pca):
        self.eigenweights = pca.decompose(self.rawImage, skl_pca)
        
        
    # Run edge detection alogrithm (Canny)
    def detectEdges(self):
        self.rawImage = edgeDetection.cannyEdges(self.rawImage)
        
        # Dilate the edgemap to allow for small perturbations
        if p.EDGE_DILATION_RADIUS > 0:
            kernelsize = (int)(p.EDGE_DILATION_RADIUS*(self.nx+self.ny)/2)
            dkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelsize,kernelsize))
            self.rawImage = cv2.dilate(self.rawImage, dkernel, iterations=1)
        #cv2.imshow("edges", self.rawImage)
        #cv2.waitKey(0)
        #cv2.destroyWindow("edges")
    
    # Compare this image to another image, get a score for it
    def compare(self, other):
        # This function runs an algorithm to compare the current ear image to
        # another ear image (other).
        
        # Call the external image comparison algorithm.
        # These functions are the real meat of the project.
        # ONLY ONE should be used, the rest should be commented out.
        
        if not p.DO_PCA:
            # Option 1: simple pixel-by-pixel sum squared difference comparison
            score = compare_sumsqdiff.compare(self.rawImage, other.rawImage)
    
        else:        
            # Option 2: PCA/SVD decomosition comparison
            score = compare_sumsqdiff.compare(self.eigenweights, other.eigenweights)
        
        return score

    # Read keypoints from keypoints file. Used for homographic registration
    def readKeypoints(self):
        keyptDict = pd.read_csv(p.KEYPOINT_FILE, index_col=0, header=None) # read the csv keypoint file
        #keyptDict.set_index(keyptDict.columns[0])
        keypts = keyptDict.loc[self.nameString]
        self.keypoints = []
        for pt in keypts:
            self.keypoints.append(pt)
            
        # Don't use the first one
        self.keypoints = self.keypoints[2:]
            
        self.keypoints = np.reshape(self.keypoints, (-1,2))
        
        
        # Keypoints are in image fraction (x,y), need to multiply by
        # our image size to use them. Also swap dimensions
        for pt in self.keypoints:
            pt[0] *= self.nx
            pt[1] *= self.ny
            
