#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.

# The "template" method is an attempt to implement a technique that is common 
# in the literature, namely taking the image and matching it to a known 
# template, to be used for registration. The same method is also commonly used 
# for matching pairs of ears, but that is beyond the scope of this project.

# A 'template" for our purposes is a binary mask that looks somewhat like a
# dilated edgemap of an ear photograph. To do registration, we take a set
# of templates that have known x shifts, y shifts, and rotation applied to
# them. Then we simple correlate the edgemap of an ear image with each of the
# templates.  The template which best matches the edgemap of the ear image is
# the most likely to represent the correct transformation.  We then take
# that transformation and apply its inverse to the image, resulting in an 
# image that is well aligned to the unperturbed original template.
# 
# The template class holds a single template that is a copy of the input 
# template with a known shift and rotation applied.

# External imports
import cv2
import numpy as np
import parameters as p

# Internal imports
import preprocess


# Class for holding binary mask ear outline templates
class template:
    def __init__(self,base):
        self.mask = base # holds the template image
        self.ny, self.nx, self.nc = self.mask.shape
        self.xs  = -9999
        self.ys  = -9999
        self.rot = -9999
        self.stretchx = -9999
        self.stretchy = -9999
        self.name = ""
    
    # Applies the specified warp to the unperturbed base template
    def applyWarp(self, xs, ys, rot, stretchx, stretchy):
        self.xs  = xs
        self.ys  = ys
        self.rot = rot
        self.stretchx = stretchx
        self.stretchy = stretchy
        
        self.mask = preprocess.scaleImage(self.mask, stretchx, stretchy)
                
        self.mask = np.roll(self.mask,int(self.xs),axis=1)
        self.mask = np.roll(self.mask,int(self.ys),axis=0)
        R = cv2.getRotationMatrix2D((self.nx/2, self.ny/2), rot, 1.0)
        self.mask = cv2.warpAffine(self.mask, R, (self.nx, self.ny))
        self.name = "template_"+str(self.xs)+"_"+str(self.ys)+"_"+str(self.rot)+"_"+str(stretchx)+"_"+str(stretchy)
        #cv2.imshow("ear",self.mask)
        #cv2.waitKey(1000)
        
    # Determines how well the input edgemap fits this template
    def checkMatchStrength(self,edgemap):
        # This is a simple matter of counting how many pixels in the input
        # edgemap match the template mask
        pixelmatches = (edgemap * self.mask) > 0
        matchStrength = np.sum(pixelmatches)
        return matchStrength


# Take the "base template" from an image file and apply a series of 
# transformations (warps, homographies) to it.  This gives an array of 
# templates to compare each input image to.
def makeTemplates():
    templateBase = cv2.imread(p.TEMPLATE_IMAGE)
    
    # resize mask to be the same as images. Original mask is an 8x shrink
    templateBase = cv2.resize(templateBase,
              (int(templateBase.shape[1]*8/p.SHRINK_FACTOR),
               int(templateBase.shape[0]*8/p.SHRINK_FACTOR))) 
    ny,nx,nc = templateBase.shape
    
    # These are the shift, rotation, and stretch parameters that will be 
    # represented in the template list
    xshifts   = (np.arange(10)-5)/5 * nx*0.1 #0.05
    yshifts   = (np.arange(10)-5)/5 * ny*0.1 #0.05
    rotations = (np.arange(10)-5) * 3 + 3
    stretchx  = [1]
    stretchy  = [1]
    #xshifts   = (np.arange(16)-8)/8 * nx*0.03
    #yshifts   = (np.arange(16)-8)/8 * ny*0.03 # 0.15
    #rotations = (np.arange(16)-8) * 5 + 5
    #stretchx  = (np.arange(5)-2)/2 * 0.1 + 1
    #stretchy  = (np.arange(5)-2)/2 * 0.1 + 1
    
    xshifts = xshifts.astype(int)
    yshifts = yshifts.astype(int)
    
    # Generate all the templates
    templates = []
    for strx in stretchx:
        for stry in stretchy:
            for xs in xshifts:
                for ys in yshifts:
                    for rot in rotations:
                        thisTemplate = template(templateBase) # make template
                        thisTemplate.applyWarp(xs,ys,rot,strx,stry) # apply warp
                        #thisTemplate.applyWarp(xs,ys,rot) # apply warp
                        templates.append(thisTemplate)
                        #cv2.imshow(thisTemplate.name, thisTemplate.mask)
                        #cv2.waitKey(0)
                        #cv2.destroyWindow(thisTemplate.name)
                        #cv2.imwrite(thisTemplate.name+".png", thisTemplate.mask)
    
    return templates


# This code generates a template from a stack of edge images, although
# realistically some hand-adjustment of the template in a program like MS
# paint is needed.
def makeTemplateFromMeanstack():
    meanstack = cv2.imread("meanStackDonut8x.jpg")
    
    dkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    template = cv2.dilate(meanstack, dkernel, iterations=3)
    template = cv2.erode(template, dkernel, iterations=4)
    template = template > 60
    template = template.astype(np.uint8) * 255
    template = cv2.dilate(template, dkernel, iterations=1)
    cv2.imshow("tempate", template)
    cv2.waitKey(0)
    cv2.destroyWindow("template")
    cv2.imwrite("meanStackTemplate8x.png", template)