#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.

# Functions used in preprocessing imagery.

# External imports
import numpy as np
import cv2
import copy

# Internal imports
import edgeDetection

# Parameters for ORB feature detection and alignment algorithm
# (ORB algorithm is not used because it did not work on this data. 
# See explanation below)
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.05 #0.15


# Uses the already-exising keypoints from both the image and the template image
# that will be aligned to, generates a homography matrix using the keypoint
# pairs, and transforms the image using that homography.
def alignViaKeypoints(image, templateImage):
    # Find homography matrix using keypoints
    kp1 = image.keypoints.astype(np.int32)
    kp2 = templateImage.keypoints.astype(np.int32)
    h, mask = cv2.findHomography(kp1, kp2)
    #h, mask = cv2.findHomography(kp2, kp1)
    
    #cv2.imshow("image", image.rawImage)
    #cv2.imshow("template image", templateImage.rawImage)
    #cv2.waitKey(0)
    
    #imMatches = cv2.drawMatches(image.rawImage, image,keypoints,
    #                            templateImage.rawImage, templateImage.keypoints, matches, None)
    #cv2.imwrite("matches.jpg", imMatches)
     
    # Use homography
    height, width, channels = templateImage.rawImage.shape
    alignedImage = cv2.warpPerspective(image.rawImage, h, (width, height))
    alignedImage = np.array(alignedImage)
    return alignedImage, h

# Uses a keypoint detection and matching method to determine the homography
# between a pair of images, then uses that homography to align (register) them.
# This method did NOT produce reasonable results in testing.  This is primarily
# because the keypoint detection algorithm likes to pick strong edges in the
# image, and the strongest edges in the image tend to be the sides of the
# donut device. The donut device is not what we want to register to, so the
# algorithm did not work well.  The attempt to remove them below only resulted
# in very poorly warped images.
# This function is based on:
# https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
def alignViaORB(im1, im2):
 
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
       
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    
    # CUSTOM CS640: we are working with masked imagery at this point, so the 
    # ORB algorithm absolutely LOVES to pull the edges of the mask itself.
    # This is exactly what we DON'T want, so lets remove all keypoints that
    # are within a pixel on any side of a masked out (exactly zero) pixel!!!
    #keypoints1, descriptors1 = trimKeypoints(keypoints1, descriptors1, im1Gray)
    #keypoints2, descriptors2 = trimKeypoints(keypoints2, descriptors2, im2Gray)
        
    
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
       
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
     
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    print("Good matches: " + str(numGoodMatches))
    matches = matches[:numGoodMatches]
     
    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)
       
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
     
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
       
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
     
    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    im1Reg = np.array(im1Reg)
    return im1Reg, h
    

# CUSTOM CS640: we are working with masked imagery at this point, so the 
# ORB algorithm absolutely LOVES to pull the edges of the mask itself.
# This is exactly what we DON'T want, so lets remove all keypoints that
# are within a pixel on any side of a masked out (exactly zero) pixel!!!
def trimKeypoints(keypoints, descriptors, image):
    newkeypoints = []
    newdescriptors = []
    neighborOffsets = [(-1,-1), (-1,0), (-1,1), \
                       ( 0,-1), ( 0,0), ( 0,1), \
                       ( 1,-1), ( 1,0), ( 1,1)]
    
    for keypoint, descriptor in zip(keypoints,descriptors):
        y,x = keypoint.pt
        x = int(x)
        y = int(y)
        neighborVals = [image[x+offset[0], y+offset[1]] for offset in neighborOffsets]
        
        
        # Keep this keypoint if it doesn't neighbor any exactly-zero (i.e. 
        # likely masked) pixels
        if not 0 in neighborVals:
            newkeypoints.append(keypoint)
            newdescriptors.append(descriptor)
    
    return newkeypoints, np.array(newdescriptors)


# Scale the input image by scalex and scaley. 
# scalex is a value near 1, where 1 represents no scaling, 0.9 would represent
# a 10% squeeze, and 1.1 would represent a 10% stretch.
# scaley is the same, just applied in the vertical direction.
def scaleImage(image, scalex, scaley):
    ny,nx,nc = image.shape
    # Crop and expand
    if scaley > 1.0:
        cropped = image[int(ny*(scaley-1)/2):int(ny-ny*(scaley-1)),:]
        image = cv2.resize(cropped, (nx, ny))
    if scalex > 1.0:
        cropped = image[:,int(nx*(scalex-1)/2):int(nx-nx*(scalex-1)),:]
        image = cv2.resize(cropped, (nx, ny))
        
    # Contract and pad
    if scaley < 1.0:
        newy = int(scaley*ny)
        if newy % 2 == 1:
            newy += 1
        contracted = cv2.resize(image, (nx,newy))
        padded = np.zeros_like(image)
        yextra = ny-contracted.shape[0]
        if yextra%2 == 1:
            yextra -= 1 # force even
        padded[int(yextra/2):-int(yextra/2),:,:] = contracted
        image = padded
    if scalex < 1.0:
        newx = int(scalex*nx)
        if newx % 2 == 1:
            newx += 1
        contracted = cv2.resize(image, (newx,ny))
        padded = np.zeros_like(image)
        xextra = nx-contracted.shape[1]
        if xextra%2 == 1:
            xextra -= 1 # force even
        padded[:,int(xextra/2):-int(xextra/2),:] = contracted
        image = padded

    return image

# Main function for template alignment routine. Detects the edges of the input
# image, then compares that edgemap to a preconstructed template that has had
# known shifts and rotations applied to it. The template that best matches the
# edgemap represents the best estimate of the correct transformation to put
# the ear in a consistent place and with a consistent rotation.
# Once we find the best transformation, we simply apply its inverse to the 
# original image, giving us an image that is nicely aligned to the untransformed
# original template.
def alignViaTemplate(image, templates):    
    ny,nx,nc = image.shape
    
    # Calculate image edges to use in template matching
    edgemap = edgeDetection.cannyEdges(image)   
    
    # Loop through each template and calculate a match score using this image's
    # edgemap compared with the template.
    templateScores = []
    for template in templates:
        score = template.checkMatchStrength(edgemap)
        templateScores.append(score)
        
    # Find template with best score
    bestTemplateIdx = np.argmax(templateScores)
    bestTemplate = templates[bestTemplateIdx]
    
    # Apply the best template's transformation IN REVERSE!!!
    print("Alignment params: xs=" + str(int(bestTemplate.xs)) +
          ", ys=" + str(int(bestTemplate.ys)) + ", rot=", str(bestTemplate.rot))
    image = scaleImage(image,2.0-bestTemplate.stretchx, 2.0-bestTemplate.stretchy)
    image = np.roll(image,-int(bestTemplate.xs),axis=1) #x shift
    image = np.roll(image,-int(bestTemplate.ys),axis=0) #y shift
    R = cv2.getRotationMatrix2D((nx/2, ny/2), -bestTemplate.rot, 1.0)
    image = cv2.warpAffine(image, R, (nx, ny)) # rotation
    
    # Display the template over the (now aligned) ear image to check for 
    # goodness of fit
    #baseTemplate = []
    #for t in templates:
    #    if t.xs == 0 and t.ys == 0 and t.rot == 0:
    #        baseTemplate = t
    #edgeghost = copy.deepcopy(image)
    #edgeghost[baseTemplate.mask > 0] = 255
    #cv2.imshow("", edgeghost)
    #cv2.waitKey(0)
    #cv2.destroyWindow("")
    
    return image, R

# Removes background that is not part of the ear via removal of the donut
# device and keeping pixels that are consistent with the color of human skin.
def removeBackground(image):
    
    # Remove donut if it is present. This did not work well in testing and
    # so we are not using it.
    #image = removeDonut(image)
    #cv2.imshow("DonutGone", image)
    #cv2.waitKey(0)
    #cv2.destroyWindow("DonutGone")
    
    # Match skin color
    skinMask = skinDetect(image)
    skinMask = cleanMask(skinMask)
    #cv2.imshow("SkinMask", skinMask*255)
    #cv2.waitKey(0)
    #cv2.destroyWindow("SkinMask")
    
    # Apply the mask
    image = cv2.bitwise_and(image,image,mask = skinMask)
    #cv2.imshow("SkinDetected", image)
    #cv2.waitKey(0)
    #cv2.destroyWindow("SkinDetected")
    
    return image


# skinDetect function adapted from CS640 lab held on March 29, 2019
# Function that detects whether a pixel belongs to the skin based on RGB values
# src - the source color image
# dst - the destination grayscale image where skin pixels are colored white and the rest are colored black
def skinDetect(src):
    # Surveys of skin color modeling and detection techniques:
    # 1. Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
    # 2. Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.
                
    # Fast array-based way
    dst = np.zeros((src.shape[0], src.shape[1], 1), dtype = "uint8")
    b = src[:,:,0]
    g = src[:,:,1]
    r = src[:,:,2]
    dst = (r>95) * (g>40) * (b>20) * (r>g) * (r>b) * (abs(r-g) > 15) * \
        ((np.amax(src,axis=2)-np.amin(src,axis=2)) > 15)
        
                
    return dst.astype(np.uint8)


# Cleans up the input mask. This is a multipart process:
# 1. Force the outer edges of the image to be masked out since the ears are
#    always at the center.
# 2. Run a dilation followed by an erosion to the skin mask ("opening"). This
#    connects areas of skin that are not quite touching, but should be. 
#    Effectively removes noise in the skin masking process.
def cleanMask(mask):
    ny,nx = mask.shape
    x = np.arange(nx)-nx/2
    y = np.arange(ny)-ny/2
    x = np.repeat(x[:,np.newaxis],ny,axis=1).T
    y = np.repeat(y[:,np.newaxis],nx,axis=1)
    distFromCenter2 = x*x + y*y
    
    # Remove the inner piece of the mask - this usually forces the ear canal
    # entrance to remain unmasked. Did not work well in testing, not used.
    #innerRadiusFrac = 0.5
    #centerMask = distFromCenter2 < ((nx+ny)/4*innerRadiusFrac)**2
    #mask = ((mask + centerMask) > 0).astype(np.uint8)
    
    # Force the outer piece of the mask - the ear is never near the edges
    outerRadiusFrac = 0.9
    outerMask = distFromCenter2 < ((nx+ny)/4*outerRadiusFrac)**2
    outerMask = outerMask.astype(np.uint8)
    mask = cv2.bitwise_and(mask,mask,mask=outerMask)
    
    # Open the mask (erode+dilate)
    kernelsize = (int)(0.1*(nx+ny)/2)
    dkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelsize,kernelsize))
    mask = cv2.dilate(mask, dkernel, iterations=1)
    mask = cv2.erode(mask, dkernel, iterations=1)
    
    
    return mask

# Removes the "donut" around the ear that is present in some of the image
# by color and/or position. Did not work well in testing, not using it.
def removeDonut(image):
    # Donut color is around (B,G,R) = (230,230,230)
    donutMinBGR = [210,210,210]
    b = image[:,:,0].astype(float)
    g = image[:,:,1].astype(float)
    r = image[:,:,2].astype(float)
    #donutMask = (b > donutMinBGR[0]) * (g > donutMinBGR[1]) * (r > donutMinBGR[2]) * \
    donutMask = (abs(b-r) < 20) * abs((b-g) < 20) * (abs(g-r) < 20)
    
    donutMask = np.invert(donutMask)
    donutMask = donutMask.astype(np.uint8)
    
    
    image = cv2.bitwise_and(image,image,mask = donutMask)
    
    return image