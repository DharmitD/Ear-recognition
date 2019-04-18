#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.

# main() is the highest level function. Call this from your python environment.

# External library imports
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Internal imports
import earImage
import pca
import parameters as p
import template


# Reads every image and applies preprocessing. Images are split into two sets:
# the set with a "t" in the filename, and the set without a "t" in the
# filename.  The images in the "no t" set are then compared only with the 
# images in the "t" set.
def readImages():
    # Generate template set once ahead of time
    templates = []
    if p.DO_TEMPLATE_ALIGN:
        print("Generating alignment templates from " + p.TEMPLATE_IMAGE)
        templates = template.makeTemplates()
        
    # Loop over every image file in the directory so we can sort them in 
    # name order.
    filelist = glob.glob(p.DATA_PATH + "/*jpg*")
    sfilelist = sorted(filelist)
        
    # Loop over the images, reading them in
    firstSet = [] # all the images without a 't' in their title
    secondSet = [] # all the images with a 't' in their title
    print("Reading images from file")
    i = 0
    #for file in filelist:
    for file in sfilelist:
        # Skip images that do/don't have the "donut" device per the DONUT parameter
        itype = file.split(os.sep)[-1].split('.')[0].split('_')[-1]
        if p.DONUT: # skip non-donut images
            if not 'd' in itype:
                continue
        else: # skip donut images
            if 'd' in itype:
                continue
            
        image = earImage.earImage(file) # read and initialize the image
        
        # Save first image as the template for the remaining images
        if p.USE_KEYPOINT_FILE and i==0:
            templates = image
        
        # Preprocess the image
        image.preprocess(templates=templates)
        
        # FOR TESTING ONLY
        #cv2.imshow("image.nameString", image.rawImage)
        #cv2.waitKey(1000)
        #cv2.destroyWindow(image.nameString)
        #image.displayRawRGB(time=1000) # display image for 1s
        
        # Add image to the first set if it doesn't have a 't' in its name,
        # add to the second set if it does have a 't' in its name.
        if not 't' in image.typeStr:
            firstSet.append(image)
        else:
            secondSet.append(image)
        
        print("Successfully read image: ", image.nameString)
        
        # TO MAKE TESTING QUICKER, just read in the first NUM_TO_READ images
        # Stop after reading in the first N for testing more quickly
        if len(secondSet) >= p.NUM_TO_READ:
            break
        
        i += 1
    
    return firstSet, secondSet


# Calculates the similarity between each pair of images and places into a matrix.
def calculateSimilarityMatrix(firstSet, secondSet):
    # Pairwise image similarity measure in an NxN matrix
    similarityMatrix = np.zeros([len(firstSet),len(secondSet)])
        
    # Loop over each image, then see if it matches the rest
    for i1, image in enumerate(firstSet):
        print("Calculating similarities for image ", i1)
        for i2, image2 in enumerate(secondSet):
            score = image.compare(image2) # function that actually does the comparison
            similarityMatrix[i1,i2] = score
    
    return similarityMatrix


# Calculates accuracy by determining if the peak pixel of the similarity
# matrix in each row corresponds to the other image of the same truth ear.
def calculateAccuracy(similarityMatrix, firstSet, secondSet):
    peakId = []
    trueId = []
    for i,row in enumerate(similarityMatrix):
        #peakVal = np.amax(row)
        peakIndex = np.argmax(row)
        
        # Get the ID of the image being compared and the ID of the most-similar
        # of all the other images
        trueId.append(firstSet[i].number) # id of the image
        peakId.append(secondSet[peakIndex].number) # id of the peak match
        
    # Overall accuracy is the number correct over the total number
    isCorrect = [a == b for a,b in zip(trueId, peakId)]
    accuracy = np.mean(isCorrect)
    
    return accuracy, isCorrect, peakId
  
    
# Calculate rank of the true match ear images
def calculateRankOfTrueMatch(similarityMatrix, firstSet, secondSet):
    rankOfTruth = []
    for i,row in enumerate(similarityMatrix):
        trueId = firstSet[i].number
        compIds = []
        for j,item in enumerate(row):
            compIds.append(secondSet[j].number)
            
        sortedStuff = sorted((e,i) for i,e in zip(compIds,row))
        sortedStuff.reverse()
        
        # Rank of the true match is the index of sortedstuff where the compID,
        # i.e. the 2nd entry of sortedstuff, is equal to the trueID
        for i,sp in enumerate(sortedStuff):
            if sp[1] == trueId:
                rankOfTruth.append(i+1) # add 1 so that 1 is "perfect"
        
    return rankOfTruth


# Put a colored border onto the image (BGR color order assumed)
def giveBorder(image, color, npix=3):
    c3 = [0,0,0]
    if color == 'red':
        c3 = [0,0,255]
    elif color == 'green':
        c3 = [0,255,0]
    else:
        print("Inavlid color: ", color)
        return image
    
    # Add the colored border
    image[0:npix,:,0] = c3[0]
    image[0:npix,:,1] = c3[1]
    image[0:npix,:,2] = c3[2]
    image[:,0:npix,0] = c3[0]
    image[:,0:npix,1] = c3[1]
    image[:,0:npix,2] = c3[2]
    image[-npix:,:,0] = c3[0]
    image[-npix:,:,1] = c3[1]
    image[-npix:,:,2] = c3[2]
    image[:,-npix:,0] = c3[0]
    image[:,-npix:,1] = c3[1]
    image[:,-npix:,2] = c3[2]
    
    return image

# Displays results in a useful way
def displayResults(accuracy, isCorrect, similarityMatrix, 
                   firstSet, secondSet, rankOfTruth, peakIds):         
    print("=========================") # dividing line
    p.printParameters() # print parameters we used for clarity        
    print("=========================") # dividing line
    if p.DONUT:
        print("Performance for images WITH donut-device")
    else:
        print("Performance for images \"in the wild\"")
    
    print("ACCURACY: ", accuracy)
    
    simOfBest = []
    for row in similarityMatrix:
        simOfBest.append(np.amax(row))
    print("AVG SIMILARITY SCORE OF BEST MATCH: ", np.mean(simOfBest))
   
    # Rank of true match is a way of seeing how well the "true-match" ear
    # did in comparison to the others. If the true-match ear was correctly
    # chosen, then the rank of the true match is 1. If the true-match ear was
    # the 2nd best match among all the images, then the rank of the true match
    # is 2, etc...
    print("AVG RANK OF TRUE MATCH: ", np.mean(rankOfTruth))
    
    # Display an strip of thumbnails with each ear in the first set next
    # to the ear that the algorithm calculated as the best match.
    thumbstrip = []
    i = 0
    for image1,peakId in zip(firstSet,peakIds):
        thumb1 = cv2.resize(image1.rawImage, p.THUMBSIZE)
        for image2 in secondSet:
            if image2.number == peakId:
                thumb2 = cv2.resize(image2.rawImage, p.THUMBSIZE)
        for image2 in secondSet:
            if image2.number == image1.number:
                thumb3 = cv2.resize(image2.rawImage, p.THUMBSIZE)

        thumbtrio = np.concatenate((thumb1,thumb2,thumb3), axis=0)
        
        # Green box if correct, red box if incorrect
        if isCorrect[i]:
            thumbtrio = giveBorder(thumbtrio, 'green')
        else:
            thumbtrio = giveBorder(thumbtrio, 'red')
            
        # Tack on the thumbpair to the final display image
        if i ==0:
            thumbstrip = thumbtrio
        else:
            thumbstrip = np.concatenate((thumbstrip,thumbtrio), axis=1)
        i += 1
    plt.imshow(cv2.cvtColor(thumbstrip, cv2.COLOR_BGR2RGB), extent=[0.5,len(peakIds)+0.5,2,0])
    plt.title("First set images (top) with their best matches in 2nd set (middle)\n" + 
              "and the true match (bottom)")
    cv2.imwrite("bestMatchDisplay.png", thumbstrip)
    
    # Histogram of rank of best match
    # The more datapoints that are at or near 1, the better
    nImages = len(isCorrect)
    plt.figure() # make a new figure
    bins = np.arange(int(nImages/5)+1) * 5
    n, bins, patches = plt.hist(rankOfTruth, bins=bins, facecolor='blue', 
                                alpha=0.5)
    plt.xlabel("Rank of true match")
    plt.ylabel("Count (out of " + str(len(isCorrect)) + ")")
    plt.title("Histogram of true match rank")
    plt.show()
    plt.savefig('rankOfTrueMatchHistogram.png')


# Main function. Call this to run everything!!!
def main():
    
    # Read in all the images
    firstSet, secondSet = readImages()
    
    # Do edge detection if requested
    if p.DO_EDGE_DETECTION:
        #meanStack = np.zeros_like(firstSet[0].rawImage).astype(float)
        #N = float(len(firstSet) + len(secondSet))
        for i, (image1, image2) in enumerate(zip(firstSet,secondSet)):
            print("Running edge detection on image ", i, " of ", len(firstSet))
            image1.detectEdges()
            image2.detectEdges()
            
            # DILATE EDGES HERE???
    
            #dkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
            #im1d = cv2.dilate(image1.rawImage, dkernel, iterations=1)
            #im2d = cv2.dilate(image2.rawImage, dkernel, iterations=1)
    
            # Accumulate the mean of the images for later use
            #meanStack += (im1d + im2d) / N
    

        # Display the average edgemap. Used to generate the tempate that
        # is used for registration.
        #cv2.imwrite("meanStackDonut8x.jpg", meanStack)        
        #cv2.imshow("Mean of ears", meanStack/np.amax(meanStack))
        #cv2.waitKey(0)
        #cv2.destroyWindow("Mean of ears")
        
    
    # Generate an eigendecomposition for each image for use in similarity calculation
    if p.DO_PCA:
        print("Running PCA fit...")
        # Run PCA fitting
        skl_pca = pca.fit(firstSet)
        
        # Display eigenbasis (for debugging)
        pca.displayEigenbasis(skl_pca)
        
        # Decompose all the images in each set
        for i, (image1, image2) in enumerate(zip(firstSet,secondSet)):
            print("Decomposing image ", i, " of ", len(firstSet))
            image1.pcaDecomposition(skl_pca)
            image2.pcaDecomposition(skl_pca)
        
        
    # Calculate the pairwise similarity between images
    similarityMatrix = calculateSimilarityMatrix(firstSet, secondSet)
    
    # Calculate accuracy using the similarity matrix
    accuracy, isCorrect, peakId = \
        calculateAccuracy(similarityMatrix, firstSet, secondSet)
    
    # Calculate similarity ranking of the true match ears in the 2nd set
    rankOfTruth = calculateRankOfTrueMatch(similarityMatrix, firstSet, secondSet)
    
    # Displays results
    displayResults(accuracy, isCorrect, similarityMatrix, 
                   firstSet, secondSet, rankOfTruth, peakId)
    
    # Return useful results to the calling function
    return accuracy, similarityMatrix, isCorrect, rankOfTruth, firstSet, secondSet


# Actually execute the program so that calling this file runs everything.
accuracy, similarityMatrix, isCorrect, rankOfTruth, firstSet, secondSet = main()