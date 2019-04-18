#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.

# This file contains all PCA eigendecomposition functions.  It is essentially
# a wrapper for the sklearn library's PCA capabilities.

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import copy
import numpy as np
import cv2

import parameters as p

# Calculates the eigenears based on the input stack of ear images.
def fit(images):
    # Set up the input data vector (images) in a sklearnPCA-friendly format
    baseImages = []
    for image in images:
        baseImages.append(copy.copy(image.rawImage.ravel()))
        
        
    scaler = StandardScaler()
    scaler.fit(baseImages)
    print("MEAN BEFORE: ", np.mean(baseImages))
    baseImages = scaler.transform(baseImages)
    print("MEAN AFTER: ", np.mean(baseImages))
    
    pca = PCA(n_components=p.NUM_COMPONENTS, whiten=True).fit(baseImages)
    
    explained_variance = pca.explained_variance_ratio_ 
    print("EXPLAINED VARIANCE RATIO: ", explained_variance)
        
    
    pca.shape = images[0].rawImage.shape #tack on a shape parameter for later use
    return pca


# Decompose the input image per the already-fit pca basis images (with pca.fit())
def decompose(image, pca):
    unraveled = image.ravel().reshape(1,-1)
    decomposedImage = np.squeeze(pca.transform(unraveled))
    return decomposedImage


# Gets the eigenears for visual debugging
def getEigenbasis(pca):
    eigenbasis = []
    for eigenimage in pca.components_:
        eigenbasis.append(np.reshape(eigenimage, pca.shape))
        
    return eigenbasis
    
# Display the eigenears for visual debugging
def displayEigenbasis(pca):
    eigenbasis = getEigenbasis(pca)
    displayImage = []
    for i,eigenear in enumerate(eigenbasis):
        if i ==0:
            displayImage = eigenear
        else:
            displayImage = np.concatenate((displayImage,eigenear), axis=1)
            
    #plt.imshow(cv2.cvtColor(displayImage, cv2.COLOR_BGR2RGB))
    displayImage -= np.amin(displayImage)
    displayImage /= np.amax(displayImage)
    displayImage *= 255
    displayImage= displayImage.astype(np.uint8)
    cv2.imwrite("Top_" + str(len(eigenbasis)) + "_Eigenears" + ".jpg", displayImage)
    #cv2.imshow("Top_" + str(len(eigenbasis)) + "_Eigenears", displayImage)
    #cv2.waitKey(5000)
    #cv2.destroyWindow("Top- " + str(len(eigenbasis)) + " Eigenears")
