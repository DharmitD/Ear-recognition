#!/usr/bin/python
# Dharmit Dalvi and James Dunn, Spring 2019, Boston University
# Code written for CS 640 (Artificial Intelligence) project.
import numpy as np
import cv2
import parameters as p

# Super basic pixel-by-pixel sum-squared difference.
def compare(image1, image2):
    
    # Disallow any pixels that are exactly zero in either image from
    # contributing to the score. Don't do this if we are comparing PCA
    # vectors because they do not have zero pixels.
    if not p.DO_PCA:
        zeropixels1 = np.sum(image1,axis=2) == 0
        zeropixels2 = np.sum(image2,axis=2) == 0
        zeropixels = zeropixels1*zeropixels2
        nonzeropixels = zeropixels == False
        zeropixels = zeropixels.astype(np.uint8)
        #image1c = cv2.bitwise_and(image1, image1, mask=nonzeropixels)
        #image2c = cv2.bitwise_and(image2, image2, mask=nonzeropixels)
        
        # Sum squared difference, scaled to be between zero and one
        score = 1.0 / (1.0 + np.mean( \
             (image1[nonzeropixels]/np.mean(image1[nonzeropixels]) - \
             image2[nonzeropixels]/np.mean(image2[nonzeropixels]))**2))
    else:
        
        # Sum squared difference, scaled to be between zero and one
        score = 1.0 / (1.0 + np.mean( \
            (image1/np.mean(image1) - \
             image2/np.mean(image2))**2))
    
    #cv2.imshow("nozeros", image1c)
    #cv2.waitKey(0)
    #cv2.destroyWindow("nozeros")
    
    
    return score