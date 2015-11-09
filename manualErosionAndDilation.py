#!/usr/bin/env python

# This file is meant to demonstrate erosions and dilations in OpenCV.
# For some of the theory, please see wikipedia / etc.

import cv2
import numpy as np

WHITE = 255
BLACK = 0
THRESH = 127

ksize = 5

# Dilation essentially makes white / bright areas bigger, and makes
# black / dark images smaller. It is done by taking the max of the 
# kernel iterated over the entire image.
# Only works for grayscale images
# This function is SLOW
def manual_dilate(image):
    dilated = np.zeros(image.shape, np.uint8)
    i,j = 0,0
    offset = int(ksize / 2)
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            a = image.take(range(i-offset, i+offset + 1), mode="wrap", axis=0)
            b = a.take(range(j-offset, j+offset + 1), mode="wrap", axis=1)
            dilated[i][j] = np.amax(b)
    return dilated

# The opposite of dilation, erosion makes dark areas bigger, and bright
# areas smaller. This is done by taking the min of the kernel, iterated
# over the entire image.
# Only works for grayscale images
# This function is SLOW
def manual_erode(image):
    eroded = np.zeros(image.shape, np.uint8)
    i,j = 0,0
    offset = int(ksize / 2)
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            a = image.take(range(i-offset, i+offset + 1), mode="wrap", axis=0)
            b = a.take(range(j-offset, j+offset + 1), mode="wrap", axis=1)
            eroded[i][j] = np.amin(b)
    return eroded

# Our simple threshold function from before.
def threshold(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(grey, THRESH, WHITE, cv2.THRESH_BINARY)
    return thresholded

def main():

    window_name = "Webcam!"

    cam_index = 0
    cv2.namedWindow(window_name, cv2.CV_WINDOW_AUTOSIZE)

    cap = cv2.VideoCapture(cam_index)
    cap.open(cam_index)

    while True:

        ret, frame = cap.read()

        if frame is not None:
            # First we do a threshold on our image
            thresh = threshold(frame)
            cv2.imshow(window_name, thresh)
            cv2.imshow("Manual Dilate", manual_dilate(thresh))
            cv2.imshow("Manual Erode", manual_erode(thresh))
       
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # Escape key
            cv2.destroyAllWindows()
            cap.release()
            break

if __name__ == "__main__":
    main()