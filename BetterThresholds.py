#!/usr/bin/env python

# This file contains a few different variations on thresholding, showing some
# various functionality of numpy as well as finally how to do it with openCV

import cv2
import numpy as np

# Define our constants
WHITE = 255
BLACK = 0
THRESH = 127

# Thresholding using numpy iterators
def iter_threshold(image):
    # Returns us a copy, so we can modify it
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # nditer gives us a memory-efficient iterator over the array, which we
    # can then write to with special numpy syntax
    for val in np.nditer(grey, op_flags=['readwrite']):
        val[...] = WHITE if val > THRESH else BLACK
    return grey

# Binary mask in numpy
# We can index into an array with another array (the binary mask), and
# by assigning a scalar value we assign to all values that were masked in.
def mask_threshold(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Index into grey with a binary mask, and assign binary values
    grey[grey > THRESH] = WHITE
    grey[grey <= THRESH] = BLACK
    return grey

# And finally, we demonstrate the OpenCV thresholding function, which
# is able to use some more advanced thresholds rather than a fixed constant.
def threshold(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # There are a few thresholding modes available. In this one, the output
    # goes to 0 if grey(x, y) <= THRESH, else WHITE. See docs for other options.
    _, thresholded = cv2.threshold(grey, THRESH, WHITE, cv2.THRESH_BINARY)
    return thresholded

# And finally, we demonstrate the OpenCV thresholding function, which
# is able to use some more advanced thresholds rather than a fixed constant.
# We also blur our image first in order to remove some noise.
def threshold_otsu(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    value = (31, 31)
    blurred = cv2.GaussianBlur(grey, value, 0)
    # Otsu thresholding is able to automtically determine what the threshold
    # value should be. Currently only works on 8 bit images.
    # We also use _ for the return value to simply ignore it.
    _, thresholded = cv2.threshold(blurred, 0, WHITE,
                                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresholded

# Modified to take a variety of threshold functions
def main(fn):

    window_name = "Webcam!"

    cam_index = 0
    cv2.namedWindow(window_name, cv2.CV_WINDOW_AUTOSIZE)

    cap = cv2.VideoCapture(cam_index)
    cap.open(cam_index)

    while True:

        ret, frame = cap.read()

        if frame is not None:
            # Instead of showing the original image, show the thresholded one
            cv2.imshow(window_name, fn(frame))
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # Escape key
            cv2.destroyAllWindows()
            cap.release()
            break

# Run through all of the demonstrations
if __name__ == "__main__":
    main(iter_threshold)
    main(mask_threshold)
    main(threshold)
    main(threshold_otsu)