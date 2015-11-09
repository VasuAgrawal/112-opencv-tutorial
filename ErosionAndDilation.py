#!/usr/bin/env python

# This file is meant to demonstrate erosions and dilations in OpenCV.
# For some of the theory, please see wikipedia / etc.

import cv2
import numpy as np

WHITE = 255
BLACK = 0
THRESH = 127

# Create our erosion / dilation kernels, which are in this case
# just a 5 x 5 array of ones.
kernel = np.ones((5,5), np.uint8)

# Dilation essentially makes white / bright areas bigger, and makes
# black / dark images smaller. It is done by taking the max of the 
# kernel iterated over the entire image.
def dilate(image):
    return cv2.dilate(image, kernel)

# The opposite of dilation, erosion makes dark areas bigger, and bright
# areas smaller. This is done by taking the min of the kernel, iterated
# over the entire image.
def erode(image):
    return cv2.erode(image, kernel)

# An open operation is simply an erosion followed by a dilation. It
# is very useful in removing noise, among other things. We can also
# imagine it as being able to separate disjoint parts of an image
# connected by only small slivers.
def open(image):
    return dilate(erode(image))

# The opposite of open, a close operation is a dilation followed by
# an erosion. This is often useful for closing small holes inside
# various objects.
def close(image):
    return erode(dilate(image))

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

            # We demonstrate a couple different versions of
            # the same thresholded image.

            # We should notice that the erosion a lot of the noise that
            # was left over from the thresholding operation.
            cv2.imshow("Erode", erode(thresh))
            cv2.imshow("Dilate", dilate(thresh))
            cv2.imshow("Open", open(thresh))
            cv2.imshow("Close", close(thresh))
       
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # Escape key
            cv2.destroyAllWindows()
            cap.release()
            break

if __name__ == "__main__":
    main()