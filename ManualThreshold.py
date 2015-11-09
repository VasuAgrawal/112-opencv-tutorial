#!/usr/bin/env python

import cv2
import numpy as np

# This is going to be EXTREMELY slow, since we're using python for loops
def manual_threshold(image):
    # Define some constants
    WHITE = 255
    BLACK = 0
    THRESH = 127

    # Convert our input image to grayscale so that it's easy to threshold
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a new array of all zeros to store our thresholded image in
    # It will be the same size as our grey image
    thresholded = np.zeros(grey.shape, np.uint8)
    
    # Iterate over the grey image, and store results in thresholded
    for i in xrange(grey.shape[0]):
        for j in xrange(grey.shape[1]):
            # If we're over a certain target value, then saturate to white
            # otherwise, we're under the bar, dilute to black
            thresholded[i][j] = WHITE if grey[i][j] > THRESH else BLACK

    # Return our handiwork
    return thresholded

# We've finally put our code in a function instead!
def main():

    window_name = "Webcam!"

    cam_index = 0
    cv2.namedWindow(window_name, cv2.CV_WINDOW_AUTOSIZE)

    cap = cv2.VideoCapture(cam_index)
    cap.open(cam_index)

    while True:

        ret, frame = cap.read()

        if frame is not None:
            # Instead of showing the original image, show the thresholded one
            cv2.imshow(window_name, manual_threshold(frame))
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # Escape key
            cv2.destroyAllWindows()
            cap.release()
            break

if __name__ == "__main__":
    main()