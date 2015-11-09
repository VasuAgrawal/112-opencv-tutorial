#!/usr/bin/env python

import cv2

window_name = "Webcam!"

cam_index = 0 # Default camera is at index 0.

# Create a window to display to
cv2.namedWindow(window_name, cv2.CV_WINDOW_AUTOSIZE)

cap = cv2.VideoCapture(cam_index) # Video capture object
cap.open(cam_index) # Enable the camera

# Loop indefinitely
while True:

    # Read from the camera, getting the image and some return value
    ret, frame = cap.read()

    # If frame is valid, display the image to our window
    if frame is not None:
        cv2.imshow(window_name, frame)
    
    # wait for some key with a small timeout.
    # We need the & 0xFF on 64bit systems to strip just the last 8 bits.
    k = cv2.waitKey(1) & 0xFF

    # If we hit the escape key, destroy all windows and release the capture
    # object. If we don't release cleanly, we might still have a lock and
    # no one else could use it, which is bad.
    if k == 27: # Escape key
        cv2.destroyAllWindows()
        cap.release()
        break