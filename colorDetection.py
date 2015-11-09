#!/usr/bin/env python

import cv2
import numpy as np

window_name = "Webcam!"
cam_index = 1 #my computer's camera is index 1, usually it's 0
cv2.namedWindow(window_name, cv2.CV_WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(cam_index)
cap.open(cam_index)

#initialize the range of colors you want to track
#values will be in HSV
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

while True:
    ret, frame = cap.read()
    if frame is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convert to HSV
        mask = cv2.inRange(hsv, lower_blue, upper_blue) #find pixels in range
        frame = cv2.bitwise_and(frame, frame, mask=mask) #zero out pixels not in range
        cv2.imshow(window_name, frame)
    k = cv2.waitKey(10) & 0xFF
    if k == 27: #ESC key quits the program
        cv2.destroyAllWindows()
        cap.release()
        break
