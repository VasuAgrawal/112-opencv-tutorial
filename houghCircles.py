#!/usr/bin/env python

import cv2
import numpy as np

#Read in our image, blur it and convert to grayscale
img = cv2.imread('coins.jpg',0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

#create a list of circles in the image
circles = cv2.HoughCircles(img,cv2.cv.CV_HOUGH_GRADIENT,1,50)

#show our circles on the original image
#np.around rounds each index of circles to the nearest int so we can draw it
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

#display and save our new image
cv2.imshow('detected circles',cimg)
cv2.imwrite('detectedCoins.jpg', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
