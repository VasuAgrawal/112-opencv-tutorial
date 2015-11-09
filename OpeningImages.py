#!/usr/bin/env python

# Import opencv
import cv2

window_name = "Images"

# Importantly, images are stored as BGR
# Use the following function to read images.
image = cv2.imread("LightCat.jpg")
# Error checking to make sure that our image actually loaded properly
# Might fail if we have an invalid file name (or otherwise)
if image is not None:
    # Display our loaded image in a window with window_name
    cv2.imshow(window_name, image)
    # Wait for any key to be pressed
    cv2.waitKey(0)

# Load another image, this time in grayscale directly
image = cv2.imread("LightCat.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
if image is not None:
    cv2.imshow(window_name, image)
    cv2.waitKey(0)

# Clean up before we exit!
cv2.destroyAllWindows()