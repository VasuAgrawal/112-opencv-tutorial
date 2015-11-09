#!/usr/bin/env python

# This code allows us to resize the input image in addition
# to loading it.

# Import opencv
import cv2

window_name = "Images"
desired_size = 500.0 # we want the max dimension to be 500

# Importantly, images are stored as BGR
# Use the following function to read images.
image = cv2.imread("LightCat.jpg")
# Error checking to make sure that our image actually loaded properly
# Might fail if we have an invalid file name (or otherwise)
if image is not None:

    # Get the size of the image, which is a numpy array
    size = image.shape
    print size # Just so that we see what format it's in
    # Notice that it's a 1000 x 1000 x 3 image, where the last
    # dimension is the 3 values, BGR, per pixel.
    
    # We now want to resize the image to fit in our window, while
    # maintaining an aspect ratio
    fx = desired_size / size[0]
    fy = desired_size / size[1]
    scale_factor = min(fx, fy)

    # Get the resized image. The (0,0) parameter is desired size, which we're
    # setting to zero to let OpenCV calculate it from the scale factors instead
    resized = cv2.resize(image, (0,0), fx = scale_factor, fy = scale_factor)

    # Display our loaded image in a window with window_name
    cv2.imshow(window_name, resized)
    # Wait for any key to be pressed
    cv2.waitKey(0)

# Clean up before we exit!
cv2.destroyAllWindows()