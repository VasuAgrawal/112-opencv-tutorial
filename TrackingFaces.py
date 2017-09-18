#!/usr/bin/env python

# These are necessary to make python2 act more like python3
from __future__ import print_function
from __future__ import division

import numpy as np
import cv2
import sys
import time
import math
import collections

try:
    import coloredlogs, logging
except ImportError:
    print("You should install coloredlogs! \"sudo pip install coloredlogs\"")
    import logging

# The FaceDetector class performs all of the detection and drawing of the mask 
# on top of the detected face. The mask image below (Mask Of Sliske), will be 
# drawn on top of all detected faces, given that the face is basically vertical.
class FaceDetector(object):

    # During initialization, all files are loaded, and constants are set which
    # affect the accuracy and speed of processing.
    def __init__(self):

        # Load in the classifiers we need for the face and eye.
        # These are files defining a haar cascade, which is a standard classical
        # technique for extracting some sort of feature (e.g. face, eyes). For
        # examples of more detectors, see the opencv source directory:
        # 
        # https://github.com/opencv/opencv/tree/master/data/haarcascades
        # 
        # For more information on how haar cascades work in general, and how to
        # do face detection with haar cascades, see this post:
        # 
        # http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
        facePath = "haarcascades/haarcascade_frontalface_default.xml"
        logging.info("Loading face classifier from %s", facePath)
        self.faceCascade = cv2.CascadeClassifier(facePath)

        eyePath = "haarcascades/haarcascade_eye.xml"
        logging.info("Loading eye classifier from %s", eyePath)
        self.eyeCascade = cv2.CascadeClassifier(eyePath)
      
        # Load in the mask we overlay on top of the face.
        # This image was shamelessly ripped from somewhere on Google.
        maskPath = "Mask_of_Sliske,_Light_detail.png"
        logging.info("Loading mask from %s", maskPath)
        self.face_mask = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)

        # Set max sizes for iamge processing. These are used when rescaling the
        # images before processing, in order to tradeoff accuracy and
        # performance. If you find the haar cascades are taking too long to run
        # on your computer, try reducing these slightly. Alternatively, if you
        # have a beefy machine, try increasing these and notice the differences.
        self.frame_max_dim = 96
        self.face_max_dim = 96

        # Use a deque as a fixed size list for tracking the eye centers and
        # lengths to add some more filtering to it. Increasing the HISTORY
        # dampens the motion of the mask more, trading off between stationary
        # head performance and moving head performance. Try changing it!
        # TODO: Improve tracking performance for non-stationary targets.
        HISTORY = 5
        self.eye_center_deq = collections.deque(maxlen=HISTORY)
        self.eye_len_deq = collections.deque(maxlen=HISTORY)


    # This poorly named function will take a face, find the eyes inside of it,
    # and use those to determine how to position and scale the mask. The mask is
    # then drawn on top of the original frame, potentially with debug
    # information.
    def findEyes(self, gray_face, face_pos):
        # Rescale the gray_face to a max size. See earlier comments about
        # resizing image, this is the same idea.
        rows, cols = gray_face.shape
        face_scale_factor = self.face_max_dim / max(rows, cols)
        gray_face_small = cv2.resize(gray_face, (0, 0),
                fx=face_scale_factor, fy=face_scale_factor)


        # Find the eyes using the eye classifier, different than the face one.
        eyes = self.eyeCascade.detectMultiScale(
                gray_face_small,
        )

        # Figure out where the eyes are in the original image (this is just
        # doing some coordinate transformations)
        x1, y1, x2, y2 = face_pos
        eye_points = []
        # Use :2 because the return values seem to be ordered by confidence, so
        # taking the highest 2 confidence values gets us the eyes most often.
        # Most people only have 2 eyes, so that works out well.
        for eye_pos in eyes[:2]:
            x, y, w, h = eye_pos
            x = int(x / face_scale_factor)
            y = int(y / face_scale_factor)
            w = int(w / face_scale_factor)
            h = int(h / face_scale_factor)

            # Add a box for the eyes onto the debug frame.
            cv2.rectangle(self.debug_frame, (x1 + x, y1 + y), 
                    (x1 + x + w, y1 + y + h), (1, 1, 255), 2)
            eye_points.append((x1 + x + w // 2, y1 + y + h // 2))

        # This is a "clean" way to check for the existence of 2 eyes, since the
        # loop won't execute if 2 eyes (1 or 0) aren't present. The zip function
        # is magic, just play with it a bit.
        for p1, p2 in zip(eye_points, eye_points[1:]):
            # Draw debug line between eyes, to show this loop executed.
            cv2.line(self.debug_frame, p1, p2, (1, 1, 255), 2)
            
            p1 = np.array(p1)
            p2 = np.array(p2)

            # Adding some filtering around the eye length
            eye_len = np.linalg.norm(p1 - p2)
            self.eye_len_deq.append(eye_len)
            eye_len = sum(self.eye_len_deq) / len(self.eye_len_deq)

            eye_center = (p1 + p2) / 2
            self.eye_center_deq.append(eye_center)
            # I'm very surprised this works with numpy arrays. Yay python!
            eye_center = sum(self.eye_center_deq) / len(self.eye_center_deq)
            
            # Calculate angle between the eyes.
            # TODO: Use this information to rotate the mask slightly.
            eye_line = p2 - p1
            eye_line_norm = eye_line / np.linalg.norm(eye_line)
            horizontal = np.array([1, 0], dtype=float)
            angle = math.acos(np.dot(eye_line_norm, horizontal))
            logging.debug("Calculated eye angle: %f deg", math.degrees(angle))

            # Scale the mask to make it fit the person wearing it in the camera
            # image. This is assuming that the length between the eyes is some
            # fixed ratio smaller than the entire head (e.g. 2.25).
            mask_scale_factor = eye_len * 2.25 / self.face_mask.shape[1]
            scaled_mask = cv2.resize(self.face_mask, (0, 0),
                    fx=mask_scale_factor, fy=mask_scale_factor)

            # Position the mask so that the eyelines will roughly line up. With
            # the current parameters, 35% of the mask is above the eyeline, 65%
            # below, and 50% on either side. In other words, centered
            # horizontally about the center of the eye line, not centered
            # vertically about the line itself.
            # 
            # Note that this add in the size (rows / cols) in order to ensure
            # there aren't weird sizing issues in the assignment later. However,
            # this doesn't do bounds checking if the face would be outside.
            # TODO: Clip mask if it would land outside image. Crashes right now.
            mask_rows, mask_cols, _ = scaled_mask.shape
            y1_mask = int(eye_center[1]) - int(mask_rows * 0.35)
            y2_mask = y1_mask + scaled_mask.shape[0]
            x1_mask = int(eye_center[0]) - int(mask_cols * 0.5)
            x2_mask = x1_mask + scaled_mask.shape[1]

            frame_roi = self.frame[y1_mask:y2_mask, x1_mask:x2_mask, :]

            # Channel by channel, assign the mask onto the original frame. This
            # is to make use of the fact that we have a PNG image for the mask
            # (with transparency).
            mask = scaled_mask[:, :, 3] != 0
            frame_roi[:, :, 0][mask] = scaled_mask[:, :, 0][mask]
            frame_roi[:, :, 1][mask] = scaled_mask[:, :, 1][mask]
            frame_roi[:, :, 2][mask] = scaled_mask[:, :, 2][mask]


    # Processing the face simply involves using the face_pos coordinates to rip
    # the face from the original gray image, and calling findEyes on that
    # smaller face image.
    def processFace(self, face_pos):
        x, y, w, h = face_pos
        sf = 1.0 / self.frame_scale_factor
        x1 = int(x * sf)
        y1 = int(y * sf)
        x2 = int((x + w) * sf)
        y2 = int((y + h) * sf)

        # First, draw the face onto the debug frame.
        cv2.rectangle(self.debug_frame, (x1, y1), (x2, y2), (1, 255, 1), 2)

        # For the face, find all of the eyes.
        gray_face = self.gray_frame[y1:y2, x1:x2]
        self.findEyes(gray_face, (x1, y1, x2, y2))


    # Draws tracking history for the eye centers. If the blue dots are clustered
    # together, we can expect good filtering performance. If they're clustered
    # poorly (i.e. in a line), then we have motion.
    def drawTrackingHistory(self):
        for p1, p2 in zip(self.eye_center_deq, list(self.eye_center_deq)[1:]):
            cv2.line(self.debug_frame, tuple(p1.astype(int)),
                    tuple(p2.astype(int)), (255, 127, 1), 1)
        
        for p in self.eye_center_deq:
            cv2.circle(self.debug_frame, tuple(p.astype(int)), 2, (255, 1, 1))

    # The function where the majority of the logic hapens. `frame` is a numpy
    # array (opencv image) where faces are to be detected. The process function
    # will detect faces, overlay the mask, and optionally overlay the debug
    # information on top of the face as well.
    def process(self, frame):
        self.frame = frame

        # The debug frame is so that debug information can be drawn above
        # everything else. Based on how the mask is calculated (!= 0, per
        # channel), we need to use values like (1, 255, 1) for the color rather
        # than just (0, 255, 0).
        self.debug_frame = np.zeros(frame.shape, dtype=frame.dtype)

        # Convert to black and white because the haar cascades don't operate on
        # RGB / BGR frames.
        self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y = self.gray_frame.shape

        # Resize the frame to be smaller. This is crucial in ensuring a
        # performant application, since the number of pixels to process reduces
        # quadratically with the length / width. The smaller the image, the
        # faster the prediction, though the lower the quality of the prediction
        # (sometimes nothing is found at all).
        self.frame_scale_factor = self.frame_max_dim / max(x, y)
        self.gray_frame_small = cv2.resize(self.gray_frame, (0, 0),
                fx=self.frame_scale_factor, fy=self.frame_scale_factor)

        # This is the line that actually performs the detection. Faces is a list
        # of tuples of the form (x, y, w, h) in image coordinates.
        faces = self.faceCascade.detectMultiScale(
                self.gray_frame_small,
        )

        for face_pos in faces:
            # face_pos = (x, y, w, h)
            self.processFace(face_pos)

        # Draw debug on top of everything. See above comments about debugger
        # frame. This will draw a box around the face, boxes around the eyes,
        # connect the eye centers, and draw center tracking dots.
        # TODO: Tie this into logger level?
        self.drawTrackingHistory()
        mask = self.debug_frame != 0
        self.frame[mask] = self.debug_frame[mask]
            

# ======================= START READING HERE ==============================
def main():
   
    # Open a camera, allowing us to read frames. Note that the frames are BGR,
    # not RGB, since that's how OpenCV stores them. The index 0 is usually the
    # integrated webcam on your laptop. If this fails, try different indices, or
    # get a webcam.
    camera = cv2.VideoCapture(0)
    detector = FaceDetector()

    # In every loop, we read from the camera, process the frame, and then
    # display the processed frame. The process function is where all of the
    # interesting logic happens.
    while True:
        # Timing code to get diagnostics on per frame performance
        start = time.time()
        _, frame = camera.read()
        end = time.time()
        logging.debug("Took %2.2f ms to read from camera", 
                round((end - start) * 1000, 2))

        start = time.time()
        detector.process(frame) # Process 
        end = time.time()
        logging.info("Took %2.2f ms to process frame", 
                round((end - start) * 1000, 2))
   
        # Standard display code, showing the resultant frame in a window titled
        # "Video". Close the window with 'q'.
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up after ourselves.
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    try:
        coloredlogs.install(level="INFO")
    except NameError:
        logging.basicConfig(level=logging.DEBUG)

    main()
