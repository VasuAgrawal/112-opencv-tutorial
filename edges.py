import cv2
import numpy as np

window_name = "Webcam!"
cam_index = 1 #my computer's camera is index 1, usually it's 0
cv2.namedWindow(window_name, cv2.CV_WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(cam_index)
cap.open(cam_index)

findVertEdges = False
findHorzEdges = False
findAllEdges = False
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.blur(gray, (3,3))
    if frame is not None:
        if findVertEdges:
            frame = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
        if findHorzEdges:
            frame = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
        if findAllEdges:
            frame = cv2.Canny(frame, 100, 200) 
        cv2.imshow(window_name, frame)
    k = cv2.waitKey(10) & 0xFF
    if k == 27: #ESC key quits the program
        cv2.destroyAllWindows()
        cap.release()
        break
    elif k == ord('v'):
        findVertEdges = not findVertEdges
    elif k == ord('h'):
        findHorzEdges = not findHorzEdges
    elif k == ord('a'):
        findAllEdges = not findAllEdges
