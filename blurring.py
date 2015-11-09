import cv2

#set up the webcam
window_name = "Webcam!"
cam_index = 1 #my computer's camera is index 1, usually it's 0
cv2.namedWindow(window_name, cv2.CV_WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(cam_index)
cap.open(cam_index)

inBlurMode = False #allows us to turn blur on and off
while True:
    ret, frame = cap.read()
    if frame is not None:
        if inBlurMode:
            frame = cv2.blur(frame, (10,10)) #blur the current frame
        cv2.imshow(window_name, frame)
    k = cv2.waitKey(10) & 0xFF
    if k == 27: #ESC key quits the program
        cv2.destroyAllWindows()
        cap.release()
        break
    if k == ord('b'): #turns blurring on and off
        inBlurMode = not inBlurMode
