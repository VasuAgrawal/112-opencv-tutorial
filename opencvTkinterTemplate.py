import time
import sys

# Tkinter selector
if sys.version_info[0] < 3:
    from Tkinter import *
    import Tkinter as tk
else:
    from tkinter import *
    import tkinter as tk

import numpy as np
import cv2
from PIL import Image, ImageTk

def opencvToTk(frame):
    """Convert an opencv image to a tkinter image, to display in canvas."""
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    tk_image = ImageTk.PhotoImage(image=pil_img)
    return tk_image


def mousePressed(event, data):
    pass


def keyPressed(event, data):
    if event.keysym == "q":
        data.root.destroy()
    pass


def timerFired(data):
    pass


def cameraFired(data):
    """Called whenever new camera frames are available.

    Camera frame is available in data.frame. You could, for example, blur the
    image, and then store that back in data. Then, in drawCamera, draw the
    blurred frame (or choose not to).
    """

    # For example, you can blur the image.
    data.frame = cv2.GaussianBlur(data.frame, (11, 11), 0)
   

def drawCamera(canvas, data):
    data.tk_image = opencvToTk(data.frame)
    canvas.create_image(data.width / 2, data.height / 2, image=data.tk_image)


def redrawAll(canvas, data):
    drawCamera(canvas, data)


def run(width=300, height=300):

    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.camera_index = 0

    data.timer_delay = 100 # ms
    data.redraw_delay = 50 # ms
    
    # Initialize the webcams
    camera = cv2.VideoCapture(data.camera_index)
    data.camera = camera

    # Make tkinter window and canvas
    data.root = Tk()
    canvas = Canvas(data.root, width=data.width, height=data.height)
    canvas.pack()

    # Basic bindings. Note that only timer events will redraw.
    data.root.bind("<Button-1>", lambda event: mousePressed(event, data))
    data.root.bind("<Key>", lambda event: keyPressed(event, data))

    # Timer fired needs a wrapper. This is for periodic events.
    def timerFiredWrapper(data):
        # Ensuring that the code runs at roughly the right periodicity
        start = time.time()
        timerFired(data)
        end = time.time()
        diff_ms = (end - start) * 1000
        delay = int(max(data.timer_delay - diff_ms, 0))
        data.root.after(delay, lambda: timerFiredWrapper(data))

    # Wait a timer delay before beginning, to allow everything else to
    # initialize first.
    data.root.after(data.timer_delay, 
        lambda: timerFiredWrapper(data))

    def redrawAllWrapper(canvas, data):
        start = time.time()

        # Get the camera frame and get it processed.
        _, data.frame = data.camera.read()
        cameraFired(data)

        # Redrawing code
        canvas.delete(ALL)
        redrawAll(canvas, data)

        # Calculate delay accordingly
        end = time.time()
        diff_ms = (end - start) * 1000

        # Have at least a 5ms delay between redraw. Ideally higher is better.
        delay = int(max(data.redraw_delay - diff_ms, 5))

        data.root.after(delay, lambda: redrawAllWrapper(canvas, data))

    # Start drawing immediately
    data.root.after(0, lambda: redrawAllWrapper(canvas, data))

    # Loop tkinter
    data.root.mainloop()

    # Once the loop is done, release the camera.
    print("Releasing camera!")
    data.camera.release()

if __name__ == "__main__":
    run(800, 800)
