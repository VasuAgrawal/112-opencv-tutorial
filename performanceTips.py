#!/usr/bin/env python

tips = """
            General performance tips for your OpenCV Projects:

1. Python is SLOW.
    Unfortunately, as awesome as Python is, those very features that make it
    awesome mean that it's going to be too slow for our purposes. To give you
    a concrete example, you're not going to be able to iterate over a single
    image using just Python for loops in 30 ms (i.e. 30 Hz) and get any
    meaningful work done.

2. Learn Numpy, learn it well!
    Numpy is an amazing number manipulation library -- the standard, in fact,
    for any Python applications. It is all compiled C, and so, similar to how
    built in functions are orders of magnitude faster than their Python
    versions, Numpy operations will be orders of magnitude faster than the
    corresponding Python operations.

3. Forget about iteration
    One of the keys to using Numpy well is vectorization -- that is, turning
    your iterative operations into something that can be applied to the whole
    array at once. For example, instead of iterating through an entire array
    in order to try to threshold it, we simply create a binary mask using >
    and then use an assignment to assign to all values at once.

4. OpenCV is even faster than Numpy
    While Numpy is great for general purpose array manipulation, if you're
    trying to do specific manipulation tasks (i.e. thresholding, erode, etc),
    OpenCV has algorithms optimized specifically for that, and will thus be
    even faster than the Numpy implementation.

5. Make it work, and then make it work fast.
    This is just a general optimization tip, but you should always make
    sure that your code works before attempting to make it fast. Who knows,
    maybe the naiive implementation is fast enough and you can get some sleep
    instead of optimizing your code?

6. Profile your code in order to find performance bottlenecks.
    If you do need to optimize, make sure to profile your code using utilities
    like profile and time.time() to figure out which parts of your code are
    taking the longest to execute. For loops, start with the innermost and
    work your way out. If absolutely necessary, consider writing your loops
    in C, called via ctypes, Numpy's C API, Python's C API, or similar.
"""

if __name__ == "__main__":
    print tips
