#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Median_Kernel_Filterings.py A proof of concept image kernels processing template.

This implementation can be utilized for applications more specific than
median/mean filtering. While it can be used for such filtering applications
already made, optimized functions can be found in the OpenCV module/library.

To implement specific processing prosedures, modify kernelProc function.
"""

# TODO: GUI window for image selection, GUI window for storing resulting frame
#       Implement backend function for storing resulting frame.

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

__author__ = "I. Tzanetatos"
__version__ = "1.0.0"


def readFrame():
    '''
    Function that reads a user-specified image, asserts whether image is
    BGR or grayscale/binary image & returns the appropriate measurements.
    
    Returns
    -------
    int values & multidimensional array (image)
        Returns the dimensions of the input image & the input image.
        
    '''
    # Read frame
    img = cv.imread('inpt_img.jpg')
    
    # Get frame's dimensions
    try:
        row, colm, col = img.shape
        
    # Frame is Grayscale/Binary
    except ValueError:
        row, colm = img.shape
        
        # Return image & image dimensions
        return (row, colm, img)
    except:
        sys.exit("Something went wrong, exiting.")
    else:
        # Return image & image dimensions
        return (row, colm, col, img)

def frameScale():
    # TODO: re-implement input legality, while it works behaviour not consistent.
    '''
    Function that asks user for kernel size. Will only accept legal values
    (i.e. integers).
    
    Returns
    -------
    kern_size : uint8
        Kernel size.
        
    '''
    
    # Ensure input legality
    while True:
        try:
            kern_size = input("Enter kernel size (must be odd):>> ") or '3'
            # BUG: Need to fix sanity check
            if kern_size%2 != 0 and kern_size > 1:
                break
            else:
                print("Number not odd or not greater than 1.")
                kern_size = input("Enter kernel size (must be odd):>> ") or '3'
        except TypeError:
            print("TypeError.")
            kern_size = input("Enter kernel size (must be odd):>> ") or '3'
        except ValueError:
            print("User did not enter a number.")
            kern_size = input("Enter kernel size (must be odd):>> ") or '3'
    print("Kernel size:>> ", kern_size)
    # Return kernel size
    return np.uint8(kern_size)

def kernelProc(row, colm, col, img, kern_size):
    '''
    Image processing via kernel means. Functions has a generic structure for
    any processing that requires image kernels. Potential applications include
    median kernel filtering, mean kernel filtering, median element kernel filtering.
    
    For applications such as median/mean kernel filtering, it is more efficient
    to utilize the already implemented functions found in OpenCV.
    
    Parameters
    ----------
    width : int
        Width/columns of input preprocessed frame.
    height : int
        Height/rows of input preprocessed frame.
    col : int, optional
        Number of colour channels in the preprocessed frame.
    img : uint8 or binary multidimensional array
        Input, preprocessed frame.
    
    Returns
    -------
    out_frame : uint8 or binary multidimensional array
        Postprocessed frame, via kernel means.
    
    '''
    # Initialize output array
    try:
        out_frame = np.zeros((row//kern_size, colm//kern_size, col), dtype=np.uint8)
    except NameError:
        out_frame = np.zeros((row//kern_size, colm//kern_size), dtype=np.uint8)
    
    # Width & height indices
    hdx = 0
    wdx = 0
    # Can also implement with:
        # out_frame = cv.medianBlur(img, ksize=kern_size)
    # Safeguard for grayscale/binary images
    try:
        assert col
    # Image is grayscale/binary
    except NameError:
        for i in range(1, row, kern_size):
            for k in range(1, colm, kern_size):
                # Median element kernel
                out_frame[hdx, wdx] = img[i, k]
                # Median value kernel
                # out_frame[hdx, wdx] = np.median(img[i-1:i+2, k-1:k+2])
                # Update width index
                wdx += 1
            # Update height index & reset width index
            wdx = 0
            hdx += 1
        
    # Image is BGR/RGB
    else:
        # Median image kernels
        for c in range(col):
            for i in range(1, row, kern_size):
                for k in range(1, colm, kern_size):
                    # Median element kernel
                    out_frame[hdx, wdx, c] = img[i, k, c]
                    # Median value kernel
                    # out_frame[hdx, wdx, c] = np.median(img[i-1:i+2, k-1:k+2, c])
                    # Update width index
                    wdx += 1
                # Update height index & reset width index
                wdx = 0
                hdx += 1
            # Reset indices for next colour channel iteration
            hdx = 0
            wdx = 0
    
    # Return resulting image
    return out_frame

def plotFrame(img):
    '''
    Function that asks user whether to plot the input frame. Function will
    accept only legal inputs (i.e. 'y' or 'n', case insensitive).
    
    Parameters
    ----------
    img : uint8, or binary multidimensional array
        User entered frame.
    
    Returns
    -------
    None.
    
    '''
    # User flag input
    usr_flg = input("Visualize image [Y/n]>> ").lower() or 'y'
    
    # Sanity check of user input
    while usr_flg != 'y' and usr_flg != 'n':
        print("Invalid argument entered.")
        usr_flg = input("Visualize image [Y/n]>> ").lower() or 'y'
    
    # User opted for ploting frame
    if usr_flg == 'y':
        plt.imshow(img)
        plt.show()

def saveFrame(out_frame):
    # TODO: Implement OS operations; implement GUI function (wx or QT)
    usr_flg = input("Store resulting image [Y/n]>> ").lower()
    
    # Sanity check of user input
    while usr_flg != 'y' and usr_flg != 'n':
        print("Invalid argument entered.")
        usr_flg = input("Visualize image [Y/n]>> ").lower()
    
    # User opted for ploting frame
    if usr_flg == 'y':
        # TODO: implement user-defined image encodings.
        # Ask user for output image filename
        filename = input("Enter desired filename:>> ") + str('.jpg')
        try:
            # Store resulting image
            cv.imwrite(filename, out_frame)
        except OSError:
            # TODO: Error handling for OS operations
            sys.exit("Something unexpected occured. Exiting..")
    else:
        pass

def main():
    '''
    Main function of implementation/template
    
    Returns
    -------
    None.
    
    '''
    
    try:
        width, height, col, img = readFrame()
    except NameError:
        width, height, img = readFrame()
    
    # Plot input frame
    plotFrame(img)
    
    # kernel size
    kern_size = frameScale()
    
    # Output frame
    out_frame = kernelProc(width, height, col, img, kern_size)
    
    # Plot output frame
    plotFrame(out_frame)
    
    # Save image to disk
    # saveFrame(out_frame)

if __name__ == "__main__":
    main()
