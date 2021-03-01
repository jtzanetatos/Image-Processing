#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GrabCut Segmentation - OpenCV

Unsupervised image segmentation utilizing GrabCut algorithm.
Segmentation is performed utilizing a bounding box rectangle.
In this example the rectangle covers the entire image.

Reference:
    https://docs.opencv.org/master/d8/d83/tutorial_py_grabcut.html
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import wx

def get_path(wildcard):
    wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path


def main():
    '''
    
    
    Returns
    -------
    img : unit8 array
        Segmented image.
    
    '''
    # Get image path
    path = get_path("*.jpg")
    
    # Read image
    img = cv.imread(path)
    
    # Initialize mask
    mask = np.zeros(img.shape[:2],np.uint8)
    
    # Background & foreground models initialization
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    # Bounding box - Entire image
    rect = (0,0,img.shape[1]-1 ,img.shape[0]-1)
    
    # GrabCut algorithm
    cv.grabCut(img,mask,rect,bgdModel,fgdModel,15,cv.GC_INIT_WITH_RECT)
    
    # Final mask forming
    mask2 = np.where((mask==2)|(mask==0),1,0).astype('uint8')
    
    # Apply mask to input frame
    img = img*mask2[:,:,np.newaxis]
    
    # Return segmented image
    return img

if __name__ == '__main__':
    img = main()
    plt.imshow(img),plt.colorbar(),plt.show()
