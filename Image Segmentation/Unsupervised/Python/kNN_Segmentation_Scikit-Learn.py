#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kNN Segmentation - Scikit-Learn.

Unsupervised segmentation utilizing k-Nearest Neighbors algorithm.
Implements kNN from Scikit-Learn library

Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
"""

import cv2 as cv
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
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


def kNN(img ,K):
    '''
    k-Nearest Neighbors algorithm. kNN is utilized to segment the input frame
    into k number of clusters. Radius is set to be equal to k and use all
    available CPU's.
    
    Parameters
    ----------
    img : unit8 array
        Input image.
    K : uint
        Number of clusters to segment.
    
    Returns
    -------
    clusters : object array
        Array containing resulting clusters.
    
    '''
    # Input frame's dimensions
    rows, colms, cols = img.shape
    
    # Prepare input
    inpt = np.float32(img.reshape((-1, 3)))
    
    # kNN algorithm
    knn = NearestNeighbors(n_neighbors=K, radius=K, n_jobs=-1)
    
    # Fit input frame into kNN model
    knn.fit(inpt)
    
    # Neighbor indices
    neigh_ind = knn.kneighbors(inpt, return_distance=False)
    
    # Initialize array to store clusters
    clusters = np.zeros(K, dtype=np.object)
    
    # Iterate over clusters
    for k in range(K):
        # Get current cluster indices
        ys, xs = np.unravel_index(neigh_ind[:, k], (rows, colms), 'C')
        
        # Initialize current cluster
        pred = np.zeros_like(img, dtype=np.uint8)
        
        # Iterate over cluster indices
        for i in range(len(ys)):
            # Construct cluster
            pred[ys[i], xs[i], :] = img[ys[i], xs[i], :]
        # Store current cluster
        clusters[k] = pred
    
    # Return clusters
    return clusters

def main(K):
    '''
    
    
    Parameters
    ----------
    K : uint
        Number of clusters to segment input image.
    
    Returns
    -------
    clusters : object array
        Array containing evaluated clusters.
    
    '''
    # Read input image
    img = cv.imread(get_path('*.jpg'))
    
    # kNN model
    clusters = kNN(img, K)
    
    return clusters

if __name__ == '__main__':
    # Number of clusters to produce
    K=7
    # Segment image
    clusters = main(K)
    
