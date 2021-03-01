#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kmeans Segmentation - OpenCV & Scikit-Learn.

Unsupervised image segmentation utilizing k-Means algorithm. 
Implements k-Means from OpenCV & Scikit-Learn libraries

References: 
       [1] https://docs.opencv.org/master/d1/d5c/tutorial_py_kmeans_opencv.html   
       [2] https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html#sphx-glr-auto-examples-cluster-plot-color-quantization-py
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.cluster import KMeans
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

def kmeans_cv(frame, n_clusters):
    '''
    K-Means utilizing the already built-in function found in the
    OpenCV library. The k-means algorithm segments the input image into K
    clusters. The algorithm utilizes the Kmeans++ initialization.
    The criteria for the K-Means are defined as, max number of iterations
    set to 300, and the acceptable error rate is set to 1e-4.
    
    Parameters
    ----------
    frame : uint8 array
        Input image.
    n_clusters : uint
        Number of clusters to segment input image.
    
    Returns
    -------
    clusters : object array
        Array containing each evaluated cluster.
    
    '''
    # Input frame's dimensions
    rows, colms, cols = frame.shape
    # Define criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 300, 1e-4)
    
    # Flatten input frame
    inpt_cv = np.float32(frame.reshape((-1, 3)))
    
    # Fit current frame to the k-means algorithm
    ret,label,center = cv.kmeans(inpt_cv, n_clusters, None, criteria,
                               10, cv.KMEANS_PP_CENTERS)
    
    # Obtain labels
    center = np.uint8(center)
    
    # Initialize array to store clusters
    clusters = np.zeros(K, dtype=np.object)
    
    # Iterate over clusters
    for k in range(K):
        # Indices of current cluster
        ys, xs = np.unravel_index(np.where(label == k), (rows, colms), 'C')
        
        # Initialize custer
        pred = np.zeros_like(frame, dtype=np.uint8)
        
        # Iterate over indices
        for i in range(len(ys)):
            # Construct cluster
            pred[ys[i], xs[i], :] = frame[ys[i], xs[i], :]
        
        # Store cluster
        clusters[k] = pred
    
    # Return resulting array
    return clusters

def kmeans_sk(frame, n_clusters):
    '''
    K-Means utilizing the built-in function found in the Scikit-Learn library.
    The k-means algorithm segments the input image into K
    clusters. The algorithm utilizes the Kmeans++ initialization.
    The criteria for the K-Means are defined as, max number of iterations
    set to 300, and the acceptable error rate is set to 1e-4.
    
    Parameters
    ----------
    frame : uint8 array
        Input image.
    n_clusters : uint
        Number of clusters to segment input image.
    
    Returns
    -------
    clusters : object array
        Array containing each evaluated cluster.
    
    '''
    # Get image dimmentions
    row, colm, chns=  frame.shape
    
    # Flatten image values for the k-means algorithm
    inpt = np.reshape(frame, (row * colm, chns))
    
    # Initialize the k-means model
    kmeans = KMeans(n_clusters, init='k-means++')
    
    # Fit the input image into the model
    kmeans.fit(inpt)
    
    # Predict the closest cluster each sample in input image belongs to
    labels = kmeans.predict(inpt)
    
    # Output separated objects into image
    res_frame = np.zeros((row, colm, chns), dtype=np.uint8)
    
    # Initialize label index
    label_idx = 0
    
    # Initialize array to store clusters
    clusters = np.zeros(n_clusters, dtype=np.object)
    
    # Iterate over clusters
    for j in range(n_clusters):
    	# Iterate over pixels
        for i in range(row):
            for k in range(colm):
            	# Current pixel belongs to current cluster
                if labels[label_idx] == j:
                    # Reconstruct cluster
                    res_frame[i, k] = kmeans.cluster_centers_[labels[label_idx]]
                # Update label index
                label_idx += 1
        # Set index to zero
        label_idx = 0
        
        # Store current frame
        clusters[j] = res_frame
        
        # Set cluster to zer
        res_frame = np.zeros((row, colm, chns), dtype=np.uint8)
    
    # Return clusters
    return clusters

def main(K):
    '''
    
    
    Parameters
    ----------
    K : uint
        Number of clusters to produce.
    
    Returns
    -------
    clusters : object array
        Array containing resulting clusters.
    
    '''
    # Get input image
    path = get_path("*.jpg")
    
    # Read image
    img = cv.imread(path)
    
    # Produced clusters
    clusters = kmeans_cv(img, K)
    
    return clusters

if __name__ == '__main__':
    # Number of clusters
    K = 10
    
    # Produced clusters
    clusters = main(K)

