#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 2 On-the-fly data generation

Author(s):
Nitin Suresh,
ECE - UMD
"""

# Code starts here:

import numpy as np
import cv2
import pdb
# Add any python libraries here

def gen_data(DataSelect):
    rho = 32 # max corner perturbation
    psize = 128 # patch-size

    # Use DataSelect to select image from folder
    if DataSelect==0:
        imgDir = '/Train/'
        MaxImNum = 1000
    else:
        imgDir = '/Val/'
        MaxImNum = 1000

    while True:
        im_num = np.random.randint(1, MaxImNum+1)
        I_a = cv2.imread('../Data'+imgDir+str(im_num)+'.jpg',0)
        if (I_a.shape[0]!=0) and (I_a.shape[1]!=0):
            break

    # Image resized to (320,240)
    I_a = cv2.resize(I_a, (320, 240))

    # Extract top-left corner of patch randomly
    y_0 = np.random.randint(35, 75)
    x_0 = np.random.randint(35, 150)

    # Coordinates of initial patch
    C_a = np.array([[y_0,x_0],
           [y_0,x_0+psize],
           [y_0+psize,x_0+psize],
           [y_0+psize,x_0]], np.int32)

    # Extract patch
    P_a = I_a[C_a[0][0]:C_a[2][0], C_a[0][1]:C_a[1][1]]
    # cv2.imshow('P_a', P_a)

    # Introduce random perturbation
    H4Pt = np.random.randint(-rho, rho, size=(4,2), dtype=np.int32)
    # print(H4Pt)
    # Calculate homography using the original and distorted corner points
    C_b = C_a + H4Pt
    # print(C_b)

    # TODO: Display initial and distorted patch on the original image

    H_ab = cv2.getPerspectiveTransform(C_a.astype(np.float32), C_b.astype(np.float32))
    H_ba = np.linalg.inv(H_ab)

    I_b = cv2.warpPerspective(I_a, H_ba, (I_a.shape[1], I_a.shape[0]))

    # cv2.imshow('Warped image', I_b)
    # cv2.waitKey(0)

    # Extract corresponding patch from the warped image
    P_b = I_b[C_a[0][0]:C_a[2][0], C_a[0][1]:C_a[1][1]]
    # cv2.imshow('P_b', P_b)
    # cv2.waitKey(0)

    # Check if correct
    # I_check_1 = cv2.warpPerspective(I_a, H_ab, (I_a.shape[1], I_a.shape[0]))
    # cv2.imshow('I_check', I_check)
    # cv2.waitKey(0)

    # Stack the images into 1
    stack_img = np.zeros((P_a.shape[0], P_a.shape[1], 2))
    stack_img[:,:,0] = P_a
    stack_img[:,:,1] = P_b

    H4Pt = np.reshape(H4Pt, (8))
    C_a = np.reshape(C_a, (8))

    # im_data.append(stack_img.astype(np.uint8))
    # H4Pt_data.append(H4Pt)

    return stack_img, H4Pt, C_a, I_a, I_b
