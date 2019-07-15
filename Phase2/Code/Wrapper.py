#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin Suresh
ECE - UMD
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import tensorflow as tf
import cv2
import os
import sys
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *
from Misc.helpers import *


# Don't generate pyc codes
sys.dont_write_bytecode = True

def TestOperation(ImgPH, training, pics, ModelPath):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    training specifies training/testing for the batch normalization
    net_inp is the input image stack to obtain homography for
    ModelPath - Path to load trained model from
    """
    Length = 1
    ImageSize = (128,128,2)

    # Predict output with forward pass, MiniBatchSize for Test is 1
    H4Pt = HomographyModel(ImgPH, ImageSize, 1, training)

    # Setup Saver for model restoration
    Saver = tf.train.Saver()

    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        im2 = pics[0]
        pics = pics[1:]
        npics = 1

        # Loop over all the pics in the directory
        while pics:
            print('\nUsing image #'+str(npics))
            im1 = pics[0]
            pics = pics[1:]
            im1hgt,im1width,im1channels = im1.shape
            im2hgt,im2width,im2channels = im2.shape
            im1gray = cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY)
            im2gray = cv2.cvtColor(im2,cv2.COLOR_RGB2GRAY)

            # Resize to fit model
            im1_resz = cv2.resize(im1gray, (320, 240))
            im2_resz = cv2.resize(im2gray, (320, 240))

            # TODO: Take multiple patches and average the homography

            # Extract patches from the images of size 128x128
            y_0 = np.random.randint(35, 75)
            x_0 = np.random.randint(35, 150)

            psize = 128 # patch-size

            # Coordinates of initial patch
            C_a = np.array([[y_0,x_0],
                   [y_0,x_0+psize],
                   [y_0+psize,x_0+psize],
                   [y_0+psize,x_0]], np.int32)

            # Extract patch
            P_a = im1_resz[C_a[0][0]:C_a[2][0], C_a[0][1]:C_a[1][1]]
            P_b = im2_resz[C_a[0][0]:C_a[2][0], C_a[0][1]:C_a[1][1]]

            # Stack the images into 1 to get the input for the DL model
            net_inp = np.zeros((P_a.shape[0], P_a.shape[1], 2),dtype=np.float32)
            net_inp[:,:,0] = P_a
            net_inp[:,:,1] = P_b

            # Standardize the input by subtracting mean and division by std
            net_inp = (net_inp - 80.0)/80.0

            '''
            Obtain homography using DL model
            '''
            ImageSize = net_inp.shape
            # print(ImageSize)
            #
            # ImgPH = tf.placeholder(tf.float32, shape=(1, ImageSize[0], ImageSize[1], 2))
            # training = tf.placeholder(tf.bool, name='training')

            # Run the net and get the 4-pt homography as output
            net_inp = np.expand_dims(net_inp, axis=0) # To enable passing to the placeholder
            Test_FeedDict = {ImgPH: net_inp, training: False}
            calcH4Pt = sess.run(H4Pt, feed_dict=Test_FeedDict)

            # print(calcH4Pt)

            '''
            Warp and blend the resized images using the obtained 4-pt homography
            '''
            im2 = stitchFromH4pt(points1=C_a.astype(np.float32),h4pt=(calcH4Pt.reshape(4,2)).astype(np.float32),im1=cv2.resize(im1, (320, 240)),im2=cv2.resize(im2, (320, 240)),valid2=None)
            npics+=1
    return im2


def main():
    """
    Inputs:
    Folder containing images to create panorama from
    Outputs:
    Outputs the generated panorama image as my_pano.png, in the same folder

    Pipeline:
    Read a set of images for panorama processing

    Obtain homography using DL model, for pairs of images

    Warp and blend each image -> use this blended image along with next image in the sequence

    Output final image
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='../Checkpoints/11/49model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--ImDir', default='../Data/Pano/TestSet1/*.jpg', help='Directory to find images to stitch')

    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    ImDir = Args.ImDir

    """
	Read a set of images for panorama stitching
	"""
    pics = getIms(ImDir)
    ImgPH = tf.placeholder(tf.float32, shape=(1, 128, 128, 2))
    training = tf.placeholder(tf.bool, name='training')
    my_pano = TestOperation(ImgPH, training, pics, ModelPath)

    cv2.imshow('Final panorama', my_pano)
    cv2.waitKey(0)
    plt.imsave("./Output/mypano.png",my_pano)


if __name__ == '__main__':
    main()
