#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin Suresh,
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


# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
    """
    Inputs:
    BasePath - Path to images
    Outputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    """
    # Image Input Shape
    ImageSize = [32, 32, 3]
    DataPath = []
    NumImages = len(glob.glob(BasePath+'*.jpg'))
    SkipFactor = 1
    for count in range(1,NumImages+1,SkipFactor):
        DataPath.append(BasePath + str(count) + '.jpg')

    return ImageSize, DataPath

def ReadImages(ImageSize, DataPath):
    """
    Inputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """

    ImageName = DataPath

    I1 = cv2.imread(ImageName)

    if(I1 is None):
        # OpenCV returns empty list if image is not read!
        print('ERROR: Image I1 cannot be read')
        sys.exit()

    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################

    # I1S = iu.StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1, axis=0)

    return I1Combined, I1


def TestOperation(ImgPH, LabelPH, training, TestImages, TestLabels, ImageSize, ModelPath):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    Length = TestImages.shape[0]

    # Predict output with forward pass, MiniBatchSize for Test is 1
    H4Pt = HomographyModel(ImgPH, ImageSize, 1, training)

    # Setup Saver
    Saver = tf.train.Saver()

    with tf.name_scope('Loss'):
        # Calculate EPE loss for 1 image
        diff_tensor = H4Pt-LabelPH
        loss = tf.reduce_mean(tf.norm(H4Pt-LabelPH, axis=1))
        l1_loss = tf.reduce_mean(tf.abs(H4Pt-LabelPH))

    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        # OutSaveT = open(LabelsPathPred, 'w')
        EpochLoss = 0
        Epoch_l1_loss = 0
        total_time = 0

        for count in tqdm(range(Length)):
            TestImg = TestImages[count]
            TestImg = np.expand_dims(TestImg, axis=0) # To enable passing to the placeholder

            TestLabel = TestLabels[count]
            TestLabel = TestLabel.reshape(1,8)

            Test_FeedDict = {ImgPH: TestImg, LabelPH: TestLabel, training: False}
            start_time = time.time()
            Loss_Img, calcH4Pt, calcLabelPH, calcdiff, calcl1_loss = sess.run([loss, H4Pt, LabelPH, diff_tensor, l1_loss], feed_dict=Test_FeedDict)
            total_time += (time.time()-start_time)
            EpochLoss = EpochLoss + Loss_Img
            Epoch_l1_loss = Epoch_l1_loss + calcl1_loss

            # Debugging
            # print('Image Loss: '+str(Loss_Img))
            # print('\nH4Pt\n')
            # print(calcH4Pt)
            # print('\nLabelPH\n')
            # print(calcLabelPH)
            # print('\nDiff tensor\n')
            # print(calcdiff)


        EPE_Loss = EpochLoss/Length
        Epoch_l1_loss = Epoch_l1_loss/Length
        total_time = total_time/Length
        print('\nTotal testset EPE Loss: '+str(EPE_Loss))
        print('\nTotal testset L1 Loss: '+str(Epoch_l1_loss))
        print('\nAverage forward pass runtime: '+str(total_time))

            # DataPathNow = DataPath[count]
            # Img, ImgOrg = ReadImages(ImageSize, DataPathNow)
            # FeedDict = {ImgPH: Img}
            # PredT = np.argmax(sess.run(prSoftMaxS, FeedDict))
            #
            # OutSaveT.write(str(PredT)+'\n')

        # OutSaveT.close()

# def ReadLabels(LabelsPathTest, LabelsPathPred):
#     if(not (os.path.isfile(LabelsPathTest))):
#         print('ERROR: Test Labels do not exist in '+LabelsPathTest)
#         sys.exit()
#     else:
#         LabelTest = open(LabelsPathTest, 'r')
#         LabelTest = LabelTest.read()
#         LabelTest = map(float, LabelTest.split())
#
#     if(not (os.path.isfile(LabelsPathPred))):
#         print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
#         sys.exit()
#     else:
#         LabelPred = open(LabelsPathPred, 'r')
#         LabelPred = LabelPred.read()
#         LabelPred = map(float, LabelPred.split())
#
#     return LabelTest, LabelPred


def main():
    """
    Inputs:
    None
    Outputs:
    Outputs the EPE
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='../Checkpoints/11/49model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='/home/chahatdeep/Downloads/aa/CMSC733HW0/CIFAR10/Test/', help='Path to load images from, Default:BasePath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--ImageFile', default='./im_Test.npy', help='Generated test image file')
    Parser.add_argument('--LabelsFile', default='./h4_Test.npy', help='Generated test label file')
    Parser.add_argument('--ModelType', default='Sup', help='Model type: Sup or Unsup')

    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath
    ModelType = Args.ModelType

    TestImages = np.load(Args.ImageFile)
    TestLabels = np.load(Args.LabelsFile)

    TestMean = np.mean(TestImages, (0,1,2))
    print('\nImageset mean: '+str(TestMean))
    TestStd = np.std(TestImages, (0,1,2))
    print('\nImageset std: '+str(TestStd))

    unsup_mean = 52.0;
    unsup_std = 80.0

    if ModelType is 'Sup':
        # Preprocess test images
        TestImages = (TestImages - TestMean)/TestStd
    else:
        TestImages = (TestImages - unsup_mean)/unsup_std

    # Setup all needed parameters including file reading
    # ImageSize, DataPath = SetupAll(BasePath)
    # DataPath is not required since all the data is pregenerated

    ImageSize = TestImages[0].shape
    # print('ImageSize: '+str(ImageSize))

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder(tf.float32, shape=(1, ImageSize[0], ImageSize[1], 2))
    LabelPH = tf.placeholder(tf.float32, shape=(1, 8))
    training = tf.placeholder(tf.bool, name='training')
    # LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

    print('\nRunning testing\n')
    # Run the test operation
    TestOperation(ImgPH, LabelPH, training, TestImages, TestLabels, ImageSize, ModelPath)


    # Plot Confusion Matrix
    # LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    # ConfusionMatrix(LabelsTrue, LabelsPred)

if __name__ == '__main__':
    main()
