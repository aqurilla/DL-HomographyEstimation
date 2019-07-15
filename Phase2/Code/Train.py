#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Nitin Suresh,
ECE - UMD
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import tensorflow as tf
import cv2
import sys
import os
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *

# Don't generate pyc codes
sys.dont_write_bytecode = True


def GenerateBatch(BasePath, TrainImages, TrainLabels, ImageSize, MiniBatchSize):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels
    """
    I1Batch = []
    LabelBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, TrainImages.shape[0]-1)

        # RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '.jpg'
        RandImage = TrainImages[RandIdx]

        ImageNum += 1

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        I1 = np.float32(RandImage)
        Label = TrainLabels[RandIdx]

        # Append images and labels
        I1Batch.append(I1)
        LabelBatch.append(Label)

    return I1Batch, LabelBatch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)


def TrainOperation(ImgPH, LabelPH, training,  TrainImages, TrainLabels, ValImages, ValLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    LabelPH is the label placeholder
    TrainImages - Training images file
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
    ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass

    # Convert training to placeholder
    # training = True

    H4Pt = HomographyModel(ImgPH, ImageSize, MiniBatchSize, training)

    with tf.name_scope('Loss'):
        ###############################################
        # Fill your loss function of choice here!
        ###############################################
        diff_tensor = H4Pt - LabelPH
        loss = tf.reduce_mean(tf.norm(diff_tensor, axis=1))

    with tf.name_scope('Adam'):
        ###############################################
        # Fill your optimizer of choice here!
        ###############################################
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            Optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    # Tensorboard
    # Create a summary to monitor loss tensor
    tf.summary.scalar('LossEveryIter', loss)
    # tf.summary.histogram('H4Pt', H4Pt)
    # tf.summary.histogram('Difference Tensor', diff_tensor)

    # Merge all summaries into a single operation
    MergedSummaryOP = tf.summary.merge_all()

    # Setup Saver
    Saver = tf.train.Saver()

    with tf.Session() as sess:
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
            # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())

        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
            EpochLoss = 0
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                I1Batch, LabelBatch = GenerateBatch(BasePath, TrainImages, TrainLabels, ImageSize, MiniBatchSize)
                FeedDict = {ImgPH: I1Batch, LabelPH: LabelBatch, training: True}
                _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)

                # Save checkpoint every some SaveCheckPoint's iterations
                if PerEpochCounter % SaveCheckPoint == 0:
                    # Save the Model learnt in this epoch
                    SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                    Saver.save(sess,  save_path=SaveName)
                    print('\n' + SaveName + ' Model Saved...')

                # Tensorboard
                Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()
                EpochLoss = EpochLoss + LossThisBatch

            # Print out loss per epoch
            EpochLoss = EpochLoss/NumIterationsPerEpoch
        print('Epoch number: '+str(Epochs)+',Epoch Loss: '+str(EpochLoss))

        # Tensorboards
        ELoss = tf.Summary()
        ELoss.value.add(tag='Epoch Loss', simple_value=EpochLoss)
        Writer.add_summary(ELoss, Epochs)
        # Writer.flush()

        # Validation Loss
        Val_I1Batch, Val_LabelBatch = GenerateBatch(BasePath, ValImages, ValLabels, ImageSize, MiniBatchSize)
        Val_FeedDict = {ImgPH: Val_I1Batch, LabelPH: Val_LabelBatch, training:False}
        ValLoss = sess.run(loss, feed_dict=Val_FeedDict)
        print(', Val Loss: '+str(ValLoss))

        # Tensorboard - validation loss
        ValLossSummary = tf.Summary()
        ValLossSummary.value.add(tag='Validation Loss', simple_value=ValLoss)
        Writer.add_summary(ValLossSummary, Epochs)
        Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
        Saver.save(sess, save_path=SaveName)
        print('\n' + SaveName + ' Model Saved...')


def main():
    """
    Inputs:
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/meera/Desktop/Nitin/cmsc733/P1/YourDirectoryID_p1/Phase2/Data', help='Base path of images')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    # Parser.add_argument('--ModelType', default='Sup', help='Model type Sup or Unsup')
    Parser.add_argument('--NumEpochs', type=int, default=5, help='Number of Epochs to Train for, Default:5')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=64, help='Size of the MiniBatch to use, Default:64')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
    Parser.add_argument('--ImageFile', default='./im_Train.npy', help='Path to pre-generated training images datafile')
    Parser.add_argument('--LabelsFile', default='./h4_Train.npy', help='Path to pre-generated training labels datafile')
    Parser.add_argument('--ValImageFile', default='./im_Val.npy', help='Path to pre-generated validation images datafile')
    Parser.add_argument('--ValLabelsFile', default='./h4_Val.npy', help='Path to pre-generated validation labels datafile')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    # ModelType = Args.ModelType

    TrainImages = np.load(Args.ImageFile)
    TrainLabels = np.load(Args.LabelsFile)

    ValImages = np.load(Args.ValImageFile)
    ValLabels = np.load(Args.ValLabelsFile)

    ImgMean = np.mean(TrainImages, (0,1,2))
    ImgStd = np.std(TrainImages, (0,1,2))

    ValMean = np.mean(ValImages, (0,1,2))
    ValStd = np.std(ValImages, (0,1,2))

    TrainImages = (TrainImages - ImgMean)/ImgStd
    ValImages = (ValImages - ValMean)/ValStd

    # Setup all needed parameters including file reading
    # DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)

    SaveCheckPoint = 100
    ImageSize = TrainImages[0].shape
    NumTrainSamples = TrainImages.shape[0]

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], 2))
    LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 8)) # 1x8 homography labels
    EpochLoss = tf.constant(0.0, shape=None)
    training = tf.placeholder(tf.bool, name='training')

    TrainOperation(ImgPH, LabelPH, training, TrainImages, TrainLabels, ValImages, ValLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath)


if __name__ == '__main__':
    main()
