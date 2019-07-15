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
from Misc.TFDLT import *
from Misc.online_data_gen import *
import pdb

# Don't generate pyc codes
sys.dont_write_bytecode = True


def GenerateBatch(BasePath, ImageSize, MiniBatchSize, DataSelect):
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
    DataSelect - '0': Training, '1': Validation, '2': Testing dataset
    """
    # I1Batch = []
    # LabelBatch = []
    # I_a_Batch = []
    # I_b_Batch = []
    # C_a_Batch = []

    I1Batch = np.zeros((MiniBatchSize, 128, 128, 2)).astype(np.float32)
    LabelBatch = np.zeros((MiniBatchSize, 8)).astype(np.float32)
    I_a_Batch = np.zeros((MiniBatchSize, 240, 320, 1)).astype(np.float32)
    I_b_Batch = np.zeros((MiniBatchSize, 240, 320, 1)).astype(np.float32)
    C_a_Batch = np.zeros((MiniBatchSize, 8)).astype(np.float32)

    ImageNum = 0
    while ImageNum < MiniBatchSize:

        # Generate random index
	# if DataSelect==0:
        #     RandIdx = np.random.randint(1, 5001)
	# else:
	#     RandIdx = np.random.randint(1, 950)

        stack_img, H4Pt, C_a, I_a, I_b = gen_data(DataSelect=DataSelect)

        I1 = np.float32(stack_img)
        I_a = np.float32(I_a)
        I_b = np.float32(I_b)

        # Append images and labels
        I1Batch[ImageNum,:,:,:] = I1
        LabelBatch[ImageNum,:] = H4Pt
        C_a_Batch[ImageNum,:] = C_a
        I_a_Batch[ImageNum,:,:,0] = I_a
        I_b_Batch[ImageNum,:,:,0] = I_b

        ImageNum += 1

    # I1Batch = (I1Batch - np.mean(I1Batch, (0,1,2)))/np.std(I1Batch, (0,1,2))
    # I_a_Batch = (I_a_Batch - np.mean(I_a_Batch, (0,1,2)))/np.std(I_a_Batch, (0,1,2))
    # I_b_Batch = (I_b_Batch - np.mean(I_b_Batch, (0,1,2)))/np.std(I_b_Batch, (0,1,2))
    I1Batch = (I1Batch-52.0)/80.0
    I_a_Batch = (I_a_Batch-52.0)/80.0
    I_b_Batch = (I_b_Batch-52.0)/80.0

    return I1Batch, LabelBatch, C_a_Batch, I_a_Batch, I_b_Batch


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


def TrainOperation(ImgPH, LabelPH, C_a_PH, I_a_PH, I_b_PH, NumTrainSamples, ImageSize,
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
    training = True

    # identity = np.array([[1., 0., 0.],
    #                      [0., 1., 0.],
    #                      [0., 0., 1.]])
    #
    # identity = identity.flatten()
    #
    # theta = tf.Variable(initial_value=identity)
    # theta_t = tf.expand_dims(theta, 0)
    # pdb.set_trace()
    # H9_mat = tf.tile(theta_t, [MiniBatchSize, 1, 1])

    H4Pt = HomographyModel(ImgPH, ImageSize, MiniBatchSize, training)

    H9_mat = DLT(C_a=C_a_PH, H4Pt=H4Pt)

    # Calculate H_inv
    img_w = I_b_PH.get_shape().as_list()[2]
    img_h = I_b_PH.get_shape().as_list()[1]

    M = np.array([[img_w/2.0, 0.0, img_w/2.0],\
                 [0.0, img_h/2.0, img_h/2.0],\
                 [0.0, 0.0, 1.0]]).astype(np.float32)

    Minv = np.linalg.inv(M)

    M_t  = tf.constant(M, tf.float32)
    M_rep = tf.tile(tf.expand_dims(M_t, [0]), [MiniBatchSize, 1,1])

    Minv_t  = tf.constant(Minv, tf.float32)
    Minv_rep = tf.tile(tf.expand_dims(Minv_t, [0]), [MiniBatchSize, 1,1])

    # Convert to H_inv
    H9_inv = tf.matmul(tf.matmul(Minv_rep, H9_mat), M_rep)

    with tf.name_scope('Loss'):
        ###############################################
        # Fill your loss function of choice here!
        ###############################################
        out_size = (img_h, img_w)

        # I_a_PH = tf.expand_dims(I_a_PH, 3)
        # I_b_PH = tf.expand_dims(I_b_PH, 3)

        # Warp the image I_a with the inverse transform
        warped_I_a,_ = transformer(U=I_a_PH, theta=H9_inv, out_size=out_size)
        # pdb.set_trace()
        loss = tf.reduce_mean(tf.abs(warped_I_a - I_b_PH))

    with tf.name_scope('Accuracy'):
        accuracy = tf.reduce_mean(tf.abs(H4Pt-LabelPH))


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
    tf.summary.scalar('Accuracy', accuracy)
    
    tf.summary.histogram('I_a_PH',I_a_PH)
    tf.summary.histogram('I_b_PH',I_b_PH)
    tf.summary.histogram('warped_I_a', warped_I_a)
    tf.summary.histogram('H4Pt', H4Pt)

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
                I1Batch, LabelBatch, C_a_Batch, I_a_Batch, I_b_Batch = GenerateBatch(BasePath, ImageSize, MiniBatchSize, DataSelect=0)
                FeedDict = {ImgPH: I1Batch, LabelPH: LabelBatch, C_a_PH: C_a_Batch, I_a_PH: I_a_Batch, I_b_PH: I_b_Batch}
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

            # Tensorboard - Epoch Loss
            ELoss = tf.Summary()
            ELoss.value.add(tag = 'Epoch Loss', simple_value=EpochLoss)
            Writer.add_summary(ELoss, Epochs)
            Writer.flush()

            # Validation loss
            I1Batch_v, LabelBatch_v, C_a_Batch_v, I_a_Batch_v, I_b_Batch_v = GenerateBatch(BasePath, ImageSize, MiniBatchSize, DataSelect=1)
            FeedDict_v = {ImgPH: I1Batch_v, LabelPH: LabelBatch_v, C_a_PH: C_a_Batch_v, I_a_PH: I_a_Batch_v, I_b_PH: I_b_Batch_v}
            ValLoss = sess.run(loss, feed_dict=FeedDict_v)

            # Tensorboard - Validation loss
            ValLossSummary = tf.Summary()
            ValLossSummary.value.add(tag = 'Validation Loss', simple_value=ValLoss)
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
    Parser.add_argument('--ImageFile', default='./testsave_im.npy', help='')
    Parser.add_argument('--LabelsFile', default='./testsave_h4.npy', help='')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    # ModelType = Args.ModelType

    # TrainImages = np.load(Args.ImageFile)
    # TrainLabels = np.load(Args.LabelsFile)

    # Setup all needed parameters including file reading
    # DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)

    SaveCheckPoint = 100
    ImageSize = (128, 128, 2)
    NumTrainSamples = 5000

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 128, 128, 2))
    LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 8)) # 1x8 homography labels
    C_a_PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 8)) # 1x8 C_a corner points
    I_a_PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 240, 320, 1))
    I_b_PH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 240, 320, 1))

    TrainOperation(ImgPH, LabelPH, C_a_PH, I_a_PH, I_b_PH, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath)


if __name__ == '__main__':
    main()
