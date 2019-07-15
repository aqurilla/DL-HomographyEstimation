"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin Suresh,
ECE - UMD
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

# Define the convolutional layer
def conv_layer(input, channels_in, channels_out, name='conv', dropout=False):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(shape=[3,3,channels_in,channels_out], mean=0, stddev=0.1), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name='B')
        conv = tf.nn.conv2d(input, w, strides=[1,1,1,1], padding='SAME')
        act = tf.nn.relu(conv + b)
        if dropout:
            act = tf.nn.dropout(act, 0.5)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

# Define FC layer
def fc_layer(input, channels_in, channels_out, use_relu=False, name='FC'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(shape=[channels_in,channels_out], mean=0, stddev=0.1), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name='B')
        if use_relu:
	    act = tf.nn.relu(tf.matmul(input, w) + b)
	else:
	    act = tf.matmul(input,w)+b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

def HomographyModel(Img, ImageSize, MiniBatchSize, training):
    """
    Inputs:
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    H4Pt - Estimated 4-point homography
    """

    #############################
    # Fill your network here!
    #############################

    # Implementing the regression homography net
    # ImageSize = (128,128)

    conv1 = conv_layer(Img, 2, 64, "conv1")
    conv2 = conv_layer(conv1, 64, 64, "conv2")
    conv2_bn = tf.layers.batch_normalization(conv2, training=training)
    pool1 = tf.nn.max_pool(conv2_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    conv3 = conv_layer(pool1, 64, 64, "conv3")
    conv4 = conv_layer(conv3, 64, 64, "conv4", dropout=True)
    conv4_bn = tf.layers.batch_normalization(conv4, training=training)
    pool2 = tf.nn.max_pool(conv4_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    conv5 = conv_layer(pool2, 64, 128, "conv5")
    conv6 = conv_layer(conv5, 128, 128, "conv6", dropout=True)
    conv6_bn = tf.layers.batch_normalization(conv6, training=training)
    pool3 = tf.nn.max_pool(conv6_bn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    conv7 = conv_layer(pool3, 128, 128, "conv7")
    conv8 = conv_layer(conv7, 128, 128, "conv8", dropout=True)

    conv8_bn = tf.layers.batch_normalization(conv8, training=training)

    # Flatten input
    flattened = tf.reshape(conv8_bn, [-1, 16 * 16 * 128])

    fc1 = fc_layer(flattened, 16 * 16 * 128, 1024, use_relu=True, name="fc1")

    H4Pt = fc_layer(fc1, 1024, 8, use_relu=False, name="fc2")

    return H4Pt
