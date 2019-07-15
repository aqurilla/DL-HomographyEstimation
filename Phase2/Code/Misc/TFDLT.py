#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 2 DLT algorithm

Adapted from:
https://github.com/tynguyen/unsupervisedDeepHomographyRAL2018

Author(s):
Nitin Suresh,
ECE - UMD
"""

# Code starts here:

import numpy as np
import tensorflow as tf
import cv2
import pdb
# Add any python libraries here

# Auxiliary matrices used to solve DLT
AM1  = np.array([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float32)

AM2  = np.array([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ]], dtype=np.float32)

AM3  = np.array([
          [0],
          [1],
          [0],
          [1],
          [0],
          [1],
          [0],
          [1]], dtype=np.float64)

AM4  = np.array([
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float32)

AM5  = np.array([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float32)

AM6  = np.array([
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ]], dtype=np.float64)

AM7_1 = np.array([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float32)

AM7_2 = np.array([
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ]], dtype=np.float32)

AM8  = np.array([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ]], dtype=np.float32)

b_mat_np  = np.array([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , -1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float32)

def DLT(C_a, H4Pt):
    """
    Tensorflow implementation of the Direct Linear Transformation
    Inputs:
    C_a - original image points of shape [num_batches, 8]
    H4Pt - distortion matrix of shape [num_batches, 8]
    """
    # Generate C_b by adding the distortion matrix to C_a
    num_batch = tf.shape(C_a)[0]
    C_a_rep = tf.expand_dims(C_a, 2)
    H4Pt_rep = tf.expand_dims(H4Pt, 2)
    # C_b = C_a + H4Pt
    C_b_rep = tf.add(C_a_rep, H4Pt_rep)

    # Matrix for implementing over batch
    rep_mat = [num_batch, 1, 1]

    # Create the Ax=b equation

    # A_i = np.array([[0,   0,   0, -u_i, -v_i, -1, v_i_d*u_i, v_i_d*v_i],
    #                 [u_i, v_i, 1,  0,    0,    0, -u_i_d*u_i, -u_i_d*v_i]])
    #
    # b_i = np.array([[-v_i_d],[u_i_d]])

    # Repeat for all 4 points

    # A_mat
    # A1
    A1_i = tf.constant(AM1, tf.float32)
    # Tile it with num_batch
    A1_rep = tf.tile(tf.expand_dims(A1_i, 0), rep_mat)
    # A1 = [0, u_i]' for all 4 points
    A1_col = tf.matmul(A1_rep, C_a_rep)
    A1_col = tf.reshape(A1_col,[-1,8])

    # A2
    A2_i = tf.constant(AM2, tf.float32)
    # Tile it with num_batch
    A2_rep = tf.tile(tf.expand_dims(A2_i, 0), rep_mat)
    # A2 = [0, v_i]' for all 4 points
    A2_col = tf.matmul(A2_rep, C_a_rep)
    A2_col = tf.reshape(A2_col,[-1,8])

    # A3
    A3_i = tf.constant(AM3, tf.float32)
    # Tile it with num_batch
    # A3 = [0,1]' for all 4 points
    A3_col = tf.tile(tf.expand_dims(A3_i, 0), rep_mat)
    A3_col = tf.reshape(A3_col,[-1,8])

    # A4
    A4_i = tf.constant(AM4, tf.float32)
    # Tile it with num_batch
    A4_rep = tf.tile(tf.expand_dims(A4_i, 0), rep_mat)
    # A4 = [-u_i, 0]' for all 4 points
    A4_col = tf.matmul(A4_rep, C_a_rep)
    A4_col = tf.reshape(A4_col,[-1,8])

    # A5
    A5_i = tf.constant(AM5, tf.float32)
    # Tile it with num_batch
    A5_rep = tf.tile(tf.expand_dims(A5_i, 0), rep_mat)
    # A5 = [-v_i, 0]' for all 4 points
    A5_col = tf.matmul(A5_rep, C_a_rep)
    A5_col = tf.reshape(A5_col,[-1,8])

    # A6
    A6_i = tf.constant(AM6, tf.float32)
    # Tile it with num_batch
    # A3 = [-1,0]' for all 4 points
    A6_col = tf.tile(tf.expand_dims(A6_i, 0), rep_mat)
    A6_col = tf.reshape(A6_col,[-1,8])


    # A7 = [v_i_d * u_i, u_i_d * -u_i]
    A7_1_i = tf.constant(AM7_1, tf.float32)
    A7_2_i = tf.constant(AM7_2, tf.float32)
    # Tile it with num_batch
    A7_1_rep = tf.tile(tf.expand_dims(A7_1_i, 0), rep_mat)
    A7_2_rep = tf.tile(tf.expand_dims(A7_2_i, 0), rep_mat)
    # A7 = [v_i_d * u_i, u_i_d * -u_i] = [v_i_d u_i_d] * [u_i -u_i]
    A7_col = tf.matmul(A7_1_rep, C_b_rep)*tf.matmul(A7_2_rep, C_a_rep)
    A7_col = tf.reshape(A7_col,[-1,8])

    A8_i = tf.constant(AM8, tf.float32)
    A8_rep = tf.tile(tf.expand_dims(A8_i, 0), rep_mat)
    # A8 = [v_i_d * v_i, -u_i_d * v_i] = [v_i_d u_i_d] * [v_i -v_i]
    A8_col = tf.matmul(A7_1_rep, C_b_rep)*tf.matmul(A8_rep, C_a_rep)
    A8_col = tf.reshape(A8_col,[-1,8])

    # b_mat
    b_mat_i = tf.constant(b_mat_np, tf.float32)
    # Tile it with num_batch
    b_mat_rep = tf.tile(tf.expand_dims(b_mat_i, 0), rep_mat)
    # Implement b_i = np.array([[-v_i_d],[u_i_d]]) for all 4 points
    b_mat = tf.matmul(b_mat_rep, C_b_rep)

    # From columns - stack and transpose to get A_mat
    # A_mat = [A1 A2 A3 A4 A5 A6 A7 A8]'
    A_stack = tf.stack([A1_col, A2_col, A3_col, A4_col, A5_col, A6_col, A7_col, A8_col],axis=1)
    # transpose in dim=0
    A_mat = tf.transpose(A_stack, perm=[0,2,1])

    # Solve equation Ax=b
    # H4Pt_estimate = np.linalg.solve(A_t,b_t)
    H8_mat = tf.linalg.solve(A_mat, b_mat)

    # Append ones to generate 3x3 matrix finally
    # H4Pt_estimate = (np.append(H4Pt_estimate, [[1]], axis=0)).reshape([3,3])
    H9_mat_i = tf.reshape(tf.concat([H8_mat, tf.ones(rep_mat)], 1), [-1,9])
    H9_mat = tf.reshape(H9_mat_i,[-1,3,3])

    # H9_mat estimates H_ab, and is used by the spatial transformer network to transform I_a
    return H9_mat
