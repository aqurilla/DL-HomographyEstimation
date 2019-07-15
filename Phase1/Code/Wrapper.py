#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code
"""

# Code starts here:

import numpy as np
import cv2
from Helpers.helpers import *
import argparse
import matplotlib.pyplot as plt
# Add any python libraries here



def main():
	# Add any Command Line arguments here
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--NumFeatures', default=250, help='Number of best features to extract from each image, Default:100')
	Parser.add_argument('--ImDir', default='../Data/Test/TestSet4/*.jpg', help='Directory to find images to stitch')
	Parser.add_argument('--RansacIters', default=700, help='Number of RANSAC iterations to perform')
	
	Args = Parser.parse_args()
	NumFeatures = Args.NumFeatures
	imDir = Args.ImDir
	ransacIters = Args.RansacIters

	"""
	Read a set of images for Panorama stitching
	"""
	pics = getIms(imDir)
	im2 = pics[0]
	validpts2 = np.ones(im2.shape[:-1],dtype=np.bool)
	pics = pics[1:]
	npics = 1
	while pics:
		im1 = pics[0]
		validpts1 = np.ones(im1.shape[:-1],dtype=np.bool)
		pics = pics[1:]
		im1hgt,im1width,im1channels = im1.shape
		im2hgt, im2width,im2channels = im2.shape
		im1YCrCb = cv2.cvtColor(im1,cv2.COLOR_RGB2YCrCb)
		im2YCrCb = cv2.cvtColor(im2,cv2.COLOR_RGB2YCrCb)

	# """
	# Corner Detection
	# Save Corner detection output as corners.png
	# """

		im1scores = cv2.cornerHarris(im1YCrCb[:,:,0],9,3,0.04)
		im2scores = cv2.cornerHarris(im2YCrCb[:,:,0],9,3,0.04)

		# displayAndSave([im1scores,im2scores],1,2,'corners.png')

	# """
	# Perform ANMS: Adaptive Non-Maximal Suppression
	# Save ANMS output as anms.png
	# """

		im1corners = ANMS(im1scores,validpts1,NumFeatures)
		im2corners = ANMS(im2scores,validpts2,NumFeatures*npics)

		def plotScatter(idx):
			vals = [im1corners,im2corners][idx]
			plt.scatter([val[1] for val in vals],[val[0] for val in vals],1,'r')

		# displayAndSave([im1,im2],1,2,'anms.png',plotScatter)

	# """
	# Feature Descriptors
	# Save Feature Descriptor output as FD.png
	# """

		im1vectors = vectorizePoints(im1YCrCb,im1corners,3,40,8)
		im2vectors = vectorizePoints(im2YCrCb,im2corners,3,40,8)

		# displayAndSave([pic.reshape([8,8]) for pic in np.vstack([im1vectors[20:40:4,::3],im2vectors[20:40:4,::3]])],2,5,'FD.png')


	# """
	# Feature Matching
	# Save Feature Matching output as matching.png
	# """

		im1matches,im2matches = matchPoints(im1vectors,im2vectors,im1corners,im2corners,1.0)
		matchImg = cv2.drawMatches(im1,[cv2.KeyPoint(x,y,4) for y,x in im1matches],im2,[cv2.KeyPoint(x,y,4) for y,x in im2matches],\
									[cv2.DMatch(idx,idx,0.5) for idx in range(len(im1matches))],np.zeros([max(im1hgt,im2hgt),im1width+im2width,im1channels]))
		# displayAndSave(matchImg,0,0,'matching.png')

		if len(im1matches) < 10:
			print("Insufficient corner matches; discarding image")
			continue

	# """
	# Refine: RANSAC, Estimate Homography
	# """

		homog,inlierIds = RANSAC(im1matches,im2matches,numIters=ransacIters)
		im1matches = [im1matches[idx] for idx in inlierIds]
		im2matches = [im2matches[idx] for idx in inlierIds]
		refinedMatches = cv2.drawMatches(im1,[cv2.KeyPoint(x,y,4) for y,x in im1matches],im2,[cv2.KeyPoint(x,y,4) for y,x in im2matches],\
									[cv2.DMatch(idx,idx,0.5) for idx in range(len(im1matches))],np.zeros([max(im1hgt,im2hgt),im1width+im2width,im1channels]))
		# displayAndSave(refinedMatches,0,0,'RANSAC.png')
		
		if len(inlierIds) < 4:
			print("Insufficient confident corner matches; discarding image")
			continue

	# """
	# Image Warping + Blending
	# Save Panorama output as mypano.png
	# """

		size1,center1,homog1,homog2,size2,statbox = getStitchingStats(im1,homog,im2)

		validmask = np.zeros(size2,dtype=np.bool)
		validmask[statbox > 0] |= validpts2.ravel()
		validpts2 = validmask

		im2,validpts2 = stitchComponents(size1,im1,validpts1,homog1,center1,homog2,size2,im2,validpts2,statbox)

		displayAndSave(im2,0,0,"mypano.png")
		npics += 1

	
if __name__ == '__main__':
	main()
 
