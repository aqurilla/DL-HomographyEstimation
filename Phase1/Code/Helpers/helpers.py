from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from random import sample

def getIms(imdir):
	pics = sorted(list(glob(imdir)))
	return [cv2.cvtColor(cv2.imread(pic),cv2.COLOR_BGR2RGB) for pic in pics]

def standardizeImg(img):
	out = np.zeros(img.shape)
	means = np.mean(img,(0,1))
	stds = np.std(img,(0,1))
	for idx in range(img.shape[-1]):
		out[:,:,idx] = (img[:,:,idx] - means[idx])/stds[idx]
	return np.float32(out)

def standardizeVecs(vecs):
	return (vecs - np.mean(vecs)) / np.std(vecs)

def RANSAC(points1,points2,epsilon=800,numIters=150,quitThreshhold=0.65):
	l = len(points1)
	cutoff = quitThreshhold * l
	ids = np.arange(l)
	sampleids = range(l)
	srcpts = np.transpose(np.concatenate([np.array([(x,y) for (y,x) in points1],dtype=np.float32),np.ones([l,1],dtype=np.float32)],1))
	dstpts = np.transpose(np.concatenate([np.array([(x,y) for (y,x) in points2],dtype=np.float32),np.ones([l,1],dtype=np.float32)],1))
	maxInliers = list()
	bestHomog = np.zeros([3,3])
	for _ in range(numIters):
		pointids = sample(sampleids,4)
		p1s = np.transpose(srcpts[:2,pointids])
		p2s = np.transpose(dstpts[:2,pointids])
		homog = cv2.getPerspectiveTransform(p1s,p2s)
		inliers = computeInliers(dstpts,np.matmul(homog,srcpts),ids,epsilon)
		if len(inliers) > len(maxInliers):
			maxInliers = inliers
			bestHomog = homog
			if len(maxInliers) > cutoff:
				return bestHomog, maxInliers
	return bestHomog, maxInliers		

def computeInliers(correct,predicted,ids,epsilon):
	ssds = np.sum(np.power(correct-predicted,2),0)
	return ids[ssds < epsilon]

def vectorizePoints(img,points,channels=3,patchsize=40,subsample=8):
	img = standardizeImg(img)
	half=patchsize//2
	pointVecs = np.zeros([len(points),subsample*subsample*channels])
	substep = patchsize//subsample
	for idx,(y,x) in enumerate(points):
		patch = img[y-half:y+half,x-half:x+half,:]
		blurred = cv2.medianBlur(patch,5)
		pointVecs[idx,:] = blurred[::substep,::substep,:].ravel()
	return pointVecs

def matchPoints(vecs1,vecs2,points1,points2,threshhold=1):
	vrange = np.arange(len(points1))
	hrange = np.arange(len(points2))
	dists = cdist(vecs1,vecs2,'sqeuclidean')
	highest = np.amax(dists)

	best241s = np.argmin(dists,1)
	bestvalsH = dists[vrange,best241s]
	best142s = np.argmin(dists,0)
	bestvalsV = dists[best142s,hrange]

	dists[vrange,best241s] = highest
	secondsH = np.amin(dists,1)
	dists[vrange,best241s] = bestvalsH

	dists[best142s,hrange] = highest
	secondsV = np.amin(dists,0)
	dists[best142s,hrange] = bestvalsV

	keepH = (bestvalsH/secondsH) < threshhold
	keepV = (bestvalsV/secondsV) < threshhold

	match1 = list()
	match2 = list()
	for idx in vrange:
		best2 = best241s[idx]
		recip = best142s[best2]
		if recip == idx and (keepH[idx] or keepV[best2]):
			match1.append(points1[idx])
			match2.append(points2[best2])
	return match1,match2

def validNeighbors(pt,validpts):
	y,x = pt
	for i in range(y-1,y+2):
		for j in range(x-1,x+2):
			if not validpts[i,j]:
				return False
	return True

def stitchFromH4pt(points1,h4pt,im1,im2,valid2=None):

	'''points1 and h4pt should be 4x2 arrays  of [[x1,y1],[x2,y2]...] .
	 	points1 should come from im1, and adding h4pt and points1
	 	should yield points from im2 .
	 	im1 and im2 are expected to have multiple channels.
	 	valid2 is the same size as im2 and optionally masks away parts of it'''

	points2 = points1 + h4pt
	homog = cv2.getPerspectiveTransform(p1s,p2s)
	size1,center1,homog1,homog2,size2,statbox = getStitchingStats(im1,homog,im2)

	vp2 = np.zeros(size2,dtype=np.bool)
	vp2[statbox>0] |= (valid2 if valid2 else np.ones(im2.shape[:2])).ravel() > 0 

	result, _ = stitchComponents(size1,im1,np.ones(im1.shape[:2],dtype=np.bool),homog1,center1,homog2,size2,im2,vp2,statbox)
	return result

def getStitchingStats(im1,homog,im2):
	im1hgt,im1width,im1channels = im1.shape
	im2hgt,im2width,im2channels = im2.shape
	corners = np.matmul(homog,np.transpose(np.array([[0,0,1],[im1width,0,1],[0,im1hgt,1],[im1width,im1hgt,1]])))
	newpoints = np.transpose(corners[:2,:] / corners[2,:])

	top1 = int(np.amin(newpoints[:,1]))
	left1 = int(np.amin(newpoints[:,0]))
	overheight = int(min(top1,0))
	overwidth = int(min(left1,0))

	bottom1 = int(np.amax(newpoints[:,1])-top1)
	right1 = int(np.amax(newpoints[:,0])-left1)
	bottom = int(max(np.amax(newpoints[:,1])-overheight,im2hgt-overheight))
	right = int(max(np.amax(newpoints[:,0])-overwidth,im2width-overwidth))

	trueTop1 = max(top1,0)
	trueLeft1 = max(left1,0)
	center1 = (trueLeft1 + right1//2, trueTop1 + bottom1//2)

	statbox = np.zeros([bottom,right],dtype=np.bool)
	statbox[-overheight:im2hgt-overheight,-overwidth:im2width-overwidth] = True

	translate = np.eye(3)
	translate[:2,2] = [-left1,-top1]
	translate2 = np.eye(3)
	translate2[:2,2] = [-overwidth,-overheight]
	homog1 = np.matmul(translate,homog)
	homog2 = np.matmul(translate2,homog)

	return [bottom1,right1],center1,homog1,homog2,[bottom,right],statbox

def stitchComponents(size1,im1,valid1,homog1,center1,homog2,size2,im2,valid2,statbox):
	out = np.zeros(size2 + [im2.shape[-1]],dtype=np.uint8)
	out[statbox>0,:] = im2.reshape([-1,im2.shape[-1]])

	warped = cv2.warpPerspective(im1,homog1,dsize=(size1[1],size1[0]))
	boundingMask = cv2.warpPerspective(np.uint8(valid1)*255,homog1,dsize=(size1[1],size1[0]))
	out2 = cv2.warpPerspective(im1,homog2,dsize=(size2[1],size2[0]))
	out2[valid2>0,:] = out[valid2>0,:]
	footprint = valid2 | (cv2.warpPerspective(np.uint8(valid1),homog2,dsize=(size2[1],size2[0])) > 0)

	fail = True
	while fail:
		try:
			im2 = cv2.seamlessClone(np.uint8(warped),np.uint8(out2),np.uint8(boundingMask),center1,cv2.NORMAL_CLONE)
			fail = False
		except:
			warped = warped[1:-1,1:-1,:]
			boundingMask = boundingMask[1:-1,1:-1,:]
	return im2,footprint

def ANMS(scoreMap,validpts,nBest):
	maxima = peak_local_max(scoreMap,min_distance=5,threshold_abs=0.0,exclude_border=20)
	valids = np.array([maxpoint for maxpoint in maxima if validNeighbors(maxpoint,validpts)])
	scores = np.array([scoreMap[y,x] for y,x in valids])
	scores = np.append(scores,[np.amax(scores)+1],0)
	pairdists = cdist(valids,valids)
	pairdists = np.append(pairdists,np.repeat(np.array(np.amax(pairdists)),len(valids)).reshape([-1,1]),1)
	dists = {tuple(valids[idx]):np.amin(pairdists[idx,scores>scores[idx]]) for idx in range(len(valids))}
	out = sorted(dists.keys(),key=lambda point: dists[point], reverse= True)[:nBest]
	return sorted(out)

def displayAndSave(images,rows=0,cols=0,filename="image.png",imfuncs=None):
	fig = plt.figure()
	if rows or cols:
		for idx,img in enumerate(images):
			ax = fig.add_subplot(rows,cols,idx+1)
			ax.set_xticks([])
			ax.set_yticks([])
			plt.imshow(img)
			if imfuncs:
				imfuncs(idx)
		plt.savefig("../Output/"+filename)
	else:
		plt.imshow(images)
		if imfuncs:
			imfuncs(plt)
		plt.imsave("../Output/"+filename,images)
	plt.show()
