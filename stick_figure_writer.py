import numpy as np
import pickle
from hiddenAI.progress_bar import ProgressBar
import pygame as pg
from skimage.draw import line_aa,circle
import matplotlib.pyplot as plt
numdata = 250000
dataset = np.zeros((numdata,28,28))
def printIm(img):#prints a representation of the digit
	for preLine in img:
		newLine = ["#" if val>0 else "." for val in preLine]
		print(" ".join(newLine))
pb = ProgressBar(total = numdata)
for i in range(numdata):
	pb.update()
	rint = np.random.randint
	head_pos = [rint(12,16),rint(5,8)]
	head_rad = rint(4,6)
	neck = 	[head_pos[0]+2+head_rad,head_pos[1] + rint(-2,2)]
	arm1 = [neck[0]-rint(4,8),neck[1]+rint(-1,10)]
	arm2 = [neck[0]+rint(4,8),neck[1]+rint(-1,10)]
	torso = [neck[0]+rint(0,2),neck[1] + rint(7,8)]
	leg1 = [torso[0]-rint(0,10),torso[1]+rint(1,8)]
	leg2 = [torso[0]+rint(0,10),torso[1]+rint(1,8)]
	for cord in (head_pos,neck,arm1,arm2,torso,leg1,leg2):
		if cord[0] < 0:
			cord[0] = 0
		elif cord[0] >27:
			cord[0] = 27
		if cord[1] < 0:
			cord[1] = 0
		elif cord[1] >27:
			cord[1] = 27

	img = dataset[i]
	rr,cc,val = line_aa(*head_pos,*neck)
	img[rr,cc] = val
	rr,cc,val = line_aa(*arm1,*neck)
	img[rr,cc] = val
	rr,cc,val = line_aa(*arm2,*neck)
	img[rr,cc] = val
	rr,cc,val = line_aa(*torso,*neck)
	img[rr,cc] = val
	rr,cc,val = line_aa(*torso,*leg1)
	img[rr,cc] = val
	rr,cc,val = line_aa(*torso,*leg2)
	img[rr,cc] = val
	rr,cc = circle(*head_pos,head_rad)
	img[rr,cc] = 0.75
	dataset[i] = img.T
with open("datasets/stickfigures/stickfigures","wb") as f:
	pickle.dump(dataset,f)
