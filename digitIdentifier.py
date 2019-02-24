from main import NeuralNet
from info1D import Vector
from info2D import Matrix
from layers1D import *
from mnist import  MNIST
import time
import random
import pygame
#SCREEN = pygame.display.set_mode((400,400))
startData = time.time()
mndata = MNIST('digitdata')
images,labels = mndata.load_training()
numData = 60000
images = images
labels = labels
running = True
print(len(images),len(labels))
trainingData = []
newLabels = []
for ind in range(numData):
	val = images[ind]	
	output = Vector(data = 0,vecSize = 10)
	output.data[labels[ind]] = 1
	newLabels.append(output)
	newImage = Vector([pxl/255 for pxl in val])
	trainingData.append(newImage)
print("TrainingData rendered,totalTime:",time.time()-startData)
startTrain = time.time()

net = NeuralNet(FullyConnected(784,16),Bias(16,16),Sigmoid(),FullyConnected(16,16),Bias(16,16),Sigmoid(),FullyConnected(16,10),Bias(10,10),Sigmoid())
net.openFromFile("digitweight")
numTrials = 3
net.stochasticDescent(trainingData,newLabels,epoch = numTrials,batchSize = 10)
net.saveToFile("digitweight")
print("SAVED TO FILE: digitweight")
score = 0
def printIm(img):
	for i in range(28):
		preLine = img[i*28:28*(i+1)]
		newLine = ["#" if val>0 else "." for val in preLine]
		print(" ".join(newLine))
def drawIm(img):
	for ind,val in enumerate(img):
		x = ind%28 * 10
		y = ind//28 * 10
		pygame.draw.rect(SCREEN,(val,val,val),pygame.Rect(x,y,10,10))
total = 0
#while running:
#	for event in pygame.event.get():
#		if event.type == pygame.QUIT:
#			running = False
#			break
for i in range(100):
	total+=1
	a = random.randint(0,numData-1)
	correct = newLabels[a]
	result = net.run(trainingData[a])
	maxresult = 0
	maxInd = 0
	for ind,val in enumerate(result):
		if val>= maxresult:
			maxInd = ind
			maxresult = val
	maxresult2 = 0
	maxInd2 = 0
	for ind,val in enumerate(correct):
		if val>= maxresult2:
			maxInd2 = ind
			maxresult2 = val
	printIm(trainingData[a].data)
	#drawIm(trainingData[a].data)
	print("ACTUAL,MACHINE:",maxInd2,correct,maxInd,result)
	if maxInd2 == maxInd:	
		#time.sleep(1)
		score+=1
	#pygame.display.flip()
print("SCORE:",score,"TOTAL TIME:",time.time()-startData)
