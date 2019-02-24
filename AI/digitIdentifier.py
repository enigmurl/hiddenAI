from net import NeuralNet
from layers1D import *
from mnist import  MNIST
import time
import random
import numpy as np

def printIm(img):#prints a representation of the digit
	for i in range(28):
		preLine = img[i*28:28*(i+1)]
		newLine = ["#" if val>0 else "." for val in preLine]
		print(" ".join(newLine))
start_data = time.time()
mndata = MNIST('digitdata')
images,labels = mndata.load_training()
num_data = 60000# of the 60,000 images in the MNIST database, how many do we want to use
formatted_inputs = []
formatted_labels = []
for ind in range(num_data):
	output = np.zeros(10) 
	output[labels[ind]] = 1
	newImage = np.array([pxl/255 for pxl in images[ind]])
	
	formatted_labels.append(output)
	formatted_inputs.append(newImage)
#INITIATING THE MODEL
net = NeuralNet(FullyConnected(784,16),
				Bias(16,16),#
				Sigmoid(),
				FullyConnected(16,16),
				Bias(16,16),
				Sigmoid(),
				FullyConnected(16,10),
				Bias(10,10),
				Sigmoid())

#net.open_from_file("digitweight")# open up from digitweight a pre trained model
num_trials = 1 # how many times we run over the same 10,000 images (epoch)

#TRAINING THE MODEL
net.stochastic_descent(formatted_inputs,formatted_labels,epoch = num_trials,batch_size = 10)#training the mode using stochasatic gradient descent

net.save_to_file("digitweight")#after training is complete save the file

#ASSESING THE MODEL
score = 0
for i in range(100):
	a = random.randint(0,num_data-1)
	correct = labels[a]
	result = net.run(formatted_inputs[a])
	maxresult = 0
	max_ind = 0
	for ind,val in enumerate(result):
		if val>= maxresult:
			max_ind = ind
			maxresult = val
	maxresult2 = 0
	printIm(formatted_inputs[a])
	print("ACTUAL,MACHINE:",correct,max_ind)
	if correct == max_ind:	
		score+=1
		print('correct')
print("SCORE:",score,"TOTAL TIME:",time.time()-start_data)
