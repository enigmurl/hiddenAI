from sequential import Sequential 
from convolution import *
from activations import *
from hidden import *
from pooling import *
from mnist import  MNIST
import time
import random
import numpy as np

def printIm(img):#prints a representation of the digit
	for preLine in img[0]:
		newLine = ["#" if val>0 else "." for val in preLine]
		print(" ".join(newLine))
start_data = time.time()
mndata = MNIST('digitdata')
images,labels = mndata.load_training()
num_data = 10000# of the 60,000 images in the MNIST database, how many do we want to use
formatted_inputs = []
formatted_labels = []
for ind in range(num_data):
	output = np.zeros(10) 
	output[labels[ind]] = 1
	new_image = np.array([[pxl/255 for pxl in images[ind]]])
	new_image = np.reshape(new_image,(1,28,28))
	formatted_labels.append(output)
	formatted_inputs.append(new_image)
#INITIATING THE MODEL
'''
#net = Sequential((784),
				FullyConnected(16),
				Bias(),#
				Sigmoid(),
				FullyConnected(16),
				Bias(),
				Sigmoid(),
				FullyConnected(10),
				Bias(),
				Sigmoid())'''
net = Sequential((1,28,28),
				Convolution2D(num_filters = 16,filter_size = (4,4),stride = (4,4)),
				MaxPooling2D(),
				Bias(),	
				Convolution2D(num_filters = 8, filter_size = (3,3),stride = (3,3)),
				MaxPooling2D(),
				Bias(),
				Convolution2D(num_filters = 4, filter_size = (2,2),stride = (2,2)),
				MaxPooling2D(),
				Bias(),
				FullyConnected(16),
				Bias(),
				Sigmoid(),
				FullyConnected(10),
				Bias(),
				Sigmoid())
				
#net.open_from_file("digitweight")# open up from digitweight a pre trained model
num_trials = 1 # how many times we run over the same 10,000 images (epoch)

#TRAINING THE MODEL
net.batch_gradient_descent(formatted_inputs,formatted_labels,epoch = num_trials,batch_size = 10)#training the mode using stochasatic gradient descent

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
	print("ACTUAL,MACHINE:",correct,max_ind,result)
	if correct == max_ind:	
		score+=1
		print('correct')
print("SCORE:",score,"TOTAL TIME:",time.time()-start_data)
