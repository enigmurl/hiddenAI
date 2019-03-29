import numpy as np
from hiddenAI.sequential import Sequential
from hiddenAI.generative_adversarial import GAN,ConditionalGAN
from hiddenAI import loss as loss
from hiddenAI import optimizers as optimizers
from hiddenAI import learning_rates as learning_rates

from hiddenAI.layers.main_layers import * 
from hiddenAI.layers.activations import *
from hiddenAI.layers.convolution import *

optimizer = optimizers.BatchGradientDescent()
a = Sequential([1,1,2],
				TransposedConvolution2D(3,(3,3),(3,3)),
				Bias(),
				TransposedConvolution2D(2,(2,2),(2,2)),
				Bias(),
				optimizer = optimizer)
training = np.array([
[[[0.11987,0.98711]]],
[[[0.9143,0.12834]]],
[[[0.9723,0.183975]]],
[[[0.1534,0.91324]]],
[[[0.95143,0.1268203]]],
[[[0.13542678,0.9234]]],
[[[0.925834,0.19287]]],
[[[0.1293,0.97821654]]],
])
real =np.array( [
	[[[1,0,1],[0,1,0],[1,0,1]]],
	[[[0,1,0],[1,0,1],[0,1,0]]],
	[[[1,1,1],[1,0,1],[1,1,1]]],
	[[[0,1,1],[0,1,0],[1,0,1]]],
	[[[1,1,1],[1,0,1],[0,1,0]]],
	[[[1,0,1],[0,1,0],[1,0,0]]],
	[[[0,1,0],[1,0,1],[1,1,1]]],
	[[[0,0,1],[0,1,0],[1,0,1]]]
	])
a.train(training,real)
for ind,val in enumerate(training):
	print("REAL:",real[ind],"PREDICTED",a.run(val))
