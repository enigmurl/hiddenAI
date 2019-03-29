from hiddenAI.sequential import Sequential
from hiddenAI.generative_adversarial import GAN,ConditionalGAN
from hiddenAI import loss as loss
from hiddenAI import optimizers as optimizers
from hiddenAI import learning_rates as learning_rates
import pickle
from hiddenAI.layers.main_layers import * 
from hiddenAI.layers.activations import *
import numpy as np

def printIm(img):#prints a representation of the digit
	new_img = [img[i*28:28*(i+1)] for i in range(28)]
	for preLine in new_img:
		newLine = ["#" if val>0.5 else "." for val in preLine]
		print(" ".join(newLine))

gen_learning_rate = learning_rates.DecayLearningRate(0.1,0)
gen_optimizer = optimizers.BatchGradientDescent(momentum = 0.8,batch_size = 20,learning_rate = gen_learning_rate)
generator = Sequential([10],
						FullyConnected(32),
						Bias(),
						Sigmoid(),
						FullyConnected(64),
						Bias(),
						Sigmoid(),
						FullyConnected(64),
						Bias(),
						Sigmoid(),
						FullyConnected(784),
						Bias(),
						Sigmoid(),	
						optimizer = gen_optimizer)
generator.open_from_file("stored_weights/stickfiguregenerator")
#with open("datasets/stickfigures/stickfigures","rb") as f:
#	real_data = pickle.load(f)
noise = np.random.random(size = (10))
result = generator.run(noise)
printIm(result)
