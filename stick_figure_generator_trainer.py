from hiddenAI.sequential import Sequential
from hiddenAI.generative_adversarial import GAN,ConditionalGAN
from hiddenAI import loss as loss
from hiddenAI import optimizers as optimizers
from hiddenAI import learning_rates as learning_rates
import pickle
from hiddenAI.layers.main_layers import * 
from hiddenAI.layers.activations import *

gen_learning_rate = learning_rates.ConstantLearningRate(0.1)
gen_optimizer = optimizers.BatchGradientDescent(momentum = 0.8,batch_size = 10,learning_rate = gen_learning_rate)
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
dsc_learning_rate = learning_rates.ConstantLearningRate(0.1)
dsc_optimizer = optimizers.BatchGradientDescent(momentum = 0.75,batch_size = 10,learning_rate = dsc_learning_rate)
discriminator = Sequential([784],
							FullyConnected(64),
							Bias(),
							Sigmoid(),
							FullyConnected(64),
							Bias(),
							Sigmoid(),
							FullyConnected(16),
							Bias(),
							Sigmoid(),
							FullyConnected(2),
							Bias(),
							Sigmoid(),
							optimizer = dsc_optimizer)
gan = GAN(discriminator,generator)
with open("datasets/stickfigures/stickfigures","rb") as f:
	real_data = pickle.load(f)
numdata = 2500 
real_data = np.reshape(real_data,(len(real_data),784))[:numdata]
gan.train(real_data)
generator.save_to_file("stored_weights/stickfiguregenerator")

