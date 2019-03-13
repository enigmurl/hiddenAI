from sequential import Sequential
import numpy as np

class GAN(Sequential):
	def __init__(self,discriminator, generator,noise = np.random.random):
		self.discriminator = discriminator 
		self.generator = generator
		self.noise = noise
		if self.generator.layers[-1].output_shape != self.discriminator.layers[0].input_shape:
			raise ValueError("Generator output shape",self.generator.layers[-1].output_shape, "is not the same as the Discriminators input shape",self.discriminator[0].input_shape)
		self.generator_input_shape = generator.layers[0].input_shape
		self.discriminator_input = discriminator.layers[0].input_shape

	def train(self,real_data,real_data_output = np.array([0,1]),generated_data_output = np.array([1,0])):# MAKE THIS INCLUDE THE OPTIMIZERS BETTER
		discriminator_weighted_layers = self.discriminator.weighted_layers
		generator_weighted_layers = self.generator.weighted_layers
		print(generator_weighted_layers)
		self.generator.optimizer.last_gradients = self.generator.optimizer.blank_weights(generator_weighted_layers)
		print("first:",self.generator.optimizer.last_gradients)
		self.discriminator.optimizer.last_gradients = self.discriminator.optimizer.blank_weights(discriminator_weighted_layers)
		print(self.discriminator.optimizer.last_gradients is self.generator.optimizer.last_gradients)
		print(self.discriminator.optimizer is self.generator.optimizer)
		print("scnd:",self.generator.optimizer.last_gradients)
		
		for data in real_data:
			discriminator_gradients = self.discriminator.optimizer.blank_weights(discriminator_weighted_layers)
			generator_gradients = self.generator.optimizer.blank_weights(generator_weighted_layers)
			#train discriminator on real_data 
			discriminator_layers = self.discriminator.training_run(data)
			discriminator_loss_derivative = self.discriminator.loss.derivative_prev_layer(discriminator_layers[-1],real_data_output,batch_size = 2)
			discriminator_gradients = self.discriminator.optimizer.derive_one_data(	discriminator_gradients,
																					self.discriminator.training_layers,
																					discriminator_loss_derivative,
																					discriminator_layers)
			#train discriminator on generated data, while also training generator
			generator_noise = self.noise(size = self.generator_input_shape)
			generator_layers = self.generator.training_run(generator_noise)
			discriminator_layers = self.discriminator.training_run(generator_layers[-1])
			discriminator_loss_derivative = self.discriminator.loss.derivative_prev_layer(discriminator_layers[-1],generated_data_output,batch_size = 2)
			discriminator_gradients,generator_last_derivative = self.discriminator.optimizer.derive_one_data(	discriminator_gradients,
																												self.discriminator.training_layers,
																												discriminator_loss_derivative,
																												discriminator_layers,last_layer_return = True)
			generator_gradients = self.generator.optimizer.derive_one_data(	generator_gradients,
																			self.generator.training_layers,
																			generator_last_derivative,
																			generator_layers)
			print("last:",self.generator.optimizer.last_gradients)
			self.discriminator.optimizer.descend(discriminator_gradients,discriminator_weighted_layers)		
			self.generator.optimizer.descend(generator_gradients,generator_weighted_layers)

	def training_run(self,input_layer,use_generator = True):
		return self.generator.training_run(input_layer) if use_generator else self.discriminator.training_run(input_layer)

	def run_generator(self,input_layer):
		return self.generator.run(input_layer)

	def run_discriminator(self,input_layer):
		return self.discriminator.run(input_layer)
if __name__ == "__main__":
	from layers.hidden import *
	from layers.activations import *
	import optimizers
	discriminator = Sequential([1],FullyConnected(2),Bias(),optimizer = optimizers.BatchGradientDescent(batch_size = 4))
	generator     = Sequential([4],FullyConnected(2),Bias(),ReLU(),FullyConnected(1),optimizer = optimizers.BatchGradientDescent)
	c = GAN(discriminator,generator)
	data = [np.array([100 + np.random.random()]) for i in range(1000)]
	c.train(data)
	print(generator.run(np.random.random(size = (4))))	
