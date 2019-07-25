from hiddenAI.sequential import Sequential
from hiddenAI.progress_bar import ProgressBar
import numpy as np

class GAN(Sequential):
	def __init__(self,discriminator, generator,noise = np.random.random):
		self.discriminator = discriminator 
		self.generator = generator
		self.noise = noise
		self.generator_input_shape = generator.layers[0].input_shape
		self.discriminator_input_shape = discriminator.layers[0].input_shape

	def train(self,real_data,real_data_output = np.array([0,1]),generated_data_output = np.array([1,0]),progress_bar = True):# MAKE THIS INCLUDE THE OPTIMIZERS BETTER
		discriminator_weighted_layers = self.discriminator.weighted_layers
		self.discriminator.optimizer.last_gradients = self.discriminator.optimizer.blank_weights(discriminator_weighted_layers)
		generator_weighted_layers = self.generator.weighted_layers
		self.generator.optimizer.last_gradients = self.generator.optimizer.blank_weights(generator_weighted_layers)
		self.generator.optimizer.iterations = 0
		self.discriminator.optimizer.iterations = 0		
		discriminator_batch_size = self.discriminator.optimizer.batch_size
		generator_batch_size = self.generator.optimizer.batch_size
		if progress_bar:
			pb = ProgressBar(total = len(real_data)-1,start_message = "PROGRESS",total_chars = 100)
		discriminator_count = 0
		generator_count     = 0	
		discriminator_gradients = self.discriminator.optimizer.blank_weights(discriminator_weighted_layers)
		generator_gradients = self.generator.optimizer.blank_weights(generator_weighted_layers)
		for ind,data in enumerate(real_data):
			if progress_bar:
				pb.update()
			#train discriminator on real_data 
			discriminator_layers = self.discriminator.training_run(data)
			discriminator_loss_derivative = self.discriminator.loss.derivative_prev_layer(discriminator_layers[-1],real_data_output,batch_size = discriminator_batch_size)
			discriminator_gradients = self.discriminator.optimizer.derive_one_data(	discriminator_gradients,
																					self.discriminator.training_layers,
																					discriminator_loss_derivative,
																					discriminator_layers)
			#train discriminator on generated data, while also training generator
			generator_noise = self.noise(size = self.generator_input_shape) 
				
			generator_layers = self.generator.training_run(generator_noise)
			discriminator_layers = self.discriminator.training_run(generator_layers[-1])
			discriminator_loss_derivative = self.discriminator.loss.derivative_prev_layer(discriminator_layers[-1],generated_data_output,
											batch_size =discriminator_batch_size )
			discriminator_gradients,generator_last_derivative = self.discriminator.optimizer.derive_one_data(	discriminator_gradients,
																												self.discriminator.training_layers,
																												discriminator_loss_derivative,
																												discriminator_layers,last_layer_return = True)
			discriminator_count +=2
		
			generator_last_derivative = -generator_last_derivative
			generator_gradients = self.generator.optimizer.derive_one_data(	generator_gradients,
																			self.generator.training_layers,
																			generator_last_derivative,
																			generator_layers)
			generator_count += 1
			if discriminator_count >= self.discriminator.optimizer.batch_size or ind == len(real_data)-1: 
				self.discriminator.optimizer.descend(discriminator_gradients,discriminator_weighted_layers)		
				discriminator_gradients = self.discriminator.optimizer.blank_weights(discriminator_weighted_layers)
			if generator_count >= self.generator.optimizer.batch_size or ind == len(real_data)-1: 
				self.generator.optimizer.descend(generator_gradients,generator_weighted_layers)
				generator_gradients = self.generator.optimizer.blank_weights(generator_weighted_layers)

	def training_run(self,input_layer,use_generator = True):
		return self.generator.training_run(input_layer) if use_generator else self.discriminator.training_run(input_layer)

	def run_generator(self,noise = None):
		input_layer = noise if noise is not None else self.noise(self.generator_input_shape)
		return self.generator.run(input_layer)

	def run_discriminator(self,input_layer):
		return self.discriminator.run(input_layer)

class ConditionalGAN(GAN):
	def __init__(self,discriminator, generator,noise_shape,noise = np.random.random):
		super().__init__(discriminator,generator,noise)
		self.noise_shape = noise_shape if noise_shape is np.ndarray else np.array(noise_shape)
		self.condition_shape = self.generator_input_shape - self.noise_shape
		self.discriminator_prediction_shape = self.discriminator_input_shape-self.condition_shape	

	def train(self,real_data,conditions,real_data_output = np.array([0,1]),generated_data_output = np.array([1,0]),progress_bar = True): 
		discriminator_weighted_layers = self.discriminator.weighted_layers
		generator_weighted_layers = self.generator.weighted_layers
		self.discriminator.optimizer.startup(discriminator_weighted_layers)
		self.generator.optimizer.startup(generator_weighted_layers)
		discriminator_count = 0
		generator_count     = 0	
		if progress_bar:
			pb = ProgressBar(total = len(real_data),start_message = "PROGRESS",total_chars = 100)
		discriminator_gradients = self.discriminator.optimizer.blank_weights(discriminator_weighted_layers)
		generator_gradients = self.generator.optimizer.blank_weights(generator_weighted_layers)
		
		for ind,data in enumerate(real_data):
			if progress_bar:
				pb.update()
			#train discriminator on real_data 
			discriminator_layers = self.discriminator.training_run(np.concatenate((data,conditions[ind])))
			discriminator_loss_derivative = self.discriminator.loss.derivative_prev_layer(discriminator_layers[-1],real_data_output,batch_size = self.discriminator.optimizer.batch_size)
			discriminator_gradients = self.discriminator.optimizer.derive_one_data(	discriminator_gradients,
																					self.discriminator.training_layers,
																					discriminator_loss_derivative,
																					discriminator_layers)
			#train discriminator on generated data, while also training generator
			generator_noise = self.noise(size = self.noise_shape) 
			generator_input = np.concatenate((generator_noise,conditions[ind])) 
			generator_layers = self.generator.training_run(generator_input)
			discriminator_layers = self.discriminator.training_run(np.concatenate((generator_layers[-1],conditions[ind])))
			discriminator_loss_derivative = self.discriminator.loss.derivative_prev_layer(discriminator_layers[-1],generated_data_output,batch_size = self.discriminator.optimizer.batch_size)
			discriminator_gradients,generator_last_derivative =  self.discriminator.optimizer.derive_one_data(	discriminator_gradients,
																												self.discriminator.training_layers,
																												discriminator_loss_derivative,
																												discriminator_layers,
																												last_layer_return = True)
			generator_last_derivative = -generator_last_derivative[tuple([slice(None,dimension) for dimension in self.discriminator_prediction_shape])]
			generator_gradients = self.generator.optimizer.derive_one_data(	generator_gradients,
																			self.generator.training_layers,
																			generator_last_derivative,
																			generator_layers)
			discriminator_count +=2
			generator_count += 1
			if discriminator_count >= self.discriminator.optimizer.batch_size or ind == len(real_data)-1:
				self.discriminator.optimizer.descend(discriminator_gradients,discriminator_weighted_layers)		
				discriminator_gradients = self.discriminator.optimizer.blank_weights(discriminator_weighted_layers)
				discriminator_count = 0
			if generator_count >= self.generator.optimizer.batch_size or ind == len(real_data) -1:
				self.generator.optimizer.descend(generator_gradients,generator_weighted_layers)
				generator_gradients = self.generator.optimizer.blank_weights(generator_weighted_layers)
				generator_count     = 0	

	def run_generator(self,condition,noise = None):
		noise = self.noise(size = self.noise_shape) if noise is None else noise
		return self.generator.run(np.concatenate((noise,condition)))

	def run_discriminator(self,prediction,condition):
		return self.discriminator.run(np.concatenate((prediction,condition)))
