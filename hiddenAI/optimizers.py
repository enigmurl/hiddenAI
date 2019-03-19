from hiddenAI.progress_bar import ProgressBar
import hiddenAI.learning_rates as learning_rates
import random

class BatchGradientDescent:
	def __init__(self,batch_size = 5,momentum = 0,learning_rate = learning_rates.ConstantLearningRate(0.1)):
		self.seed = random.random()
		self.momentum = momentum 
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.iterations = 0
	
	def descend(self,weight_gradients,weighted_layers): #
		current_learning_rate = self.learning_rate.next_learning_rate(self.iterations)
		gradient_with_momentum = [self.momentum * (self.last_gradients[layer_num]) +  current_learning_rate * layer_gradients for layer_num,layer_gradients in enumerate(weight_gradients)]
		for ind,gradient in enumerate(gradient_with_momentum):
			weighted_layers[ind].descend(gradient)
		self.last_gradients = gradient_with_momentum
	
	def compute_loss_derivative(self,all_layers,expected_output,loss_derivative_function,batch_size = 1):
		end_derivative = loss_derivative_function(all_layers[-1],expected_output,batch_size=batch_size)
		return end_derivative	

	def derive_one_data(self,weight_gradients,training_layers,current_derivative,all_layers,last_layer_return = False):
		weighted_num = -1
		return_gradients  = weight_gradients[:]
		for ind,layer in enumerate(training_layers[::-1]):
			input_layer = all_layers[len(training_layers)-ind-1]
			if hasattr(layer,"weights"): #if it has weights
				layer_derivative = layer.derivative(input_layer,current_derivative)
				return_gradients[weighted_num] += layer_derivative 
				weighted_num -= 1
			if ind != len(training_layers)-1:
				current_derivative = layer.derivative_prev_layer(input_layer,current_derivative)
			elif last_layer_return:
				return return_gradients,layer.derivative_prev_layer(input_layer,current_derivative)
		return return_gradients

	def batch_data(self,training_data,batch_size):
		batched_data = []
		if len(training_data)%batch_size !=0:
			for ind in range (len(training_data)//batch_size +1):
				batched_data.append(training_data[ind*batch_size:batch_size*(ind+1)])
		else:
			for ind in range (len(training_data)//batch_size):
				batched_data.append(training_data[ind*batch_size:batch_size*(ind+1)])
		return batched_data
	
	def blank_weights(self,weighted_layers):		
		weight_gradients = []
		for layer in weighted_layers:
			weight_gradients.append(layer.blank())
		return weight_gradients
	
	#MAY WANT TO IMPLEMENT A derive one batch method
	
	def train(self,input_data,expected_outputs,epoch,training_layers,run_function,loss_function_derivative,regulizer,print_epochs = False):
		weighted_layers = []
		for layer in training_layers:
			if hasattr(layer,"weights"):
				weighted_layers.append(layer)		

		training_data = list(zip(input_data,expected_outputs))
		batched_data = self.batch_data(training_data,self.batch_size)
		self.last_gradients = self.blank_weights(weighted_layers) 
		self.iterations = 0

		for epoch_num in range(epoch):
			if print_epochs:
				progress_bar = ProgressBar(start_message ="EPOCH NUM " + str(epoch_num) + " ",total = len(training_data), total_chars = 100)
			for batch_num,batch in enumerate(batched_data):
				weight_gradients = self.blank_weights(weighted_layers)			
				for data in batch:
					expected = data[1]
					all_layers = run_function(data[0])
					current_derivative = self.compute_loss_derivative(all_layers,expected,loss_function_derivative,batch_size = len(batch))
					weight_gradients = self.derive_one_data(weight_gradients,training_layers,current_derivative,all_layers)	#see if we need to make it self	
					if print_epochs:
						progress_bar.update()
				#weight_gradients = [regulizer.apply_derivative(self.weighted_layers[layer_num].weights,layer_derivative) for layer_num,layer_derivative in enumerate(weight_gradients)] 
				self.descend(weight_gradients,weighted_layers)
			self.iterations += 1
		return training_layers		
