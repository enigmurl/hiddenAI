from convertors import *
import loss
import pickle
import regulizers
from progress_bar import ProgressBar

class Sequential:
	def __init__(self,input_shape,*layers,loss = loss.MeanSquaredLoss(),regulizer = regulizers.Lasso(1) ):
		self.layers = self.compile_layers(input_shape,layers)
		self.regulizer = regulizer
		self.loss = loss
	
	def compile_layers(self,input_shape,layers):
		output_shape = input_shape if hasattr(input_shape,"__iter__") else [input_shape]
		new_layers = []
		self.weighted_layers = []
		for ind,layer in enumerate(layers):#include convertors and check if the input size does not match the next output size
			if layer.config["dimension"] != len(output_shape) and layer.config["dimension"] != "ANY":
				if ind == 0:
					raise ValueError( "Required input layer dimension is: " + str(layer.config["dimension"])) 
				convertor = ConvertorND1D(output_shape)# make it in future to convert to something that is of any dimension
				new_layers.append(convertor)
				output_shape = convertor.output_shape
			layer.init_input_shape(output_shape)
			output_shape = layer.output_shape if hasattr(layer.output_shape,"__iter__") else [layer.output_shape]
			new_layers.append(layer)
			if hasattr(layer,"weights"):
				self.weighted_layers.append(layer)
		return new_layers

	
	def __iter__(self):
		return self.layers.__iter__()

	def __getitem__(self,ind):
		return self.layers[ind]
	
	def __repr__(self):
		string = ""
		for layer in self.layers:
			string += str(type(layer))
		return string

	def save_to_file(self,filename):
		with open(filename,"wb") as f:
			for layer in self.weighted_layers:
				pickle.dump(layer.weights,f)

	def open_from_file(self,filename):
		with open(filename,"rb") as f:
			for layer in self.weighted_layers:
				layer.weights = pickle.load(f)
	
	def run(self,input_layer,full_return = False):
		current_value = input_layer
		if full_return:
			all_values = [input_layer]	
		for ind,layer in enumerate(self.layers):
			current_value = layer.run(current_value)
			if full_return:
				all_values.append(current_value)
		if full_return:
			return all_values		
		return current_value		

	def batch_data(self,training_data,batch_size):
		batched_data = []
		if len(training_data)%batch_size !=0:
			for ind in range (len(training_data)//batch_size +1):
				batched_data.append(training_data[ind*batch_size:batch_size*(ind+1)])
		else:
			for ind in range (len(training_data)//batch_size):
				batched_data.append(training_data[ind*batch_size:batch_size*(ind+1)])
		return batched_data
	
	def descend(self,weight_gradients): #weight_gradient is a dictionary of {layer:gradient,layer2:gradient2....}
		for ind,gradient in enumerate(weight_gradients):
			self.weighted_layers[ind].descend(gradient)
	
	def compute_loss_derivative(self,all_layers,expected_output,loss_derivative_function,batch_size = 1):
		end_derivative = loss_derivative_function(all_layers[-1],expected_output,batch_size=batch_size)
		return end_derivative	

	def derive_one_data(self,weight_gradients,current_derivative,all_layers):
		weighted_num = -1
		for ind,layer in enumerate(self.layers[::-1]):
			input_layer = all_layers[len(self.layers)-ind-1]
			if hasattr(layer,"weights"): #if it has weights
				layer_derivative = layer.derivative(input_layer,current_derivative)
				regularized_layer_derivative = self.regulizer.apply_derivative(layer.weights,layer_derivative)
				weight_gradients[weighted_num] += regularized_layer_derivative
				weighted_num -= 1
			if ind != len(self.layers)-1:
				current_derivative = layer.derivative_prev_layer(input_layer,current_derivative)
		return weight_gradients

	def derive_and_descend_one_data(self,current_derivative,all_layers):
		for ind,layer in enumerate(self.layers[::-1]):
			input_layer = all_layers[len(self.layers)-ind-1]
			if hasattr(layer,"weights"): #if it has weights
				layer_derivative = layer.derivative(input_layer,current_derivative)
				regularized_layer_derivative = self.regulizer.apply_derivative(layer.weights,layer_derivative)
				layer.descend(regularized_layer_derivative)
			if ind != len(self.layers)-1:
				current_derivative = layer.derivative_prev_layer(input_layer,current_derivative)

		
	def blank_weights(self):		
		weight_gradients = []
		for layer in self.weighted_layers:
			weight_gradients.append(layer.blank())
		return weight_gradients

	def gradient_descent(self,input_data,expected_outputs,epoch = 1, learning_rate = 0.001,print_epochs = True):
		loss_function_derivatives = self.loss.derivative_prev_layer
		
		training_data = list(zip(input_data,expected_outputs))

		for epoch_num in range(epoch):
			weight_gradients = self.blank_weights()
			progress_bar = ProgressBar(start_message ="EPOCH NUM " + str(epoch_num) + " ",total = len(training_data), total_chars = 100)
			for data in training_data:
				expected = data[1]
				all_layers = self.run(data[0],full_return = True)	
				current_derivative = learning_rate * self.compute_loss_derivative(all_layers,expected,loss_function_derivatives,batch_size = len(training_data))
				weight_gradients = self.derive_one_data(weight_gradients,current_derivative,all_layers)	#see if we need to make it self	
				if print_epochs:
					progress_bar.update()
			self.descend(weight_gradients)



		

	def stochastic_gradient_descent(self,input_data,expected_outputs,epoch = 1 ,learning_rate = 1,print_epochs = True):
		loss_function_derivatives = self.loss.derivative_prev_layer

		training_data = list(zip(input_data,expected_outputs))

		for epoch_num in range(epoch):
			if print_epochs:
				progress_bar = ProgressBar(start_message ="EPOCH NUM " + str(epoch_num) + " ",total = len(training_data), total_chars = 100)
			for data in training_data:
				expected = data[1]
				all_layers = self.run(data[0],full_return = True)	
				current_derivative = learning_rate * self.compute_loss_derivative(all_layers,expected,loss_function_derivatives,batch_size = len(training_data))
				self.derive_and_descend_one_data(current_derivative,all_layers)	#see if we need to make it self	
				if print_epochs:
					progress_bar.update()
			

			if print_epochs:
				print("EPOCH#",epoch_num+1, "OUT OF",epoch)

	def batch_gradient_descent(self,input_data,expected_outputs,epoch = 1, batch_size = 5,learning_rate = 1,print_epochs = True):
		loss_function_derivatives = self.loss.derivative_prev_layer

		training_data = list(zip(input_data,expected_outputs))
		batched_data = self.batch_data(training_data,batch_size)

		for epoch_num in range(epoch):
			if print_epochs:
				progress_bar = ProgressBar(start_message ="EPOCH NUM " + str(epoch_num) + " ",total = len(training_data), total_chars = 100)
			for batch_num,batch in enumerate(batched_data):
				weight_gradients = self.blank_weights()			
				for data in batch:
					expected = data[1]
					all_layers = self.run(data[0],full_return = True)	
					current_derivative = learning_rate * self.compute_loss_derivative(all_layers,expected,loss_function_derivatives,batch_size = len(training_data))
					weight_gradients = self.derive_one_data(weight_gradients,current_derivative,all_layers)	#see if we need to make it self	
					if print_epochs:
						progress_bar.update()
			
				self.descend(weight_gradients)
			if print_epochs:
				print("EPOCH#",epoch_num+1, "OUT OF",epoch)

if __name__ == "__main__":
	import time
	from activations import *
	from convolution import *
	from pooling import *
	from hidden import *
	activation = Sigmoid() 
	a = Sequential((1,9),
				Convolution1D(num_filters = 3,filter_size = (2),stride = (2)),
				MaxPooling1D(),
				Bias(),
				Convolution1D(num_filters = 2,filter_size = (2),stride = (2)),
				MaxPooling1D(),
				Bias(),
				FullyConnected(2),
				Bias(),
				Sigmoid())
	
	trainingData = [
	np.array([[1,0,1,0,1,0,1,0,1]]),
	np.array([[0,1,0,1,0,1,0,1,0]]),
	np.array([[1,1,1,1,0,1,1,1,1]]),
	np.array([[0,1,1,0,1,0,1,0,1]]),
	np.array([[1,1,1,1,0,1,0,1,0]]),
	np.array([[1,0,1,0,1,0,1,0,0]]),
	np.array([[0,1,0,1,0,1,1,1,1]]),
	np.array([[0,0,1,0,1,0,1,0,1]])
	]
	labels = [np.array([0,1]),np.array([1,0]),np.array([1,0]),np.array([0,1]),np.array([1,0]),np.array([0,1]),np.array([1,0]),np.array([0,1])]
	startTime = time.time()
	numData = 100
	for i in range(10):
		for ind,data in enumerate(trainingData):
			b = a.run(data)
			print("RESULT 2:",b,"EXPECTED RESULT",labels[ind])
		a.batch_gradient_descent(trainingData,labels,epoch = numData,batch_size= 16,print_epochs = False)
		print("\n")
	print("TOTAL TIME:",time.time()-startTime)
