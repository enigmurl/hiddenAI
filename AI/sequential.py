from layers.convertors import *
import pickle
import loss
import regulizers
import optimizers

class Sequential:
	def __init__(self,input_shape,*layers,optimizer,loss = loss.MeanSquaredLoss(),regulizer = regulizers.Ridge(1/10)):
		self.layers = self.compile_layers(input_shape,layers)
		self.regulizer = regulizer
		self.loss = loss
		self.optimizer = optimizer
	
	def compile_layers(self,input_shape,layers):
		output_shape = input_shape if hasattr(input_shape,"__iter__") else [input_shape]
		new_layers = []
		self.weighted_layers = []
		self.training_layers = []
		print("<NeuralNet>")
		training_layer_types = ("DROPOUT")#JUST DROPOUT FOR NOW
		for ind,layer in enumerate(layers):#include convertors and check if the input size does not match the next output size
			if layer.config["dimension"] != len(output_shape) and layer.config["dimension"] != "ANY":
				if ind == 0:
					raise ValueError( "Required input layer dimension is: " ,layer.config["dimension"]) 
				convertor = ConvertorND1D(output_shape)# make it in future to convert to something that is of any dimension
				new_layers.append(convertor)
				self.training_layers.append(convertor)
				output_shape = convertor.output_shape
			layer.init_input_shape(output_shape)
			output_shape = layer.output_shape if hasattr(layer.output_shape,"__iter__") else [layer.output_shape]
			print(layer,output_shape)
			self.training_layers.append(layer)
			if layer.config["type"] not in training_layer_types:
				new_layers.append(layer)
			if hasattr(layer,"weights"):# SEEE IF THIS should be inside the "if layer.config["type"] statement 
				self.weighted_layers.append(layer)
		print("\n")
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
	
	def training_run(self,input_layer):
		current_value = input_layer
		all_values = [input_layer]	
		for ind,layer in enumerate(self.training_layers):
			current_value = layer.run(current_value)
			all_values.append(current_value)
		return all_values		
		

	def train(self,input_data,expected_outputs,epoch = 1,print_epochs = True):
		loss_function_derivative = self.loss.derivative_prev_layer
		self.training_layers = self.optimizer.train(input_data,expected_outputs,epoch,self.training_layers,self.training_run,loss_function_derivative,self.regulizer,print_epochs = print_epochs)
		return


if __name__ == "__main__":
	import time
	from layers.activations import *
	from layers.convolution import *
	from layers.pooling import *
	from layers.main_layers import *
	from layers.dropout import *
	from loss import *
	from layers.convertors import *
	import optimizers
	import learning_rates
	sig = Sigmoid()
	sig.config["dimension"] = 1
	lrning_rate = learning_rates.ConstantLearningRate(0.1)
	a = Sequential((1,3,3),
				Convolution2D(num_filters = 3,filter_size = (2,2),stride = (2,2)),
				Bias(),
				ReLU(),
				#Dropout(0.2),
				MaxPooling2D(pooling_size = [2,2]),
				Convolution2D(num_filters = 2,filter_size = (3,3),stride = (2,2)),
				Bias(),
				ReLU(),
				##ELU(),
				#Dropout(0.5),
				MaxPooling2D(pooling_size  = [2,2]),
				FullyConnected(2),
				Bias(),
				Softmax(),optimizer = optimizers.BatchGradientDescent(batch_size = 8,learning_rate = lrning_rate))#,loss = loss.CrossEntropy())
	print(a.layers[0].weights,a.layers[4].weights)
	time.sleep(1)
	'''
	a = Sequential((9),
				FullyConnected(6),
				Bias(),
				Sigmoid(),
				FullyConnected(2),
				Bias(),
				Sigmoid())
	'''
	trainingData =np.array( [
	[[[1,0,1],[0,1,0],[1,0,1]]],
	[[[0,1,0],[1,0,1],[0,1,0]]],
	[[[1,1,1],[1,0,1],[1,1,1]]],
	[[[0,1,1],[0,1,0],[1,0,1]]],
	[[[1,1,1],[1,0,1],[0,1,0]]],
	[[[1,0,1],[0,1,0],[1,0,0]]],
	[[[0,1,0],[1,0,1],[1,1,1]]],
	[[[0,0,1],[0,1,0],[1,0,1]]]
	])
	print(a.run(trainingData[0]))
	labels = [np.array([0,1]),np.array([1,0]),np.array([1,0]),np.array([0,1]),np.array([1,0]),np.array([0,1]),np.array([1,0]),np.array([0,1])]
	startTime = time.time()
	numData = 100
	for i in range(10):
		for ind,data in enumerate(trainingData):
			b = a.run(data)
			print("RESULT:",i,b,"EXPECTED RESULT",labels[ind])
		a.train(trainingData,labels,epoch = numData,print_epochs = False)
		print("\n")
	for ind,data in enumerate(trainingData):
		print("_______ \n")
		for ind2,lyer in enumerate(a.training_run(data)[6:]):
			print("LAYER:",a.training_layers[ind2+5], "\n",lyer)
	print("TOTAL TIME:",time.time()-startTime)
