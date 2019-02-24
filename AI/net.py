from layers1D import *
import costs
import pickle

class NeuralNet:
	def __init__(self,*layers):
		self.layers = list(layers)
		self.cost_function_derivatives = {costs.mean_squared_cost:costs.derivative_mean_squared_cost}
	
	def add_cost_function(self,cost_function,derivative_cost_function):
		self.cost_function_derivatives[cost_function] = derivative_cost_function
	
	def __iter__(self):
		return self.layers.__iter__()

	def __getitem__(self,ind):
		return self.layers[ind]

	def add(self,layer):
		self.layers.append(layer)

	def save_to_file(self,filename):
		with open(filename,"wb") as f:
			for layer in self.layers:
				if layer.config["type"] == "HIDDEN":
					pickle.dump(layer.weights,f)

	def open_from_file(self,filename):
		with open(filename,"rb") as f:
			for layer in self.layers:
				if layer.config["type"] == "HIDDEN":
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
				batched_data.append(training_data[ind*batch_size:batch_size*(i+1)])
		else:
			for ind in range (len(training_data)//batch_size):
				batched_data.append(training_data[ind*batch_size:batch_size*(ind+1)])
		return batched_data
	
	def descend(self,weight_gradients): #weight_gradient is a dictionary of {layer:gradient,layer2:gradient2....}
		for layer,gradient in weight_gradients.items():
			layer.descend(gradient)
	
	def compute_loss_derivative(self,all_layers,expected_output,cost_derivative_function,batch_size = 1):
		end_derivative = cost_derivative_function(all_layers[-1],expected_output,batch_size=batch_size)
		return end_derivative	

	def derive_one_data(self,weight_gradients,current_derivative,all_layers):
		for ind,layer in enumerate(self.layers[::-1]):
			input_layer = all_layers[len(self.layers)-ind-1]
			if hasattr(layer,"weights"): #if it has weights
				layer_derivative = layer.derivative(input_layer,current_derivative)
				weight_gradients[layer] += layer_derivative
			if ind != len(self.layers)-1:
				current_derivative = layer.derivative_prev_layer(input_layer,current_derivative)

		return weight_gradients

	def blank_weights(self):		
		weight_gradients= {}
		for layer in self.layers:
			if hasattr(layer,"weights"):
				weight_gradients[layer] = layer.blank() 
		return weight_gradients

	def stochastic_descent(self,input_data,expected_outputs,epoch = 1, batch_size = 5,learning_rate = 1,cost_function = costs.mean_squared_cost):
		cost_function_derivatives = self.cost_function_derivatives[cost_function]

		training_data = list(zip(input_data,expected_outputs))
		batched_data = self.batch_data(training_data,batch_size)

		for epoch_num in range(epoch):
			for batch_num,batch in enumerate(batched_data):
				weight_gradients = self.blank_weights()			
				for data in batch:
					expected = data[1]
					all_layers = self.run(data[0],full_return = True)	
					current_derivative = self.compute_loss_derivative(all_layers,expected,cost_function_derivatives,batch_size = len(batch))
					weight_gradients = self.derive_one_data(weight_gradients,current_derivative,all_layers)	#see if we need to make it self	
			
				self.descend(weight_gradients)

			print("EPOCH#",epoch_num, "OUT OF",epoch-1)

if __name__ == "__main__":
	import time
	a = NeuralNet(FullyConnected(9,5),Bias(5,5),Sigmoid(),FullyConnected(5,2),Bias(2,2),Sigmoid())
	b = a.run(np.array([0,1,0,1,0,1,0,1,0]))

	print("RESULT:",b)
	trainingData = [
	np.array([1,0,1,0,1,0,1,0,1]),
	np.array([0,1,0,1,0,1,0,1,0]),
	np.array([1,1,1,1,0,1,1,1,1]),
	np.array([0,1,1,0,1,0,1,0,1]),
	np.array([1,1,1,1,0,1,0,1,0]),
	np.array([1,0,1,0,1,0,1,0,0]),
	np.array([0,1,0,1,0,1,1,1,1]),
	np.array([0,0,1,0,1,0,1,0,1])
	]
	labels = [np.array([0,1]),np.array([1,0]),np.array([1,0]),np.array([0,1]),np.array([1,0]),np.array([0,1]),np.array([1,0]),np.array([0,1])]
	startTime = time.time()
	numData = 10000
	a.stochastic_descent(trainingData,labels,epoch = numData, batch_size = 8)
	b = a.run(np.array([0,1,0,1,0,1,0,1,0]))
	for ind,data in enumerate(trainingData):
		#b = a.run(data,debug=True,fullReturn = True)
		b = a.run(data)
		print("RESULT 2:",b,"EXPECTED RESULT",labels[ind])
	print("TOTAL TIME:",time.time()-startTime)
