import math
import numpy as np
import random
from hidden import Hidden

class Convolution(Hidden):
	def __init__(self,input_shape = None,num_filters = 1,filter_size = (2) , stride = (2)):
		self.input_shape = [input_shape] if not hasattr(input_shape,"__iter__") else input_shape
		self.padded_shape = self.get_padded_shape() if self.pad else self.input_shape
		self.individual_input_shape = self.padded_shape[1:]
		super().__init__(input_shape,self.compute_output_shape(self.padded_shape))
		self.dimension = len(self.filter_size)
		self.config["dimension"] = self.dimension + 1

		self.stride = stride if hasattr(stride,"__iter__") else [stride]	
		self.filter_size = list(filter_size) if hasattr(filter_size,"__iter__") else [filter_size]
		
		self.num_input_channels = input_shape[0]
		self.shape = [num_filters,self.num_input_channels]+ self.filter_size  # needs to be implemented
		input_num_neurons = 1
		output_num_neurons = 1
		for neuron in self.input_shape:
			input_num_neurons *= neuron
		for neuron in self.output_shape:
			output_num_neurons *= neuron
		
		starting_value = -((6/(input_num_neurons + output_num_neurons))**0.5)
		ending_value = -starting_value
		#starting_value = -1
		#ending_value = 1
		self.weights = np.random.uniform(starting_value,ending_value,size = self.shape)
	
	def get_padded_shape(self):#NEEDS WORK
		padded_shape = list(self.input_shape)
		for ind,dimension in enumerate(padded_shape[1:]):
			#offset =  (dimension - self.filter_size[ind]) %  (self.stride[ind]) - 1
			#padded_dimension = max(self.filter_size[ind],dimension + offset) 
			current = self.filter_size[ind]
			start = self.filter_size[ind]
			while dimension > start:
				current += self.stride[ind]
				dimension -= self.stride[ind] 
		
			#mimimum = max(start,dimension)
			#padded_dimension = max(0,math.ceil((dimension - start + 1)/self.stride[ind])) + start
			padded_dimension = current	 
			padded_shape[ind+1] = padded_dimension
		#print(padded_shape)
		return padded_shape
	
	def compute_output_shape(self,padded_shape):
		output_shape = [self.num_filters]
		for ind,dimension in enumerate(padded_shape[1:]):
			offset =  (dimension - self.filter_size[ind]) % self.stride[ind]
			dimension_shape = math.ceil((dimension - self.filter_size[ind] + 1) / self.stride[ind])
			output_shape.append(dimension_shape)
		return np.array(output_shape)

	def run(self,input_layer):
		output_layer = []
		for filter in self.weights:
			output_layer.append(self.apply_one_filter(filter,input_layer))
		return np.array(output_layer) 

	def derivative(self,input_layer,output_layer_derivative,**kwargs):
		layer_derivative = []
		for filter in self.weights:
			layer_derivative.append(self.derivative_one_filter(filter,input_layer,output_layer_derivative))
		return np.array(layer_derivative)

	def derivative_prev_layer(self,input_layer,output_layer_derivative,**kwargs):
		return # needs to be implemented

	def blank(self):
		return np.zeros(self.weights.shape) 
	
	def descend(self,derivatives):
		self.weights -= derivatives
		#self.weights = np.clip(self.weights-derivatives,-1,1)
	
class Convolution1D(Convolution):#NEEDS TESTING
	def __init__(self,num_filters =1, filter_size = (2), stride = (2),pad = True):
		self.config = {"dimension":2,"type":"HIDDEN"}
		self.num_filters = num_filters 
		self.filter_size = filter_size if hasattr(filter_size,"__iter__") else [filter_size]
		self.stride = stride if hasattr(stride,"__iter__") else [stride]
		self.pad = pad
	
	def init_input_shape(self,input_shape):
		super().__init__(input_shape,self.num_filters,self.filter_size,self.stride)
		self.filter_number_weights = self.filter_size[0] * self.num_input_channels
		self.number_weights = self.filter_number_weights * self.num_filters
 
	def apply_one_filter(self,filter,input_layer):	
		padded_input_layer = np.zeros(self.padded_shape)#can optimize if pad = false
		padded_input_layer[:input_layer.shape[0],:input_layer.shape[1]] = input_layer
		output_channel = np.zeros(self.output_shape[1:])
		x_pos = 0
		for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
			slice = np.array(padded_input_layer[:,x_pos:x_pos + self.filter_size[0]]) 
			multiplied_average = slice*filter
			multiplied_sum_average = np.sum(multiplied_average)
			output_channel[x_pos//self.stride[0]] = multiplied_sum_average
			x_pos += self.stride[0]
		return output_channel#it will be made into a numpy  array in the run function
	
	def derivative(self,input_layer,output_layer_derivative,**kwargs):
		padded_input_layer = np.zeros(self.padded_shape)
		padded_input_layer[:input_layer.shape[0],:input_layer.shape[1]] = input_layer
		layer_derivative = []
		for filter_num,filter in enumerate(self.weights):
			filter_derivative = np.zeros(filter.shape)
			for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
				slice = padded_input_layer[:,x_pos:x_pos + self.filter_size[0]]
				filter_derivative += slice * output_layer_derivative[filter_num,x_pos//self.stride[0]]
			layer_derivative.append(filter_derivative)
		return np.array(layer_derivative)

	def derivative_prev_layer(self,input_layer,output_layer_derivative,**kwargs):
		prev_layer_derivative = np.zeros(self.padded_shape)
		for filter_num,filter in enumerate(self.weights):
			for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
				prev_layer_derivative[:,x_pos:x_pos + self.filter_size[0]] += output_layer_derivative[filter_num,x_pos//self.stride[0]] * filter
		return prev_layer_derivative[:input_layer.shape[0],:input_layer.shape[1]] 
		
 
class Convolution2D(Convolution):#NEEDS TESTING
	def __init__(self,num_filters =1, filter_size = (2,2), stride = (2,2),pad = True):
		self.config = {"dimension":3,"type":"HIDDEN"}
		self.num_filters = num_filters
		self.filter_size = filter_size if hasattr(filter_size,"__iter__") else [filter_size]
		self.stride = stride if hasattr(stride,"__iter__") else [stride]
		self.pad = pad
	
	def init_input_shape(self,input_shape):
		super().__init__(input_shape,self.num_filters,self.filter_size,self.stride)
		self.filter_number_weights = self.filter_size[0] * self.filter_size[1] * self.num_input_channels
		self.number_weights = self.filter_number_weights * self.num_filters

	def apply_one_filter(self,filter,input_layer):
		padded_input_layer = np.zeros(self.padded_shape)
		padded_input_layer[:input_layer.shape[0],:input_layer.shape[1],:input_layer.shape[2]] = input_layer
		output_channel = np.zeros(self.output_shape[1:])
		for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
			for y_pos in range(0,self.individual_input_shape[1]-self.filter_size[1]+1,self.stride[1]):
				slice = np.array(padded_input_layer[:,x_pos:x_pos + self.filter_size[0],y_pos:y_pos + self.filter_size[1]])
				multiplied_average = slice*filter
				multiplied_sum_average = np.sum(multiplied_average)
				output_channel[x_pos//self.stride[0],y_pos//self.stride[1]] = multiplied_sum_average
		#print("CONV2D:",output_channel,"\n",input_layer,"\n",filter,"\n",padded_input_layer,"\n",self.padded_shape)
		return output_channel
	
	def derivative(self,input_layer,output_layer_derivative,**kwargs):
		padded_input_layer = np.zeros(self.padded_shape)
		padded_input_layer[:input_layer.shape[0],:input_layer.shape[1],:input_layer.shape[2]] = input_layer
		layer_derivative = []
		for filter_num,filter in enumerate(self.weights): 
			filter_derivative = np.zeros(filter.shape)
			for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
				for y_pos in range(0,self.individual_input_shape[1]-self.filter_size[1]+1,self.stride[1]):
					slice = padded_input_layer[:,x_pos:x_pos + self.filter_size[0],y_pos:y_pos + self.filter_size[1]]
					filter_derivative += slice * output_layer_derivative[filter_num,x_pos//self.stride[0],y_pos//self.stride[1]]
			layer_derivative.append(filter_derivative)
		return np.array(layer_derivative)
	
	def derivative_prev_layer(self,input_layer,output_layer_derivative,**kwargs):
		prev_layer_derivative = np.zeros(self.padded_shape)
		for filter_num,filter in enumerate(self.weights):
			for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
				for y_pos in range(0,self.individual_input_shape[1] - self.filter_size[1] + 1, self.stride[1]):
					prev_layer_derivative[:,x_pos:x_pos + self.filter_size[0],y_pos:y_pos + self.filter_size[1]] += output_layer_derivative[filter_num,x_pos//self.stride[0],y_pos//self.stride[1]] * filter
		return prev_layer_derivative[:input_layer.shape[0],:input_layer.shape[1],:input_layer.shape[1]] 
		'''
		padded_input_layer = np.zeros(self.padded_shape)
		padded_input_layer[:input_layer.shape[0],:input_layer.shape[1],:input_layer.shape[2]] = input_layer
		prev_layer_derivative = np.zeros(padded_input_layer.shape)
		x_pos = 0
		while x_pos + self.filter_size[0] <= self.individual_input_shape[0]:
			y_pos = 0
			while y_pos + self.filter_size[1] <= self.individual_input_shape[1]:
				for ind,filter in enumerate(self.weights):
					prev_layer_derivative[:,x_pos:x_pos + self.filter_size[0],y_pos:y_pos + self.filter_size[1]] += filter * output_layer_derivative[ind,x_pos//self.stride[0],y_pos//self.stride[1]]
				y_pos += self.stride[1] 
			x_pos += self.stride[0]
		return prev_layer_derivative[:input_layer.shape[0],:input_layer.shape[1],:input_layer.shape[2]] 
		'''

class Convolution3D(Convolution):
	def __init__(self,num_filters =1, filter_size = (2,2,2), stride = (2,2,2),pad = True):
		self.config = {"dimension":4,"type":"HIDDEN"}
		self.num_filters = num_filters
		self.filter_size = filter_size if hasattr(filter_size,"__iter__") else [filter_size]
		self.stride = stride if hasattr(stride,"__iter__") else [stride]
		self.pad = pad
	
	def init_input_shape(self,input_shape):
		super().__init__(input_shape,self.num_filters,self.filter_size,self.stride)
		self.filter_number_weights = self.filter_size[0] * self.filter_size[1] * self.filter_size[2] *  self.num_input_channels
		self.number_weights = self.filter_number_weights * self.num_filters
		
	def apply_one_filter(self,filter,input_layer):
		output_channel = []
		x_pos = 0
		while x_pos + self.filter_size[0] <= self.individual_input_shape[0]:
			y_pos = 0
			output_channel.append([])
			while y_pos + self.filter_size[1] <= self.individual_input_shape[1]:
				z_pos = 0
				output_channel[-1].append([])
				while z_pos + self.filter_size[2] <= self.individual_input_shape[1]:
					slice = np.array(input_layer[input_num][x_pos:x_pos + self.filter_size[0],y_pos:y_pos + self.filter_size[1],z_pos:z_pos + self.filter_size[2]]) 
					multiplied_average = slice*filter
					multiplied_sum_average = np.sum(multiplied_average)
					output_channel[-1][-1].append(multiplied_sum_average)
					z_pos += self.stride[2]
				y_pos += self.stride[1]
			x_pos += self.stride[0]		
		return output_channel#it will be made into a numpy array in the run function
if __name__ == "__main__":
	from loss import *
	a = np.array([[4.0,5.0,-2.0,3.5,10.0,-2.0]])
	con = Convolution1D()
	con.init_input_shape((1,6))
	bsize=  100
	print(con.weights)
	for i in range(bsize):
		pre_error = con.run(a)
		target = [2.75,3.125,-4]
		after_error = mean_squared_loss(pre_error,np.array(target))
		derivative1 = derivative_mean_squared_loss(pre_error,np.array(target),batch_size = 50)
		derivative2 = con.derivative_prev_layer(a,derivative1)
		derivative3 = con.derivative(a,derivative1)
		#a -= derivative2
		con.descend(derivative3)
	print(con.weights)
	con.weights = [[[-0.25,0.75]]]	
	print(con.run(a))
