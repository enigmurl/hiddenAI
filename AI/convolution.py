import math
import numpy as np
import random
from hidden import Hidden

class Convolution(Hidden):
	def __init__(self,input_shape = None,num_filters = 1,filter_size = (2) , stride = (2)):
		if not hasattr(input_shape,"__iter__"):
			input_shape = [input_shape]
		super().__init__(input_shape,self.compute_output_shape(input_shape))
		self.dimension = len(self.filter_size)
		self.config["dimension"] = self.dimension + 1

		self.stride = stride if hasattr(stride,"__iter__") else [stride]	
		self.filter_size = list(filter_size) if hasattr(filter_size,"__iter__") else [filter_size]
		
		self.num_input_channels = input_shape[0]
		self.shape = [num_filters,self.num_input_channels]+ self.filter_size  # needs to be implemented
		self.weights = 2*np.random.random(self.shape)-1
		self.padded_shape = self.get_padded_shape() if self.pad else input_shape
		self.individual_input_shape = self.padded_shape[1:]
	
	def get_padded_shape(self):
		padded_shape = list(self.input_shape)
		for ind,dimension in enumerate(padded_shape[1:]):
			offset =  (dimension-self.filter_size[ind]) % self.stride[ind]
			padded_shape[ind+1] = dimension + offset
		return padded_shape
	
	def compute_output_shape(self,input_shape):
		output_shape = [self.num_filters]
		for ind,dimension in enumerate(input_shape[1:]):
			offset =  (dimension - self.filter_size[ind]) % self.stride[ind]
			dimension_shape = (dimension + offset) // self.stride[ind]
			output_shape.append(dimension_shape)
		return np.array(output_shape)

	def apply_one_filter(self,filter,input_layer):#should be overrided for optimization, but this will work for any dimension
		output = []
		current_place = [0] * self.dimension
		done = False
		while not done:
			done,current_place = self.apply_one_stride(current_place)
			output# needs to be completed	

	def apply_one_stride(self,current_place):
		for ind, dimension in enumerate(current_place):
			real_ind = len(current_place) - ind - 1
			if dimension + self.stride(real_ind) < self.input_shape(real_ind):
				current_place[real_ind] += self.stride(real_ind)
				break
		else:
			return True , 0
		return False, current_place


	def run(self,input_layer):
		output_layer = []
		for filter in self.weights:
			output_layer.append(self.apply_one_filter(filter,input_layer))
		return np.array(output_layer) 

	def derivative(self,input_layer,output_layer_derivative):
		return self.blank()
		layer_derivative = []
		for filter in self.weights:
			layer_derivative.append(self.derivative_one_filter(filter,input_layer,output_layer_derivative))
		return np.array(layer_derivative)

	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		return # needs to be implemented

	def blank(self):
		return np.zeros(self.weights.shape) 
	
	def descend(self,derivatives):
		pre_weights = self.weights - derivatives
		pre_weights[pre_weights < -1] = -1
		pre_weights[pre_weights > 1] =  1
		self.weights = pre_weights 
	
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
		output_channel = []
		x_pos = 0
		while x_pos + self.filter_size[0] <= self.individual_input_shape[0]:# SEE WHAT TO DO 
			slice = np.array(padded_input_layer[:,x_pos:x_pos + self.filter_size[0]]) 
			multiplied_average = slice*filter
			multiplied_sum_average = np.sum(multiplied_average)
			output_channel.append(multiplied_sum_average)
			x_pos += self.stride[0]
		return output_channel#it will be made into a numpy  array in the run function
	
	def derivative_one_filter(self,filter,input_layer,output_layer_derivative):
		padded_input_layer = np.zeros(self.padded_shape)
		padded_input_layer[:input_layer.shape[0],:input_layer.shape[1]] = input_layer
		layer_derivative = np.zeros(self.weights.shape[1:]) 
		x_pos = 0
		while x_pos + self.filter_size[0] <= self.individual_input_shape[0]:# SEE WHAT TO DO 
			slice = np.array(padded_input_layer[:,x_pos:x_pos + self.filter_size[0]])# * output_layer_derivative[:,x_pos:x_pos + self.filter_size[0]] 
			layer_derivative[x_pos:x_pos+self.filter_size[0]] += slice * output_layer_derivative[x_pos/self.stride[0]]
			x_pos += self.stride[0]
		return layer_derivative 

	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		padded_input_layer = np.zeros(self.padded_shape)
		padded_input_layer[:input_layer.shape[0],:input_layer.shape[1]] = input_layer
		prev_layer_derivative = np.zeros(padded_input_layer.shape)
		x_pos = 0
		while x_pos + self.filter_size[0] <= self.individual_input_shape[0]:# SEE WHAT TO DO 
			for ind,filter in enumerate(self.weights):
				prev_layer_derivative[:,x_pos:x_pos + self.filter_size[0]] += output_layer_derivative[ind,x_pos//self.stride[0]] * filter
			x_pos += self.stride[0]
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
		output_channel = []
		x_pos = 0
		while x_pos + self.filter_size[0] <= self.individual_input_shape[0]:
			y_pos = 0
			output_channel.append([])
			while y_pos + self.filter_size[1] <= self.individual_input_shape[1]:
				slice = np.array(padded_input_layer[:,x_pos:x_pos + self.filter_size[0],y_pos:y_pos + self.filter_size[1]])
				multiplied_average = slice*filter
				multiplied_sum_average = np.sum(multiplied_average)
				output_channel[-1].append(multiplied_sum_average)
				y_pos += self.stride[1]
			x_pos += self.stride[0]		
		return output_channel#it will be made into a numpy array in the run function	
	
	def derivative_one_filter(self,filter,input_layer,output_layer_derivative):
		padded_input_layer = np.zeros(self.padded_shape)
		padded_input_layer[:input_layer.shape[0],:input_layer.shape[1],:input_layer.shape[2]] = input_layer
		layer_derivative = np.zeros(filter.shape)
		x_pos = 0
		while x_pos + self.filter_size[0] <= self.individual_input_shape[0]:
			y_pos = 0
			while y_pos + self.filter_size[1] <= self.individual_input_shape[1]:
				slice = np.array(padded_input_layer[:,x_pos:x_pos + self.filter_size[0]])# * output_layer_derivative[:,x_pos:x_pos + self.filter_size[0]] 
				layer_derivative[x_pos:x_pos+self.filter_size[0],y_pos:y_pos + self.filter_size[1]] += slice * output_layer_derivative[x_pos/self.stride[0],y_pos/self.stride[1]]
				y_pos += self.stride[1] 
			x_pos += self.stride[0]
		return layer_derivative
	
	def derivative_prev_layer(self,input_layer,output_layer_derivative):
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

