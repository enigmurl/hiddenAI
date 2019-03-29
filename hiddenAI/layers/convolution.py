import math
import numpy as np
import random
from hiddenAI.hidden import Hidden

class Convolution(Hidden):
	def __init__(self,input_shape = None,num_filters = 1,filter_size = (2) , stride = (2)):#organize the function
		self.num_filters = num_filters
		
		self.input_shape = np.array(input_shape) if input_shape is not np.ndarray else input_shape
		self.num_input_channels = self.input_shape[0]
		self.padded_shape = self.get_padded_shape() if hasattr(self,"pad") else self.input_shape
		self.individual_input_shape = self.padded_shape[1:]
		
		self.filter_size = list(filter_size) if hasattr(filter_size,"__iter__") else [filter_size]
		self.stride = stride if hasattr(stride,"__iter__") else [stride]	
		
		super().__init__(input_shape,self.compute_output_shape())
		
		input_num_neurons = 1
		output_num_neurons = 1
		for neuron in self.input_shape:
			input_num_neurons *= neuron
		for neuron in self.output_shape:
			output_num_neurons *= neuron
		starting_value = -((6/(input_num_neurons + output_num_neurons))**0.5)
		ending_value = -starting_value
		self.shape = [num_filters,self.num_input_channels]+ self.filter_size
		self.weights = np.random.uniform(starting_value,ending_value,size = self.shape)
		
		
		self.dimension = len(self.filter_size)
		self.config["dimension"] = self.dimension + 1

	
	def get_padded_shape(self):#NEEDS WORK
		return np.concatenate((np.array([self.num_input_channels]),np.array([dimension+2*self.pad for dimension in self.input_shape[1:]]) ) )

	def compute_output_shape(self):
		return np.concatenate((np.array([self.num_filters]),np.array([(dimension-self.filter_size[ind])//self.stride[ind] +1 for ind,dimension in enumerate(self.padded_shape[1:])]) ) )

	def run(self,input_layer):
		pass

	def derivative(self,input_layer,output_layer_derivative,**kwargs):
		pass

	def derivative_prev_layer(self,input_layer,output_layer_derivative,**kwargs):
		pass

	def blank(self):
		return np.zeros(self.weights.shape) 
	
	def descend(self,derivatives):
		self.weights -= derivatives
	
class Convolution1D(Convolution):#NEEDS TESTING
	def __init__(self,num_filters =1, filter_size = (2), stride = (2),pad = 0):
		self.config = {"dimension":2,"type":"HIDDEN"}
		self.num_filters = num_filters 
		self.filter_size = filter_size if filter_size is np.ndarray else np.array(filter_size) 
		self.stride = stride if stride is np.ndarray else np.array(stride)
		self.pad = pad
	
	def init_input_shape(self,input_shape):
		super().__init__(input_shape,self.num_filters,self.filter_size,self.stride)
		self.filter_number_weights = self.filter_size[0] * self.num_input_channels
		self.number_weights = self.filter_number_weights * self.num_filters
 
	def run(self,input_layer):	
		padded_input_layer = np.zeros(self.padded_shape)
		padded_input_layer[:input_layer.shape[0],self.pad:self.pad+input_layer.shape[1]] = input_layer
		output_channel = np.zeros(self.output_shape)
		for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
				slice =  np.array(padded_input_layer[:,x_pos:x_pos + self.filter_size[0]])
				for filter in self.weights:
					multiplied_average = slice*filter
					multiplied_sum_average = np.sum(multiplied_average)
					output_channel[x_pos//self.stride[0]] = multiplied_sum_average
		return output_channel
	
	def derivative(self,input_layer,output_layer_derivative):
		padded_input_layer = np.zeros(self.padded_shape)
		padded_input_layer[:,self.pad:self.pad+input_layer.shape[1],self.pad:self.pad+input_layer.shape[2]] = input_layer
		weight_derivative = np.zeros(self.weights.shape)
		for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
			slice = padded_input_layer[:,x_pos:x_pos + self.filter_size[0],y_pos:y_pos + self.filter_size[1]]
			for filter_num in range(self.num_filters): 
				weight_derivative[filter_num] += slice * output_layer_derivative[filter_num,x_pos//self.stride[0]]
		return weight_derivative

	def derivative_prev_layer(self,input_layer,output_layer_derivative,**kwargs):
		prev_layer_derivative = np.zeros(self.padded_shape)
		for filter_num,filter in enumerate(self.weights):
			for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
				prev_layer_derivative[:,x_pos:x_pos + self.filter_size[0]] += output_layer_derivative[filter_num,x_pos//self.stride[0]] * filter
		return prev_layer_derivative[:,self.pad:self.pad+input_layer.shape[1]] 
		
 
class Convolution2D(Convolution):#NEEDS TESTING
	def __init__(self,num_filters =1, filter_size = (2,2), stride = (2,2),pad = 0):
		self.config = {"dimension":3,"type":"HIDDEN"}
		self.num_filters = num_filters
		self.filter_size = filter_size if filter_size is np.ndarray else np.array(filter_size) 
		self.stride = stride if stride is np.ndarray else np.array(stride)
		self.pad = pad
	
	def init_input_shape(self,input_shape):
		super().__init__(input_shape,self.num_filters,self.filter_size,self.stride)
		self.filter_number_weights = self.filter_size[0] * self.filter_size[1] * self.num_input_channels
		self.number_weights = self.filter_number_weights * self.num_filters
	
	def run(self,input_layer):		
		padded_input_layer = np.zeros(self.padded_shape)
		padded_input_layer[:,self.pad:self.pad+input_layer.shape[1],self.pad:self.pad+input_layer.shape[2]] = input_layer
		output_channel = np.zeros(self.output_shape)
		for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
			for y_pos in range(0,self.individual_input_shape[1]-self.filter_size[1]+1,self.stride[1]):
				slice =  np.array(padded_input_layer[:,x_pos:x_pos + self.filter_size[0],y_pos:y_pos + self.filter_size[1]])
				for filter_num,filter in enumerate(self.weights):
					multiplied_average = slice*filter
					multiplied_sum_average = np.sum(multiplied_average)
					output_channel[filter_num,x_pos//self.stride[0],y_pos//self.stride[1]] = multiplied_sum_average	
		#print("CONV",output_channel)	
		return output_channel

	def derivative(self,input_layer,output_layer_derivative):
		padded_input_layer = np.zeros(self.padded_shape)
		padded_input_layer[:,self.pad:self.pad+input_layer.shape[1],self.pad:self.pad+input_layer.shape[2]] = input_layer
		weight_derivative = np.zeros(self.weights.shape)
		for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
			for y_pos in range(0,self.individual_input_shape[1]-self.filter_size[1]+1,self.stride[1]):
				slice = padded_input_layer[:,x_pos:x_pos + self.filter_size[0],y_pos:y_pos + self.filter_size[1]]
				for filter_num in range(self.num_filters): 
					weight_derivative[filter_num] += slice * output_layer_derivative[filter_num,x_pos//self.stride[0],y_pos//self.stride[1]]
		return weight_derivative

	def derivative_prev_layer(self,input_layer,output_layer_derivative,**kwargs):
		prev_layer_derivative = np.zeros(self.padded_shape)
		for filter_num,filter in enumerate(self.weights):
			for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
				for y_pos in range(0,self.individual_input_shape[1] - self.filter_size[1] + 1, self.stride[1]):
					prev_layer_derivative[:,x_pos:x_pos + self.filter_size[0],y_pos:y_pos + self.filter_size[1]] += output_layer_derivative[filter_num,x_pos//self.stride[0],y_pos//self.stride[1]] * filter
		return prev_layer_derivative[:,self.pad:self.pad+input_layer.shape[1],self.pad:self.pad+input_layer.shape[2]] 

class TransposedConvolution1D(Convolution):
	def __init__(self,num_filters =1, filter_size = (2), stride = (2)):
		self.config = {"dimension":2,"type":"HIDDEN"}
		self.num_filters = num_filters 
		self.filter_size = filter_size if filter_size is np.ndarray else np.array(filter_size) 
		self.stride = stride if stride is np.ndarray else np.array(stride)
	
	def init_input_shape(self,input_shape):
		super().__init__(input_shape,self.num_filters,self.filter_size,self.stride)
		self.filter_number_weights = self.filter_size[0] * self.num_input_channels
	def compute_output_shape(self):
		return#NEEDS TO IMPLEMENTED

	def derivative_prev_layer(self,input_layer,output_layer_derivative):#needs testing	
		padded_input_layer = np.zeros(self.padded_shape)
		padded_input_layer[:input_layer.shape[0],self.pad:self.pad+input_layer.shape[1]] = input_layer
		output_channel = np.zeros(self.output_shape)
		for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
				slice =  np.array(padded_input_layer[:,x_pos:x_pos + self.filter_size[0]])
				for filter in self.weights:
					multiplied_average = slice*filter
					multiplied_sum_average = np.sum(multiplied_average)
					output_channel[x_pos//self.stride[0]] = multiplied_sum_average
		return output_channel * output_layer_derivative
	
	def derivative(self,input_layer,output_layer_derivative):#needs to be implemented
		padded_input_layer = np.zeros(self.padded_shape)
		padded_input_layer[:,self.pad:self.pad+input_layer.shape[1],self.pad:self.pad+input_layer.shape[2]] = input_layer
		weight_derivative = np.zeros(self.weights.shape)
		for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
			slice = padded_input_layer[:,x_pos:x_pos + self.filter_size[0],y_pos:y_pos + self.filter_size[1]]
			for filter_num in range(self.num_filters): 
				weight_derivative[filter_num] += slice * output_layer_derivative[filter_num,x_pos//self.stride[0]]
		return weight_derivative

	def run(self,input_layer):#needs testing
		prev_layer_derivative = np.zeros(self.padded_shape)
		for filter_num,filter in enumerate(self.weights):
			for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
				prev_layer_derivative[:,x_pos:x_pos + self.filter_size[0]] += input_layer[filter_num,x_pos//self.stride[0]] * filter
		return prev_layer_derivative[:,self.pad:self.pad+input_layer.shape[1]] 

class TransposedConvolution2D(Convolution):#NEEDS TESTING
	def __init__(self,num_filters =1, filter_size = (2,2), stride = (2,2)):
		self.config = {"dimension":3,"type":"HIDDEN"}
		self.num_filters = num_filters
		self.filter_size = filter_size if filter_size is np.ndarray else np.array(filter_size) 
		self.stride = stride if stride is np.ndarray else np.array(stride)
	
	def init_input_shape(self,input_shape):
		super().__init__(input_shape,self.num_filters,self.filter_size,self.stride)
		self.filter_number_weights = self.filter_size[0] * self.filter_size[1] * self.num_input_channels
		self.number_weights = self.filter_number_weights * self.num_filters
	
	def compute_output_shape(self):
		channels = np.array([self.input_shape[0]])
		dimensions = self.input_shape[1:] + np.clip(self.filter_size-self.stride,0,None)
		return np.concatenate((channels,dimensions))
	
	def derivative_prev_layer(self,input_layer,output_layer_derivative):#needs testing	
		prev_layer_derivative = np.zeros(self.output_shape)
		for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
			for y_pos in range(0,self.individual_input_shape[1]-self.filter_size[1]+1,self.stride[1]):
				slice =  np.array(output_layer_derivative[:,x_pos:x_pos + self.filter_size[0],y_pos:y_pos + self.filter_size[1]])
				for filter_num,filter in enumerate(self.weights):
					multiplied_average = slice*filter
					multiplied_sum_average = np.sum(multiplied_average)
					prev_layer_derivative[filter_num,x_pos//self.stride[0],y_pos//self.stride[1]] = multiplied_sum_average		
		return prev_layer_derivative*output_layer_derivative

	def derivative(self,input_layer,output_layer_derivative):#needs to be implemented
		weight_derivative = np.zeros(self.weights.shape)
		for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
			for y_pos in range(0,self.individual_input_shape[1]-self.filter_size[1]+1,self.stride[1]):
				#slice = padded_input_layer[:,x_pos:x_pos + self.filter_size[0],y_pos:y_pos + self.filter_size[1]]
				slice =  np.array(output_layer_derivative[:,x_pos:x_pos + self.filter_size[0],y_pos:y_pos + self.filter_size[1]])
				for filter_num in range(self.num_filters): 
					weight_derivative[filter_num] += slice * input_layer[filter_num,x_pos//self.stride[0],y_pos//self.stride[1]]
		return weight_derivative

	def run(self,input_layer):#needs testing,may need padding?
		output_layer = np.zeros(self.padded_shape)
		for filter_num,filter in enumerate(self.weights):
			for x_pos in range(0,self.individual_input_shape[0]-self.filter_size[0]+1,self.stride[0]):
				for y_pos in range(0,self.individual_input_shape[1] - self.filter_size[1] + 1, self.stride[1]):
					output_layer[:,x_pos:x_pos + self.filter_size[0],y_pos:y_pos + self.filter_size[1]] += input_layer[filter_num,x_pos//self.stride[0],y_pos//self.stride[1]] * filter
		return output_layer[:,:input_layer.shape[1],:input_layer.shape[2]] 
	
if __name__ == "__main__":
	from ..loss import *
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
