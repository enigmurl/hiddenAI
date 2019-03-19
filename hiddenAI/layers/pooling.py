import numpy as np
import math
from numpy.lib.stride_tricks import as_strided


class Pooling:
	def __init__(self):
		self.config = {"type":"POOLING","dimension":1}

	def get_padded_input_shape(self):#NEEDS WORK
		padded_shape = list(self.input_shape)
		for ind,dimension in enumerate(padded_shape[1:]):
			#offset =  (dimension - self.pooling_size[ind]) %  (self.stride[ind]) - 1
			#padded_dimension = max(self.pooling_size[ind],dimension + offset) 
			current = self.pooling_size[ind]
			start = self.pooling_size[ind]
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
		output_shape = [padded_shape[0]]
		for ind,dimension in enumerate(padded_shape[1:]):
			offset =  (dimension - self.pooling_size[ind]) % self.stride[ind]
			dimension_shape = math.ceil((dimension - self.pooling_size[ind] + 1) / self.stride[ind])
			output_shape.append(dimension_shape)
		return np.array(output_shape)
	
	def init_input_shape(self,input_shape):
		self.input_shape = input_shape if hasattr(input_shape,"__iter__") else [input_shape]
		self.padded_input_shape = self.get_padded_input_shape() if self.pad else self.input_shape
		self.output_shape = self.compute_output_shape(self.padded_input_shape)	
		#print(self.padded_input_shape)	

	def run(self,input_layer): 
		pass # to be overided
	
	def derivative_prev_layer(self,input_layer,output_layer_derivative,**kwargs):
		pass

class MaxPooling1D(Pooling):
	def __init__(self,pooling_size = [2] ,stride = [2],pad = True):
		super().__init__()
		self.config = {"type":"POOLING","dimension":2}
		self.pad = pad
		self.pooling_size = pooling_size if hasattr(pooling_size,"__iter__") else [pooling_size] 
		self.stride = stride if hasattr(stride,"__iter__") else [stride]
	
	def run(self,input_layer): 
		'''
		padded_input_layer = np.full(self.padded_input_shape,np.nan)
		padded_input_layer[:self.input_shape[0],:self.input_shape[1]] = input_layer
		'''	
		output_layer = np.zeros(self.output_shape)
		for channel_num in range(self.padded_input_shape[0]):
			x_pos = 0
			while x_pos + self.pooling_size[0] <= self.padded_input_shape[1]:
				output_layer[channel_num,x_pos//self.stride[0]] = np.max(input_layer[channel_num,x_pos:self.pooling_size[0]+x_pos])
				x_pos += self.stride[0]
			'''
			strided_input_layer = as_strided(padded_input_layer, shape = self.output_shape[1:] + self.pooling_size,
						strides = (self.stride[0] * padded_input_layer.strides[0],self.stride[1] * padded_input_layer.strides[1]) + padded_input_layer.strides)
			reshaped_input_layer = strided_input_layer.reshape(-1,*self.pooling_size)
			output_layer[channel_num] = reshaped_input_layer.max(axis = (1,2)).reshape(self.output_shape) 
			'''
		return output_layer	
		
	def derivative_prev_layer(self,input_layer, output_layer_derivative,**kwargs):
		'''
		padded_input_layer = np.full(self.padded_input_shape,np.nan)
		padded_input_layer[:self.input_shape[0],:self.input_shape[1]] = input_layer
		'''
		prev_layer_derivative = np.zeros(self.input_shape)
		for channel_num in range(self.padded_input_shape[0]):
			x_pos = 0
			while x_pos + self.pooling_size[0] <= self.padded_input_shape[1]:
				input_slice = input_layer[channel_num,x_pos:self.pooling_size[0] + x_pos]
				max_value = np.max(input_slice)
				slice = prev_layer_derivative[channel_num,x_pos:self.pooling_size[0] + x_pos] 
				slice[input_slice==max_value] +=  output_layer_derivative[channel_num,x_pos//self.stride[0]] 
				x_pos += self.stride[0]

		return prev_layer_derivative

class MaxPooling2D(Pooling):
	def __init__(self,pooling_size = [2,2] ,stride = [2,2],pad = True):
		super().__init__()
		self.config = {"type":"POOLING","dimension":3}
		self.pad = pad
		self.pooling_size = pooling_size if hasattr(pooling_size,"__iter__") else [pooling_size] 
		self.stride = stride if hasattr(stride,"__iter__") else [stride]

	def run(self,input_layer): 
		output_layer = np.zeros(self.output_shape)
		for channel_num in range(self.padded_input_shape[0]):
			x_pos = 0 	
			while x_pos + self.pooling_size[0] <= self.padded_input_shape[1]:
				y_pos = 0	
				while y_pos + self.pooling_size[1] <= self.padded_input_shape[2]: 
					max_value = np.max(input_layer[channel_num,x_pos:self.pooling_size[0]+ x_pos,y_pos:self.pooling_size[1] + y_pos])
					output_layer[channel_num,x_pos//self.stride[0],y_pos//self.stride[1]] = max_value
					y_pos += self.stride[1] 
				x_pos += self.stride[0]
			'''
			strided_shape = (self.output_shape[1],self.output_shape[2],self.pooling_size[0],self.pooling_size[1])
			new_strides = (self.stride[0] * channel.strides[0], self.stride[1] * channel.strides[1]) + channel.strides
			
			strided_input_layer = as_strided(channel, shape = strided_shape,strides = new_strides) 
			reshaped_input_layer = strided_input_layer.reshape(-1,*self.pooling_size)
			output_layer[channel_num] = reshaped_input_layer.max(axis = (1,2)).reshape(self.output_shape) 
			'''
		return output_layer	
		
	def derivative_prev_layer(self,input_layer, output_layer_derivative,**kwargs):
		prev_layer_derivative = np.zeros(self.input_shape)
		for channel_num in range(self.padded_input_shape[0]):
			x_pos = 0 	
			while x_pos + self.pooling_size[0] <= self.padded_input_shape[1]:
				y_pos = 0	
				while y_pos + self.pooling_size[1] <= self.padded_input_shape[2]: 
					input_slice = input_layer[channel_num,x_pos:self.pooling_size[0]+ x_pos,y_pos:self.pooling_size[1] + y_pos]
					max_value = np.max(input_slice)
					slice = prev_layer_derivative[channel_num,x_pos:self.pooling_size[0] + x_pos,y_pos:self.pooling_size[1] + y_pos]
					slice[input_slice==max_value] += output_layer_derivative[channel_num,x_pos//self.stride[1],y_pos//self.stride[1]]
					y_pos += self.stride[1] 
				x_pos += self.stride[0]

		return prev_layer_derivative

class AvgPooling1D(Pooling):
	def __init__(self,pooling_size = [2] ,stride = [2],pad = True):
		super().__init__()
		self.config = {"type":"POOLING","dimension":2}
		self.pad = pad
		self.pooling_size = pooling_size if hasattr(pooling_size,"__iter__") else [pooling_size] 
		self.stride = stride if hasattr(stride,"__iter__") else [stride]
	
	def run(self,input_layer):
		''' 
		padded_input_layer = np.full(self.padded_input_shape,np.zeros)
		padded_input_layer[:self.input_shape[0],:self.input_shape[1]] = input_layer
		'''
		output_layer = np.zeros(self.output_shape)
		for channel_num in range(self.padded_input_shape[0]):
			x_pos = 0
			while x_pos + self.pooling_size[0] <= self.padded_input_shape[1]:
				output_layer[channel_num,x_pos//self.stride[0]] = np.mean(input_layer[channel_num,x_pos:self.pooling_size[0]+x_pos])
				x_pos += self.stride[0]
			'''
			strided_input_layer = as_strided(padded_input_layer, shape = self.output_shape[1:] + self.pooling_size,
						strides = (self.stride[0] * padded_input_layer.strides[0],self.stride[1] * padded_input_layer.strides[1]) + padded_input_layer.strides)
			reshaped_input_layer = strided_input_layer.reshape(-1,*self.pooling_size)
			output_layer[channel_num] = reshaped_input_layer.max(axis = (1,2)).reshape(self.output_shape) 
			'''
		return output_layer	
		
	def derivative_prev_layer(self,input_layer, output_layer_derivative,**kwargs):
		'''
		padded_input_layer = np.full(self.padded_input_shape,np.nan)
		padded_input_layer[:self.input_shape[0],:self.input_shape[1]] = input_layer
		'''
		prev_layer_derivative = np.zeros(self.input_shape)
		for channel_num in range(self.padded_input_shape[0]):
			x_pos = 0
			while x_pos + self.pooling_size[0] <= self.padded_input_shape[1]:
				slice = prev_layer_derivative[channel_num,x_pos:self.pooling_size[0] + x_pos] 
				slice +=  output_layer_derivative[channel_num,x_pos//self.stride[0]]/(self.pooling_size[0]) 
				x_pos += self.stride[0]

		return prev_layer_derivative

class AvgPooling2D(Pooling):
	def __init__(self,pooling_size = [2,2] ,stride = [2,2],pad = True):
		super().__init__()
		self.config = {"type":"POOLING","dimension":3}
		self.pad = pad
		self.pooling_size = pooling_size if hasattr(pooling_size,"__iter__") else [pooling_size] 
		self.stride = stride if hasattr(stride,"__iter__") else [stride]

	def run(self,input_layer): 
		output_layer = np.zeros(self.output_shape)
		for channel_num in range(self.padded_input_shape[0]):
			x_pos = 0 	
			while x_pos + self.pooling_size[0] <= self.padded_input_shape[1]:
				y_pos = 0	
				while y_pos + self.pooling_size[1] <= self.padded_input_shape[2]: 
					mean_value = np.mean(input_layer[channel_num,x_pos:self.pooling_size[0]+ x_pos,y_pos:self.pooling_size[1] + y_pos])
					output_layer[channel_num,x_pos//self.stride[0],y_pos//self.stride[1]] = mean_value
					y_pos += self.stride[1] 
				x_pos += self.stride[0]
			'''
			strided_shape = (self.output_shape[1],self.output_shape[2],self.pooling_size[0],self.pooling_size[1])
			new_strides = (self.stride[0] * channel.strides[0], self.stride[1] * channel.strides[1]) + channel.strides
			
			strided_input_layer = as_strided(channel, shape = strided_shape,strides = new_strides) 
			reshaped_input_layer = strided_input_layer.reshape(-1,*self.pooling_size)
			output_layer[channel_num] = reshaped_input_layer.max(axis = (1,2)).reshape(self.output_shape) 
			'''
		return output_layer	
		
	def derivative_prev_layer(self,input_layer, output_layer_derivative,**kwargs):
		prev_layer_derivative = np.zeros(self.input_shape)
		for channel_num in range(self.padded_input_shape[0]):
			x_pos = 0 	
			while x_pos + self.pooling_size[0] <= self.padded_input_shape[1]:
				y_pos = 0	
				while y_pos + self.pooling_size[1] <= self.padded_input_shape[2]: 
					slice = prev_layer_derivative[channel_num,x_pos:self.pooling_size[0] + x_pos,y_pos:self.pooling_size[1] + y_pos]
					slice += output_layer_derivative[channel_num,x_pos//self.stride[1],y_pos//self.stride[1]]/(self.pooling_size[0] * self.pooling_size[1])
					y_pos += self.stride[1] 
				x_pos += self.stride[0]

		return prev_layer_derivative
if __name__ == "__main__":
	import time
	max = MaxPooling2D()
	max.init_input_shape((1,4,5))
	a = np.array([ 
	[[  20,  200,   -5,   23,-1],
	[ -13,  134,  119,  100,-1],
	[ 120,   32,   49,   25,-1],
	[-120,   12,    9,   23,-1]]
	])
	#print(max.run(a))
	from activations import *
	a = np.array([[0.0,1.0,2.0,3.0,4.0,5.0]])
	max = MaxPooling1D()
	max.init_input_shape((1,6))
	bsize=  100
	for i in range(bsize):
		pre_error = max.run(a)
		after_error = mean_squared_cost(pre_error,np.array([1,7,-1]))

		print(pre_error,after_error)
	print(max.run(a))	
