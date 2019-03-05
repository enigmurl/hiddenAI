import numpy as np
from numpy.lib.stride_tricks import as_strided


class Pooling:
	def __init__(self):
		self.config = {"type":"POOLING","dimension":1}

	def get_padded_input_shape(self):
		padded_input_shape = list(self.input_shape)
		for ind,dimension in enumerate(padded_input_shape[1:]):
			offset =  (dimension-self.pooling_size[ind]) % self.stride[ind]
			padded_input_shape[ind+1] = dimension + offset
		return padded_input_shape
	
	def init_input_shape(self,input_shape):
		self.input_shape = input_shape if hasattr(input_shape,"__iter__") else [input_shape]
		self.output_shape = self.compute_output_shape(self.input_shape)	
		self.padded_input_shape = self.get_padded_input_shape() if self.pad else self.input_shape
	
	def compute_output_shape(self,input_shape):
		output_shape = [input_shape[0]]
		for ind,dimension in enumerate(input_shape[1:]):
			offset =  (dimension - self.pooling_size[ind]) % self.stride[ind]
			dimension_shape = (dimension + offset) // self.stride[ind]
			output_shape.append(dimension_shape)
		return np.array(output_shape)

	def run(self,input_layer): 
		pass # to be overided
	
	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		pass

class MaxPooling1D(Pooling):
	def __init__(self,pooling_size = [2] ,stride = [2],pad = True):
		super().__init__()
		self.config = {"type":"POOLING","dimension":2}
		self.pad = pad
		self.pooling_size = pooling_size 
		self.stride = stride
	
	def run(self,input_layer): 
		padded_input_layer = np.full(self.padded_input_shape,np.nan)
		padded_input_layer[:self.input_shape[0],:self.input_shape[1]] = input_layer
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
		
	def derivative_prev_layer(self,input_layer, output_layer_derivative):
		padded_input_layer = np.full(self.padded_input_shape,np.nan)
		padded_input_layer[:self.input_shape[0],:self.input_shape[1]] = input_layer
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
		self.pooling_size = pooling_size 
		self.stride = stride

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
		
	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		prev_layer_derivative = np.zeros(self.input_shape)
		for channel_num in range(self.padded_input_shape[0]):
			x_pos = 0 	
			while x_pos + self.pooling_size[0] <= self.padded_input_shape[1]:
				y_pos = 0	
				while y_pos + self.pooling_size[1] <= self.padded_input_shape[2]: 
					input_slice = input_layer[channel_num,x_pos:self.pooling_size[0]+ x_pos,y_pos:self.pooling_size[1] + y_pos]
					max_value = np.max(input_layer[channel_num,x_pos:self.pooling_size[0]+ x_pos,y_pos:self.pooling_size[1] + y_pos])
					slice = prev_layer_derivative[channel_num,x_pos:self.pooling_size[0] + x_pos,y_pos:self.pooling_size[1] + y_pos]
					slice[input_slice==max_value] += output_layer_derivative[channel_num,x_pos//self.stride[1],y_pos//self.stride[1]]
					y_pos += self.stride[1] 
				x_pos += self.stride[0]

		return prev_layer_derivative

class AvgPooling(Pooling):
	def __init__(self,pooling_size = 2,stride = 2):
		super().__init__()
		self.pooling_size = pooling_size 
		self.stride = stride

	def run(self,input_layer):
		output_layer = []
		x_pos = 0
		while x_pos + self.stride <len(input_layer):
			output_layer.append(sum(input_layer[x_pos:self.pooling_size+x_pos])/self.poolingSize)
			self.stride += 1
		return np.array(output_layer)	

	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		in_derivatives = np.zeros(len(input_layer)) 
		x_pos = 0
		while x_pos + self.stride <len(input_layer):
			for ind,val in enumerate(input_layer[x_pos:self.pooling_size+x_pos]):
				in_derivatives[x_pos+ind] += output_layer_derivative[x_pos]/self.pooling_size
			x_pos += self.stride	
		return in_derivatives
	
	def compute_output_shape(self,input_shape):
		pass
if __name__ == "__main__":
	max = MaxPooling2D()
	max.init_input_shape((1,4,5))
	a = np.array([ 
	[[  20,  200,   -5,   23,-1],
	[ -13,  134,  119,  100,-1],
	[ 120,   32,   49,   25,-1],
	[-120,   12,    9,   23,-1]]
	])
	print(max.run(a))
	'''
	from activations import *
	from loss import *
	a = np.array([[0.0,1.0,2.0,3.0,4.0,5.0]])
	max = MaxPooling1D()
	max.init_input_shape((1,6))
	bsize=  100
	for i in range(bsize):
		pre_error = max.run(a)
		after_error = mean_squared_cost(pre_error,np.array([1,7,-1]))
		derivative1 = derivative_mean_squared_cost(pre_error,np.array([1,7,-1]),batch_size = 10)
		derivative2 = max.derivative_prev_layer(a,derivative1)
		a -= derivative2
		print(pre_error,after_error)
	print(max.run(a))	
'''
