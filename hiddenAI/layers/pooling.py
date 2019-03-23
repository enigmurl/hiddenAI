import numpy as np
import math
from numpy.lib.stride_tricks import as_strided


class Pooling:
	def __init__(self):
		self.config = {"type":"POOLING","dimension":1}

	def get_padded_input_shape(self):#NEEDS WORK
		padded_shape = list(self.input_shape)
		for ind,dimension in enumerate(padded_shape[1:]):
			current = self.pooling_size[ind]
			start = self.pooling_size[ind]
			while dimension > start:
				current += self.stride[ind]
				dimension -= self.stride[ind] 
		
			padded_dimension = current	 
			padded_shape[ind+1] = padded_dimension
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
		output_layer = np.zeros(self.output_shape)
		for channel_num in range(self.padded_input_shape[0]):
			for x_pos in range(0,self.padded_input_shape[1] - self.pooling_size[1]+1,self.stride[0]):	
				output_layer[channel_num,x_pos//self.stride[0]] = np.max(input_layer[channel_num,x_pos:self.pooling_size[0]+x_pos])
		return output_layer	
		
	def derivative_prev_layer(self,input_layer, output_layer_derivative,**kwargs):
		prev_layer_derivative = np.zeros(self.input_shape)
		for channel_num in range(self.padded_input_shape[0]):
			for x_pos in range(0,self.padded_input_shape[1] - self.pooling_size[1]+1,self.stride[0]):	
				input_slice = input_layer[channel_num,x_pos:self.pooling_size[0] + x_pos]
				max_value = np.max(input_slice)
				slice = prev_layer_derivative[channel_num,x_pos:self.pooling_size[0] + x_pos] 
				slice[input_slice==max_value] +=  output_layer_derivative[channel_num,x_pos//self.stride[0]] 

		return prev_layer_derivative

class MaxPooling2D(Pooling):
	def __init__(self,pooling_size = [2,2] ,stride = [2,2],pad = True):
		super().__init__()
		self.config = {"type":"POOLING","dimension":3}
		self.pad = pad
		self.pooling_size = pooling_size if hasattr(pooling_size,"__iter__") else [pooling_size] 
		self.stride = stride if hasattr(stride,"__iter__") else [stride]

	def run(self,input_layer,least_value = -1000000): 
		output_layer = np.zeros(self.output_shape)
		for channel_num in range(self.padded_input_shape[0]):
			for x_pos in range(0,self.padded_input_shape[1] - self.pooling_size[1]+1,self.stride[0]):	
				for y_pos in range(0,self.padded_input_shape[2] - self.pooling_size[1]+1,self.stride[1]):	
					max_value = np.max(input_layer[channel_num,x_pos:self.pooling_size[0]+ x_pos,y_pos:self.pooling_size[1] + y_pos])
					output_layer[channel_num,x_pos//self.stride[0],y_pos//self.stride[1]] = max_value
		return output_layer	
		
	def derivative_prev_layer(self,input_layer, output_layer_derivative,**kwargs):
		prev_layer_derivative = np.zeros(self.input_shape)
		for channel_num in range(self.padded_input_shape[0]):
			for x_pos in range(0,self.padded_input_shape[1] - self.pooling_size[1]+1,self.stride[0]):	
				for y_pos in range(0,self.padded_input_shape[2] - self.pooling_size[1]+1,self.stride[1]):	
					input_slice = input_layer[channel_num,x_pos:self.pooling_size[0]+ x_pos,y_pos:self.pooling_size[1] + y_pos]
					max_value = np.max(input_slice)
					slice = prev_layer_derivative[channel_num,x_pos:self.pooling_size[0] + x_pos,y_pos:self.pooling_size[1] + y_pos]
					slice[input_slice==max_value] += output_layer_derivative[channel_num,x_pos//self.stride[1],y_pos//self.stride[1]]

		return prev_layer_derivative

class AvgPooling1D(Pooling):
	def __init__(self,pooling_size = [2] ,stride = [2],pad = True):
		super().__init__()
		self.config = {"type":"POOLING","dimension":2}
		self.pad = pad
		self.pooling_size = pooling_size if hasattr(pooling_size,"__iter__") else [pooling_size] 
		self.stride = stride if hasattr(stride,"__iter__") else [stride]
	
	def run(self,input_layer):
		output_layer = np.zeros(self.output_shape)
		for channel_num in range(self.padded_input_shape[0]):
			for x_pos in range(0,self.padded_input_shape[1] - self.pooling_size[1]+1,self.stride[0]):	
				output_layer[channel_num,x_pos//self.stride[0]] = np.mean(input_layer[channel_num,x_pos:self.pooling_size[0]+x_pos])
		return output_layer	
		
	def derivative_prev_layer(self,input_layer, output_layer_derivative,**kwargs):
		prev_layer_derivative = np.zeros(self.input_shape)
		for channel_num in range(self.padded_input_shape[0]):
			for x_pos in range(0,self.padded_input_shape[1] - self.pooling_size[1]+1,self.stride[0]):	
				slice = prev_layer_derivative[channel_num,x_pos:self.pooling_size[0] + x_pos] 
				slice +=  output_layer_derivative[channel_num,x_pos//self.stride[0]]/(self.pooling_size[0]) 

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
			for x_pos in range(0,self.padded_input_shape[1] - self.pooling_size[1]+1,self.stride[0]):	
				for y_pos in range(0,self.padded_input_shape[2] - self.pooling_size[1]+1,self.stride[1]):	
					mean_value = np.mean(input_layer[channel_num,x_pos:self.pooling_size[0]+ x_pos,y_pos:self.pooling_size[1] + y_pos])
					output_layer[channel_num,x_pos//self.stride[0],y_pos//self.stride[1]] = mean_value
		return output_layer	
		
	def derivative_prev_layer(self,input_layer, output_layer_derivative,**kwargs):
		prev_layer_derivative = np.zeros(self.input_shape)
		for channel_num in range(self.padded_input_shape[0]):
			for x_pos in range(0,self.padded_input_shape[1] - self.pooling_size[1]+1,self.stride[0]):	
				for y_pos in range(0,self.padded_input_shape[2] - self.pooling_size[1]+1,self.stride[1]):	
					slice = prev_layer_derivative[channel_num,x_pos:self.pooling_size[0] + x_pos,y_pos:self.pooling_size[1] + y_pos]
					slice += output_layer_derivative[channel_num,x_pos//self.stride[1],y_pos//self.stride[1]]/(self.pooling_size[0] * self.pooling_size[1])

		return prev_layer_derivative
if __name__ == "__main__":
	import time
	import numpy as np
	maxl = MaxPooling2D(pooling_size = [3,3])
	maxl.init_input_shape((1,4,4))
	A = np.array([[[1, 1, 2, 4],
    	          [5, 6, 7, 8],
    	          [3, 2, 1, 0],
    	          [1, 2, 3, 4]]])
	print(maxl.run(A))

