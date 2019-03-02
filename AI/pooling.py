import numpy as np

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
	
	def compute_output_shape(self,input_shape):#NEEDS TO REWORK
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
		return np.array(output_layer)	
		
	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		padded_input_layer = np.full(self.padded_input_shape,np.nan)
		padded_input_layer[:self.input_shape[0],:self.input_shape[1]] = input_layer
		prev_layer_derivative = np.zeros(self.input_shape)
		for channel_num in range(self.padded_input_shape[0]):
			x_pos = 0
			while x_pos + self.pooling_size[0] <= self.padded_input_shape[1]:
				max_value = np.max(input_layer[channel_num,x_pos:self.pooling_size[0]+ x_pos])
				slice = prev_layer_derivative[channel_num,x_pos:self.pooling_size[0] + x_pos]
				slice[slice==max_value] = 1 
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
		padded_input_layer = np.full(self.padded_input_shape,np.nan)
		padded_input_layer[:self.input_shape[0],:self.input_shape[1],:self.input_shape[2]] = input_layer
		output_layer = np.zeros(self.output_shape)
		for channel_num in range(self.padded_input_shape[0]):
			x_pos = 0
			while x_pos + self.pooling_size[0] <= self.padded_input_shape[1]:
				y_pos = 0
				while y_pos + self.pooling_size[1] <= self.padded_input_shape[2]: 
					output_layer[channel_num,x_pos//self.stride[0],y_pos//self.stride[1]] = (np.max(input_layer[channel_num,x_pos:self.pooling_size[0]+x_pos,y_pos:self.pooling_size[1]+y_pos]))
					y_pos += self.stride[1]
				x_pos += self.stride[0]	
		return np.array(output_layer)	
		
	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		padded_input_layer = np.full(self.padded_input_shape,np.nan)
		padded_input_layer[:self.input_shape[0],:self.input_shape[1],:self.input_shape[2]] = input_layer
		prev_layer_derivative = np.zeros(self.input_shape)
		for channel_num in range(self.padded_input_shape[0]):
			x_pos = 0 	
			while x_pos + self.pooling_size[0] <= self.padded_input_shape[1]:
				y_pos = 0	
				while y_pos + self.pooling_size[1] <= self.padded_input_shape[2]: 
					max_value = np.max(input_layer[channel_num,x_pos:self.pooling_size[0]+ x_pos,y_pos:self.pooling_size[1] + y_pos])
					slice = prev_layer_derivative[channel_num,x_pos:self.pooling_size[0] + x_pos,y_pos:self.pooling_size[1] + y_pos]
					slice[slice==max_value] += 1
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
	#max = MaxPooling2D()
	#max.init_input_shape((1,4,5))
	''' a = np.array([ 
	[[  20,  200,   -5,   23, -1],
	[ -13,  134,  119,  100,-1],
	[ 120,   32,   49,   25,-1],
	[-120,   12,    9,   23,-1]]
	])
	'''
	a = np.array([[0,1,2,3,4,5]])
	max = MaxPooling1D()
	max.init_input_shape((1,6))
	print("OUTPUT:",max.output_shape)
	print(max.run(a))
