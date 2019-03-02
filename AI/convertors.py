import math
import numpy as np
import random
#CONVERTORS:
class Convertor:
	def __init__(self):
		self.config = {"dimension":"ANY","type":"CONVERTOR"}
	
class ConvertorND1D(Convertor):
	def __init__(self,input_shape):
		super().__init__()
		self.input_shape = input_shape if hasattr(input_shape,"__iter__") else [input_shape]
		self.output_shape = self.compute_output_shape(input_shape)

	def compute_output_shape(self,input_shape):
		shape =1 
		for dimension in input_shape:
			shape *= dimension
		return [shape]

	def run(self,input_layer):
		return np.reshape(input_layer,self.output_shape[0])

	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		return np.reshape(output_layer_derivative,self.input_shape)


 
 



