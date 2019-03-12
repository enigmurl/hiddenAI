import numpy as np

class Dropout:
	def __init__(self,probability):	
		self.probability = probability
		self.config = {"dimension":"ANY","type":"DROPOUT"}	

	def init_input_shape(self,input_shape):
		self.input_shape = input_shape
		self.output_shape = input_shape	

	def run(self,input_layer):
		self.mask = np.random.choice((False,True),size = self.input_shape, p = [self.probability,1-self.probability] )#see if this needs to be switched,mask may be  self bc we need it for derivative_prev_layer
		output_layer = np.zeros(self.input_shape)
		output_layer[self.mask] = input_layer[self.mask]
		return output_layer
			
	def derivative_prev_layer(self,input_layer,output_layer_derivative,**kwargs):
		prev_layer_derivative = output_layer_derivative[:]
		prev_layer_derivative[self.mask] = 0
		return prev_layer_derivative
