import numpy as np
from hiddenAI.hidden import Hidden

class BrainNeuron(Hidden):
	def __init__(self,output_shape):
		self.output_shape = output_shape
		self.currentstate = np.zeros(self.output_shape)

	def reset(self):
		self.currentstate = np.zeros(self.output_shape)

	def init_input_shape(self,input_shape):
		super().__init__(input_shape,self.output_shape)	
		starting_value = -((6/(self.input_shape[0] + self.output_shape[0]))**0.5)
		self.weights = np.random.uniform(0,-starting_value, size =(self.output_shape[0],self.input_shape[0]))
		self.num_weights = self.input_shape[0]*self.output_shape[0]
	
	def compute_output_shape(self):
		return self.output_shape

	def run(self):
		self.currentstate += np.matmul(self.weights,input_layer)
		return_layer = np.zeros(self.output_shape):
		mask = self.currentstate >= 1
		self.currentstae[mask] = 0
		return_layer[mask] = 1
		return return_layer
	
	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		pass

	def derivative(self,input_layer, output_layer_derivative):


