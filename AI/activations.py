import math
import numpy as np
import random

class  Activation:
	def __init__(self,shape = None, total_items = None):
		self.config = {"dimension":1,"type":"ACTIVATION"}
		self.shape = shape # will be initiated by the neural net
		self.total_items = total_items # will be initiated by the neural net
		self.dimension = len(shape) if shape != None else None

	def run (self,input_layer): # may want to optimize
		return np.array([self.apply(val) for val in input_layer])
		'''
		formatted_input_layer = np.reshape(input_layer,self.total_items) if self.dimension != 1 else input_layer	
		start_shape = input_layer.shape	

		output_layer = []
		for val in formatted_input_layer:
			output_layer.append(self.apply(val))
		
		formatted_output_layer = np.reshape(np.array(output_layer),start_shape) if self.dimension != 1 else output_layer
			
		return formatted_output_layer
		'''
	def init_input_shape(self,input_shape):
		self.input_shape = input_shape
		self.output_shape = self.compute_output_shape(input_shape)

	def compute_output_shape(self,input_shape):
		return input_shape

	def apply(self,x):
		pass# to be implemented in subclasses

	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		return np.array([self.derivative(val) for val in input_layer]) * output_layer_derivative
		'''
		formatted_input_layer = np.reshape(input_layer,self.total_items) if self.dimension != 1 else input_layer	
		start_shape = input_layer.shape
		output_layer = []
		for val in formatted_input_layer:
			output_layer.append(self.derivative(val))
		
		formatted_output_layer = np.reshape(np.array(output_layer),start_shape) if self.dimension != 1 else output_layer
			
		return formatted_output_layer * output_layer_derivative
		'''
	def derivative(self,x):
		pass

class Sigmoid(Activation): #TESTED
	def __init__(self):
		super().__init__()
		self.config = {"dimension":"ANY","type":"ACTIVATION"}
			
	def run(self,input_layer):#apply and derivative arent needed as this is more efficient
		input_layer[input_layer>30] = 30
		input_layer[input_layer<-30] = -30
		return 1/(1+np.exp(-input_layer))
	
	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		partial_derivative = self.run(input_layer)
		return partial_derivative * (1-partial_derivative) * output_layer_derivative

class BinaryStep(Activation): #TESTED
	def __init__(self):
		super().__init__()

	def apply(self,x):
		return 1 if x>0 else 0

	def derivative(self,x):
		return 1 if x == 0 else 0 

class ReLU(Activation): #TESTED 
	def __init__(self):
		super().__init__()
		self.config = {"dimension":"ANY","type":"ACTIVATION"}

	def run(self,input_layer):
		output_layer = np.zeros(self.output_shape) 
		output_layer[input_layer>0] = input_layer[input_layer>0]
		return output_layer

	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		prev_layer_derivative = np.zeros(self.input_shape)
		prev_layer_derivative[input_layer>0] = output_layer_derivative[input_layer>0]
		return prev_layer_derivative 

	def apply(self,x):
		return max(0,x)
	
	def derivative(self,x):
		return 1 if x >= 0 else 0  

class PReLU(Activation): #TESTED
	def __init__(self,multiplier = 1/5):
		super().__init__()
		self.multiplier = multiplier
	
	def apply(self,x):
		return x if x>= 0 else self.multiplier*x
	
	def derivative(self,x):
		return 1 if x >= 0 else self.multiplier	

class ELU(Activation): #TESTED
	def __init__(self,multiplier = 1/5):
		super().__init__()
		self.multiplier = multiplier

	def apply(self,x):
		return x if x>=0 else self.multiplier*(math.e**x -1) if x>-30 else self.multiplier

	def derivative(self,x):
		return 1 if x >= 0 else self.apply(x) + self.multiplier

class Softplus(Activation): #TESTED 
	def __init__(self):
		super().__init__()

	def apply(self,x):#natural log of(1+e^x)
		return math.log(1+math.pow(math.e,x))

	def derivative(self,x):
		return 1/(1+math.e**-x) if abs(x) < 30 else 0 if x < -30 else 1

		
class Tanh(Activation): #TESTED hyperbolic tangent
	def __init__(self):
		super().__init__()

	def apply(self,x):
		return math.tanh(x)

	def derivative(self,x):
		return 1-self.apply(x)**2

class Arctan(Activation): #
	def __init__(self):
		super().__init__()

	def apply(self,x):
		return math.atan(x)

	def derivative(self,x):
		return 1/(x**2+1)

	
#May want to add pooling
