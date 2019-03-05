import numpy as np
class Lasso:
	def __init__(self,multiplier):
		self.multiplier = multiplier
		
	def loss(self,weights, input_value,expected_output,loss_function):
		return loss_function(input_value,expected_output) + self.multiplier * sum([abs(x) for x in weights])
	
	def apply_derivative(self,weights,current_weight_gradient):
		weight_derivatives = np.full(weights.shape,self.multiplier)
		weight_derivatives[weights<0] = - self.multiplier
		return weight_derivatives + current_weight_gradient
		
