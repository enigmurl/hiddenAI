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
	
class Ridge:
	def __init__(self,multiplier):
		self.multiplier = multiplier 
		
	def loss(self,weights, input_value,expected_output,loss_function):
		return loss_function(input_value,expected_output) + self.multiplier * sum([x*x for x in weights])
	
	def apply_derivative(self,weights,current_weight_gradient):
		return (self.multiplier * 2 * weights) + current_weight_gradient

if __name__ == "__main__":
	a = Lasso(1)
	b = np.array([0.,1.,2.,3.,10.])
	c = np.array([1.,1.,1.,1.,1.])
	print(a.apply_derivative(b,c),a.loss())
	for i in range(100):
		b -= a.apply_derivative(b,c) * 0.1 
	print(b)	
