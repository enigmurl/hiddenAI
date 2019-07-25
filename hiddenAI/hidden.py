#HIDDEN
class Hidden:
	def __init__(self,input_shape,output_shape):
		self.config = {"dimension":"ANY","type":"HIDDEN"}
		self.input_shape = input_shape if hasattr(input_shape,"__iter__") else [input_shape]
		self.output_shape = output_shape if hasattr(output_shape,"__iter__") else [output_shape]
	
	def reset(self):
		pass
	
	def update_weights(self,weights):
		self.weights = weights
	
	def run(self,input_layer): #may be over rided if the layer is 1d to 2d, or a different form
		pass# overided as they differ heavily

	def derivative(self, input_layer,output_layer_derivative,**kwargs):
		pass# to be overrided

	def derivative_prev_layer(self,input_layer, output_layer_derivative,**kwargs): # how does the previous layer affect the next
		pass# to be overrided
		
	def descend(self, derivatives): 
		self.weights -= derivatives
	
	def blank(self):
		pass # to be overided
