Class BatchGradientDescent:
	def __init__(self,batch_size = 5,momentum = 0,learning_rate = 1):
		self.momentum = momentum 
		self.learning_rate = learning_rate
	
	def train(self,input_data,expected_outputs,layers,run_function):
		
