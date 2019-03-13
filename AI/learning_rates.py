class ConstantLearningRate:
	def __init__(self,learning_rate):
		self.learning_rate = learning_rate
	
	def next_learning_rate(self,iteration_num):
		return self.learning_rate

class DecayLearningRate:
	def __init__(self,learning_rate,decay):
		self.learning_rate = learning_rate
		self.decay = decay

	def next_learning_rate(self,iteration_num):
		return self.learning_rate/(1+iteration_num +self.decay)
