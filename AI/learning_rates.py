class ConstantLearningRate:
	def __init__(self,learning_rate):
		self.learning_rate = learning_rate
	
	def next_learning_rate(self,iteration_num):
		return self.learning_rate
