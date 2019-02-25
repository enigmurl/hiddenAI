import numpy as np
class QLearning:#also known as reinforcement learning
	
	def __init__(self,number_of_states,number_of_actions,chance_of_exploration = 0.1)
		self.chance_of_exploration = chance_of_exploration
		self.qtable = np.random.randn((number_of_states,number_of_actions))
	
#RESEARCH Q LEARNING	
