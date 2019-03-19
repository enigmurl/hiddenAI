import numpy as np
class QLearning:#WORK ON SPECIFICS 
	
	def __init__(self,number_of_states,number_of_actions,chance_of_exploration = 0.1):
		self.chance_of_exploration = chance_of_exploration
		self.qtable = np.ones((number_of_states,number_of_actions))
		self.number_of_actions = number_of_actions
		self.number_of_states = number_of_actions
		self.last_action = 0
		self.last_state = 0	

	def policy(self,state):#returns the action to take
		action_values = self.qtable[state] 
		min_action_value = np.min(action_values)
		positive_values = action_values - (min_action_value-self.chance_of_exploration)
		sum_action_values = np.sum(positive_values)
		chosen_value = np.random.choice(self.number_of_actions,p = positive_values/sum_action_values) 
		self.last_state = state
		self.last_action = chosen_value
		return chosen_value
		
	def reward(self,reward_magnitude):
		self.qtable[self.last_state,self.last_action] += reward_magnitude

	def __repr__(self):
		return self.qtable.__repr__()

#RESEARCH Q LEARNING
if __name__ == "__main__":
	a = QLearning(1,2)
	state = 0
	for i in range(10):
		action = a.policy(0)
		print(a,action)
		
		if action == 1:
			a.reward(-2)
		else:
			a.reward(2)
	print(a)
