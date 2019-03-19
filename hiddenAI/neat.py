import random
import copy
import numpy as np
from hiddenAI.sequential import Sequential

class NEAT:
	def __init__(self,model,num_per_generation = 10):
		self.model = model
		self.num_per_generation = num_per_generation
		self.generation = 0

	def initiate_nets(self):
		return [self.model] * self.num_per_generation

	def find_parents(self,nets,scores):#finds the top 2		
		max_score = -1
		sec_score = -1
		max_net = -1
		sec_net = -1
		for ind,score in enumerate(scores):
			if score> max_score:
				max_score = score
				max_net = nets[ind]
			elif score>sec_score:
				sec_score = score
				sec_net = nets[ind]
		return max_net,sec_net

	def new_generation(self,old_generation,scores,mutation_rate = 0.1,mutation_max_size = 1):#nets should be a list of Neural Networks, scores should be how well each neural network does, the kwargs are to do with the severity of mutation rates and how big they are,mutationRate is 0 is 0% and 1 is 100% 
		self.generation += 1
		net1,net2 = self.find_parents(old_generation,scores)# find the nets we want to breed
		self.model = net1#make the model the best 
		return_models = [net1,net2] 
		for model_num in range(self.num_per_generation-2):
			return_models.append(self.breed(net1,net2,mutation_rate,mutation_max_size))		
		return return_models
		
	def breed(self,net1,net2,mutation_rate,mutation_max_size):
		layers = []
		for layer_num,layer in enumerate(net1):
			if hasattr(layer,"weights"):#if it has weights
				new_layer = copy.deepcopy(layer)
				mask = np.random.choice((True, False), size = layer.weights.shape)# breed
				new_layer.weights[mask] = net2[layer_num].weights[mask]
					
				mutations =  mutation_max_size * (2*np.random.random(layer.weights.shape)-1)#can optimize this
				mutation_mask = np.random.choice((True,False),size = layer.weights.shape, p = (mutation_rate,1-mutation_rate))
				blank_mutation = np.zeros(layer.weights.shape)
				blank_mutation[mutation_mask] = mutations[mutation_mask]#mutations are no longer blank
				new_layer.weights += blank_mutation
				layers.append(new_layer)
			else:
				layers.append(layer)
		net =  Sequential(layers[0].input_shape,optimizer = net1.optimizer)
		net.layers = layers
		return net

	def open_saved_model(self,fileName):
		self.model.open_from_file(fileName)
	
	def save_model_to_file(self,fileName):
		self.model.save_to_file(fileName)
	
if __name__ == "__main__":
	from sequential import  Sequential
	from layers.main_layers import * 
	from layers.activations import *
	net1 =  Sequential(4,FullyConnected(2),Bias(),Sigmoid()) 
	neat = NEAT(net1)
	net2 =  Sequential(4,FullyConnected(2),Bias(),Sigmoid()) 
	for layer in net1:
		print(layer) 
	for layer in net2:
		print(layer) 
	for layer in neat.breed(net1,net2,0.1,2):
		print(layer) 
