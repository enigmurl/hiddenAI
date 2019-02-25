import math
import numpy as np
import random

#SQUASHING AND ACTIVATION FUNCTIONS
class Squashing:
	def __init__(self):
		self.config = {"dimension":1,"type":"SQUASHING"}

	def run (self,input_layer):
		output_layer = []
		for ind in range(len(input_layer)):
			output_layer.append(self.apply(input_layer[ind]))
		return np.array(output_layer)

	def apply(self,x):
		pass# to be implemented in subclasses

	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		pass# to be implemented in subclasses

class Sigmoid(Squashing): #TESTED
	def __init__(self):
		super().__init__()
			
	def apply(self,x):
		return 1/(1 + math.e**-x) if abs(x) <30 else 0 if x<-30 else 1

	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		return np.array([self.partialDerivative(x) for x in input_layer]) * output_layer_derivative

	def partialDerivative(self,x):
		partD = self.apply(x)
		return partD * (1-partD)

class BinaryStep(Squashing): #TESTED
	def __init__(self):
		super().__init__()

	def apply(self,x):
		return 1 if x>0 else 0

	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		return np.array([1 if x == 0 else 0 for x in input_layer]) * output_layer_derivative

class ReLU(Squashing): #TESTED 
	def __init__(self):
		super().__init__()

	def apply(self,x):
		return max(0,x)

	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		return np.array([1 if x>= 0 else 0 for x in input_layer]) * output_layer_derivative

class PReLU(Squashing): #TESTED
	def __init__(self,multiplier = 1/5):
		super().__init__()
		self.multiplier = multiplier
	
	def apply(self,x):
		return x if x>= 0 else self.multiplier*x

	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		return np.array([1 if x>= 0 else self.multiplier for x in input_layer]) * output_layer_derivative

class ELU(Squashing): #TESTED
	def __init__(self,multiplier = 1/5):
		super().__init__()
		self.multiplier = multiplier

	def apply(self,x):
		return x if x>=0 else self.multiplier*(math.e**x -1) if x>-30 else self.multiplier

	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		return np.array([1 if x>=0 else self.apply(x) + self.multiplier for x in input_layer]) * output_layer_derivative

class Softplus(Squashing): #TESTED 
	def __init__(self):
		super().__init__()

	def apply(self,x):#natural log of(1+e^x)
		return math.log(1+math.pow(math.e,x))

	def derivative_prev_layer(self,input_layer,output_layer_derivative):#derivative is sigmoid
		return np.array([1/(1+math.e**-x) if abs(x)<30 else 0 if x<-30 else 1 for x in input_layer]) * output_layer_derivative
		
class Tanh(Squashing): #TESTED hyperbolic tangent
	def __init__(self):
		super().__init__()

	def apply(self,x):
		return math.tanh(x)

	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		return np.array([1-self.apply(x)**2 for x in input_layer]) * output_layer_derivative

class Arctan(Squashing): #
	def __init__(self):
		super().__init__()

	def apply(self,x):
		return math.atan(self,x)

	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		return np.array([1/(x**2 +1) for x in input_layer]) * output_layer_derivative

class MaxPooling(Squashing): # Pooling does change the size, but it has no weights, so it is best to keep it as a squashing function
	def __init__(self,pooling_size = 2 ,stride = 2):
		super().__init__()
		self.pooling_size = pooling_size 
		self.stride = 0
	
	def run(self,input_layer):
		output_layer = []
		start = 0
		while start + self.stride <len(input_layer):
			output_layer.append(max(input_layer[start:self.pooling_size+start]))
			start += self.stride
		return np.array(output_layer)	
	
	def derivative_prev_layer(self,input_layer,output_layer_derivative):# may want to optimize
		in_derivatives = np.zeros(len(input_layer)) 
		start = 0
		while start + self.stride < len(input_layer):
			actual = max(input_layer[start:self.pooling_size+start])
			for ind,val in enumerate(input_layer[start:self.pooling_size+start]):
				if val == actual:
					in_derivatives[start+ind] += output_layer_derivative[start] 
			stride += self.stride
		return in_derivatives

class AvgPooling(Squashing):
	def __init__(self,pooling_size):
		super().__init__()
		self.pooling_size = pooling_size 

	def run(self,input_layer):
		output_layer = []
		start = 0
		while start + self.stride <len(input_layer):
			output_layer.append(sum(input_layer[start:self.pooling_size+start])/self.poolingSize)
			self.stride += 1
		return np.array(output_layer)	

	def derivative_prev_layer(self,input_layer,output_layer_derivative):
		in_derivatives = np.zeros(len(input_layer)) 
		start = 0
		while start + self.stride <len(input_layer):
			for ind,val in enumerate(input_layer[start:self.pooling_size+start]):
				in_derivatives[start+ind] += output_layer_derivative[start]/self.pooling_size	
		return in_derivatives
	
#May want to add pooling
 
#MAIN AND HIDDEN LAYERS
class Hidden:
	def __init__(self,inputsize,outputsize):
		self.config = {"dimension":1,"type":"HIDDEN"}
		self.inputsize = inputsize
		self.outputsize = outputsize

	def run(self,input_layer): #may be over rided if the layer is 1d to 2d, or a different form
		pass# overided as they differ heavily

	def derivative(self, input_layer,output_layer_derivative):
		pass# to be overrided

	def derivative_prev_layer(self,input_layer, output_layer_derivative): # how does the previous layer affect the next
		pass# to be overrided
		
	def descend(self, derivatives): 
		self.weights -= derivatives
	
	def blank(self):
		pass # to be overided

	def __repr__(self):
		return str(self.weights)

class Bias(Hidden):#TESTED - while bias resembles an activation function as the size doesnt change, it does have variables so it is put in hidden
	def __init__(self, inputsize, outputsize = None): 
		super().__init__(inputsize,outputsize)
		self.weights = np.random.randn(inputsize) 
		self.num_weights = inputsize

	def run(self,input_layer):
		return input_layer + self.weights 

	def derivative(self,input_layer, output_layer_derivative):
		return output_layer_derivative

	def derivative_prev_layer(self,input_layer,output_layer_derivative): 
		return output_layer_derivative # the derivative of z= x+y for x is 1 and for y is 1 as well, so multiply it by the current derivative and its just current derivative 
	
	def blank(self):
		return np.zeros(self.inputsize)	

class FullyConnected(Hidden):#TESTED - FIND BETTER/ ACTUAL NAME
	def __init__(self, inputsize, outputsize):
		super().__init__(inputsize,outputsize)	
		self.weights = np.random.randn(outputsize,inputsize)	
		self.numWeights = inputsize*outputsize
	
	def run(self,input_layer):
		return np.matmul(self.weights,input_layer)
	
	def derivative(self,input_layer,output_layer_derivative):
		return np.array([[input_layer[x]*output_layer_derivative[y] for x in range(self.inputsize)] for y in range(self.outputsize)])
	
	def derivative_prev_layer(self,input_layer, output_layer_derivative): # we dont need an input layer for this derivative, but we keep it so all derivativePrevLayers can be called the same
		#input_layer_derivatives = []
		#for x in range(self.inputsize):
		#	input_layer_derivatives.append(sum([self.weights[y][x] * output_layer_derivative[y] for y in range(self.outputsize)]))
		#return np.array(input_layer_derivatives)
		return np.matmul(np.swapaxes(self.weights,0,1),output_layer_derivative)
	def blank(self):
		return np.zeros((self.outputsize,self.inputsize)) 

#POTENTIAL TO ADD: CONVULTION, OTHER STUFF

if __name__ == "__main__":
	print("TESTING SQUASHING FUNCTIONS")
	inputVec = np.array([-1,0])
	currentderivative = np.array([2,8])
	target = np.array([0,5])
	lyer = FullyConnected(2,2)
	for i in range(10):
		result = lyer.run(inputVec)
		newDerivative = (result-target)*(1/75)
		aDerivative = lyer.derivative_prev_layer(inputVec,newDerivative)
		print(result,aDerivative)
		inputVec -= aDerivative

	
