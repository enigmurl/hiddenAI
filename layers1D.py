from info1D import Vector
from info2D import Matrix
import math
import numpy as np
import random

#SQUASHING AND ACTIVATION FUNCTIONS
class Squashing:
	def __init__(self):
		self.config = {"dimension":1,"type":"SQUASHING"}

	def run (self,vecObject):
		data = vecObject.data
		returnVec = Vector(vecSize = len(vecObject))
		for ind in range(len(vecObject)):
			returnVec[ind] = self.apply(vecObject[ind])
		return returnVec

	def apply(self,x):
		pass# to be implemented in subclasses

	def derivativePrevLayer(self,x,currentDerivative):
		pass# to be implemented in subclasses

class Sigmoid(Squashing): #TESTED
	def __init__(self):
		super().__init__()
			
	def apply(self,x):
		return 1/(1 + math.e**-x) if abs(x) <30 else 0 if x<-30 else 1

	def derivativePrevLayer(self,vec,currentDerivative):
		return Vector([self.partialDerivative(x) for x in vec]) * currentDerivative

	def partialDerivative(self,x):
		partD = self.apply(x)
		return partD * (1-partD)

class BinaryStep(Squashing): #TESTED
	def __init__(self):
		super().__init__()

	def apply(self,x):
		return 1 if x>0 else 0

	def derivativePrevLayer(self,vec,currentDerivative):
		return Vector([1 if x == 0 else 0 for x in vec]) * currentDerivative

class ReLU(Squashing): #TESTED 
	def __init__(self):
		super().__init__()

	def apply(self,x):
		return max(0,x)

	def derivativePrevLayer(self,vec,currentDerivative):
		return Vector([1 if x>= 0 else 0 for x in vec]) * currentDerivative

class PReLU(Squashing): #TESTED
	def __init__(self,multiplier = 1/5):
		super().__init__()
		self.multiplier = multiplier
	
	def apply(self,x):
		return x if x>= 0 else self.multiplier*x

	def derivativePrevLayer(self,vec,currentDerivative):
		return Vector([1 if x>= 0 else self.multiplier for x in vec]) * currrentDerivative

class ELU(Squashing): #TESTED
	def __init__(self,multiplier = 1/5):
		super().__init__()
		self.multiplier = multiplier

	def apply(self,x):
		return x if x>=0 else self.multiplier*(math.e**x -1) if x>-30 else self.multiplier

	def derivativePrevLayer(self,vec,currentDerivative):
		return Vector([1 if x>=0 else self.apply(x) + self.multiplier for x in vec]) * currentDerivative

class Softplus(Squashing): #TESTED 
	def __init__(self):
		super().__init__()

	def apply(self,x):#natural log of(1+e^x)
		return math.log(1+math.pow(math.e,x))

	def derivativePrevLayer(self,vec,currentDerivative):#derivative is sigmoid
		return Vector([1/(1+math.e**-x) if abs(x)<30 else 0 if x<-30 else 1 for x in vec]) * currentDerivative
		
class Tanh(Squashing): #TESTED hyperbolic tangent
	def __init__(self):
		super().__init__()

	def apply(self,x):
		return math.tanh(x)

	def derivativePrevLayer(self,vec,currentDerivative):
		return Vector([1-self.apply(x)**2 for x in vec]) * currentDerivative

class Arctan(Squashing): #
	def __init__(self):
		super().__init__()

	def apply(self,x):
		return math.atan(self,x)

	def derivativePrevLayer(self,vec,currentDerivative):
		return Vector([1/(x**2 +1) for x in vec]) * currentDerivative

class MaxPooling(Squashing): # Pooling does change the size, but it has no weights, so it is best to keep it as a squashing function
	def __init__(self,poolingSize = 2 ,stride = 2):
		super().__init__()
		self.poolingSize = poolingSize 
		self.stride = 0
	
	def run(self,inVector):
		newVec = Vector([])
		start = 0
		while start + self.stride <len(inVector):
			newVec.append(max(inVector.data[start:self.poolingSize+start]))
			start += self.stride
		return newVec	
	
	def derivativePrevLayer(self,vec,currentDerivative):# may want to optimize
		inDerivatives = Vector(data = 0, vecSize = len(vec))
		start = 0
		while start + self.stride < len(inVector):
			actual = max(inVector.data[start:self.poolingSize+start])
			for ind,val in enumerate(inVector.data[start:self.poolingSize+start]):
				if val == actual:
					inDerivatives[start+ind] += currentDerivative[start] 
			stride += self.stride
		return Vector(inDerivatives)

class AvgPooling(Squashing):
	def __init__(self,poolingSize):
		super().__init__()
		self.poolingSize = poolingSize 

	def run(self,inVector):
		newVec = Vector([])
		start = 0
		while start + self.stride <len(inVector):
			newVec.append(sum(inVector.data[start:self.poolingSize+start])/self.poolingSize)
			self.stride += 1
		return newVec	

	def derivativePrevLayer(self,vec,currentDerivative):
		inDerivatives = Vector(data = 0,vecSize = len(currentDerivative))
		start = 0
		while start + self.stride <len(inVector):
			for ind,val in enumerate(inVector.data[start:self.poolingSize+start]):
				inDerivatives[start+ind] += currentDerivative[start]/self.poolingSize	
		return Vector(inDerivatives)
	
#May want to add pooling
 
#MAIN AND HIDDEN LAYERS
class Hidden:
	def __init__(self,inputsize,outputsize):
		self.config = {"dimension":1,"type":"HIDDEN"}
		self.inputsize = inputsize
		self.outputsize = outputsize

	def run(self,inputLayer): #may be over rided if the layer is 1d to 2d, or a different form
		pass# overided as they differ heavily

	def derivative(self, inputLayer,currentDerivative):
		pass# to be overrided

	def derivativePrevLayer(self,inputLayer, currentDerivative): # how does the previous layer affect the next
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
		self.weights = Vector(randomValues = True,vecSize = inputsize) 
		self.numWeights = inputsize

	def run(self,inputLayer):
		return inputLayer + self.weights 

	def derivative(self,inputLayer, currentDerivative):
		return currentDerivative

	def derivativePrevLayer(self,inputLayer,currentDerivative): 
		return currentDerivative # the derivative of z= x+y for x is 1 and for y is 1 as well, so multiply it by the current derivative and its just current derivative 
	
	def blank(self):
		return Vector(vecSize = self.inputsize, data = 0)
	

class FullyConnected(Hidden):#TESTED - FIND BETTER/ ACTUAL NAME
	def __init__(self, inputsize, outputsize):
		super().__init__(inputsize,outputsize)	
		self.weights = Matrix(rows = outputsize,cols = inputsize,randomValues = True)
		#self.weights = self.blank()
		self.numWeights = inputsize*outputsize
	
	def run(self,inputLayer):
		return self.weights * inputLayer
	
	def derivative(self,inputLayer,currentDerivative):
		return Matrix(data = [[inputLayer[x]*currentDerivative[y] for x in range(self.inputsize)] for y in range(self.outputsize)])
	
	def derivativePrevLayer(self,inputLayer, currentDerivative): # we dont need an input layer for this derivative, but we keep it so all derivativePrevLayers can be called the same
		vecData = []
		for x in range(self.inputsize):
			vecData.append(sum([self.weights[y][x] * currentDerivative[y] for y in range(self.outputsize)]))
		return Vector(vecData)
	
	def blank(self): 
		return Matrix(rows = self.outputsize, cols = self.inputsize, data = 0)

#POTENTIAL TO ADD: CONVULTION, OTHER STUFF

if __name__ == "__main__":
	print("TESTING SQUASHING FUNCTIONS")
	inputVec = Vector([-1,0])
	currentderivative = Vector([2,8])
	target = Vector([0,5])
	lyer = FullyConnected(2,2)
	for i in range(10):
		result = lyer.run(inputVec)
		newDerivative = (result-target).scale(1/75)
		aDerivative = lyer.derivativePrevLayer(inputVec,newDerivative)
		print(result,aDerivative)
		inputVec -= aDerivative

	
