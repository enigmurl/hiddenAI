from layers1D import *
from info1D import Vector
from info2D import Matrix
import costs
import pickle

class NeuralNet:
	def __init__(self,*layers):
		self.layers = list(layers)

	def add(self,layer):
		self.layers.append(layer)
	
	def vectorize(self, normalData):
		if type(normalData) != Vector and type(normalData) != Matrix:	
			return Vector(list(normalData)) if type(normalData[0]) != list else Matrix(list(normalData))
		else:
			return normalData

	def saveToFile(self,filename):
		with open(filename,"wb") as f:
			for layer in self.layers:
				if layer.config["type"] == "HIDDEN":
					pickle.dump(layer.weights,f)

	def openFromFile(self,filename):
		with open(filename,"rb") as f:
			for layer in self.layers:
				if layer.config["type"] == "HIDDEN":
					layer.weights = pickle.load(f)
	
	def run(self,inputLayer,fullReturn = False,debug = False):
		currentValue = inputLayer
		#if fullReturn:
		allValues = [inputLayer]	
		for ind,layer in enumerate(self.layers):
			currentValue = layer.run(currentValue)
			#if fullReturn:
			allValues.append(currentValue)
			if debug:
				print(allValues,currentValue)
		if fullReturn:
			if debug:
				print(allValues)
			return allValues
		
		return currentValue		

	def batchData(self,trainingData,batchSize):
		batchedData = []
		if len(trainingData)%batchSize !=0:
			for i in range (len(trainingData)//batchSize +1):
				batchedData.append(trainingData[i*batchSize:batchSize*(i+1)])
		else:
			for i in range (len(trainingData)//batchSize):
				batchedData.append(trainingData[i*batchSize:batchSize*(i+1)])
		return batchedData
	
	def descend(self,weightDerivatives): #weightDerivatives is a dictionary of {layer:derivative,layer2,derivative2....}
		for layer,derivative in weightDerivatives.items():
			layer.descend(derivative)
	
	def computeLossDerivative(self,allLayers,expected,costDerivativeFunction):
		endDerivative = costDerivativeFunction(allLayers[-1],expected)
		return endDerivative	

	def deriveOneData(self,weightDerivatives,currentDerivative,allLayers):
		for ind,layer in enumerate(self.layers[::-1]):
			inputLayer = allLayers[len(self.layers)-ind-1]
			if hasattr(layer,"weights"): #if it has weights
				layerDerivative = layer.derivative(inputLayer,currentDerivative)
				weightDerivatives[layer] += layerDerivative
					
			if ind != len(self.layers)-1:
				currentDerivative = layer.derivativePrevLayer(inputLayer,currentDerivative)
		return weightDerivatives

	def stochasticDescent(self,inputData,labels,epoch = 1, batchSize = 5,learningRate = 1,costFunction = costs.costMeanSquared):
		costDerivativeFunction = {costs.costMeanSquared:costs.derivativeCostMeanSquared}[costFunction]
		
		trainingData = list(zip(inputData,labels)) 
		batchedData = self.batchData(trainingData,batchSize)
		numItems = len(labels) 

		for epochNum in range(epoch):
			for batchNum,batch in enumerate(batchedData):
			
				weightDerivatives = {}
				for layer in self.layers:
					if hasattr(layer,"weights"):
						weightDerivatives[layer] = layer.blank() 
				totalCost = 0
				for data in batch:
					expected = data[1]
					allLayers = self.run(data[0],fullReturn = True)	
					currentDerivative = self.computeLossDerivative(allLayers,expected,costDerivativeFunction).scale(1/len(batch))
					self.weightDerivatives = self.deriveOneData(weightDerivatives,currentDerivative,allLayers)		
				self.descend(weightDerivatives)

				totalCost2 = 0
				for data in batch:
					expected = data[1]
					fullValues = self.run(data[0],fullReturn = True)
					cost =  costFunction(fullValues[-1],expected)
					totalCost2 +=  sum(cost.data)
				print(totalCost2)
				if (batchNum+1)%100 == 0 :
					print("BATCH#",batchNum, "OUT OF",len(batchedData))
if __name__ == "__main__":
	import time
	a = NeuralNet(FullyConnected(9,5),Bias(5,5),Sigmoid(),FullyConnected(5,2),Bias(2,2),Sigmoid())
	b = a.run(Vector([0,1,0,1,0,1,0,1,0]))

	print("RESULT:",b)
	trainingData = [
	Vector([1,0,1,0,1,0,1,0,1]),
	Vector([0,1,0,1,0,1,0,1,0]),
	Vector([1,1,1,1,0,1,1,1,1]),
	Vector([0,1,1,0,1,0,1,0,1]),
	Vector([1,1,1,1,0,1,0,1,0]),
	Vector([1,0,1,0,1,0,1,0,0]),
	Vector([0,1,0,1,0,1,1,1,1]),
	Vector([0,0,1,0,1,0,1,0,1])
	]
	labels = [Vector([0,1]),Vector([1,0]),Vector([1,0]),Vector([0,1]),Vector([1,0]),Vector([0,1]),Vector([1,0]),Vector([0,1])]
	startTime = time.time()
	numData = 100
	a.stochasticDescent(trainingData,labels,epoch = numData, batchSize = 8)
	b = a.run(Vector([0,1,0,1,0,1,0,1,0]))
	for ind,data in enumerate(trainingData):
		#b = a.run(Vector(data),debug=True,fullReturn = True)
		b = a.run(Vector(data))
		print("RESULT 2:",b,"EXPECTED RESULT",labels[ind])
	print("TOTAL TIME:",time.time()-startTime)
