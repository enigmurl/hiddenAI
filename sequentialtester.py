if __name__ == "__main__":
	import time
	from hiddenAI.sequential import *
	from hiddenAI.layers.activations import *
	from hiddenAI.layers.convolution import *
	from hiddenAI.layers.pooling import *
	from hiddenAI.layers.main_layers import *
	from hiddenAI.layers.dropout import *
	from hiddenAI.loss import *
	from hiddenAI.layers.convertors import *
	import hiddenAI.optimizers as optimizers
	import hiddenAI.learning_rates as learning_rates
	sig = Sigmoid()
	sig.config["dimension"] = 1
	lrning_rate = learning_rates.ConstantLearningRate(0.1)
	a = Sequential((1,1,9),
				Convolution2D(num_filters = 3,filter_size = [2,2],stride = [2,2],pad = 2),
				Bias(),
				ReLU(),
				#Dropout(0.2),
				MaxPooling2D(pooling_size = [2,2]),
				Convolution2D(num_filters = 2,filter_size = [3,3],stride = [2,2],pad = 2),
				Bias(),
				ReLU(),
				##ELU(),
				#Dropout(0.5),
				MaxPooling2D(pooling_size  = [2,2]),
				FullyConnected(2),
				Bias(),
				Softmax(),optimizer = optimizers.BatchGradientDescent(batch_size = 32,learning_rate = lrning_rate),loss=BinaryCrossEntropy())
	'''
	a = Sequential((9),
				FullyConnected(6),
				Bias(),
				Sigmoid(),
				FullyConnected(2),
				Bias(),
				Sigmoid())
	'''
	trainingData =np.array( [
	[[[1,0,1,0,1,0,1,0,1]]],
	[[[0,1,0,1,0,1,0,1,0]]],
	[[[1,1,1,1,0,1,1,1,1]]],
	[[[0,1,1,0,1,0,1,0,1]]],
	[[[1,1,1,1,0,1,0,1,0]]],
	[[[1,0,1,0,1,0,1,0,0]]],
	[[[0,1,0,1,0,1,1,1,1]]],
	[[[0,0,1,0,1,0,1,0,1]]]
	])
	#trainingData = np.reshape(trainingData,(8,9))
	#trainingData = np.reshape(trainingData,(8,1,9))
	print(a.run(trainingData[0]))
	labels = [np.array([0,1]),np.array([1,0]),np.array([1,0]),np.array([0,1]),np.array([1,0]),np.array([0,1]),np.array([1,0]),np.array([0,1])]
	startTime = time.time()
	numData = 100
	for i in range(10):
		for ind,data in enumerate(trainingData):
			b = a.run(data)
			print("RESULT:",i,b,"EXPECTED RESULT",labels[ind])
		a.train(trainingData,labels,epoch = numData,print_epochs = False)
		print("\n")
	print("TOTAL TIME:",time.time()-startTime)
