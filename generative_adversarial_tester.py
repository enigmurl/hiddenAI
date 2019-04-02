import numpy as np
def noise(size = 1,a=1):
	return np.random.uniform(-a,a,size= size)
if __name__ == "__main__":
	import matplotlib.pyplot as plt
	from hiddenAI.generative_adversarial import *
	from hiddenAI.layers.main_layers import *
	from hiddenAI.layers.activations import *
	import hiddenAI.optimizers as optimizers
	import hiddenAI.loss as loss
	import hiddenAI.learning_rates as learning_rates
	los = loss.BinaryCrossEntropy()
	lr = learning_rates.ConstantLearningRate(0.01)
	lr2 = learning_rates.ConstantLearningRate(0.01)
	bsize = 250
	discriminator = Sequential([2],FullyConnected(8),Bias(),RReLU(),FullyConnected(8),Bias(),RReLU(),FullyConnected(2),Bias(),Sigmoid(),optimizer = optimizers.BatchGradientDescent(batch_size = bsize,learning_rate = lr2,momentum = 0.9),loss=los)
	generator     = Sequential([2],FullyConnected(6),Bias(),FullyConnected(6),Bias(),FullyConnected(2),optimizer = optimizers.BatchGradientDescent(learning_rate = lr,batch_size = bsize,momentum = 0.9))
	c = GAN(discriminator,generator,noise = noise)
	numdata = 256000
	#data = target + np.random.randn(numdata,1)
	data = np.random.uniform(-10,10,size = (numdata,1))
	output = data**2 + 10
	plt.plot(data,output,"ro")
	vals = [c.run_generator() for _ in data]
	plt.plot([val[0] for val in vals],[val[1] for val in vals],"go")
	invals = [np.concatenate((val,output[ind])) for ind,val in enumerate(data)]
	c.train(invals)
	innum = 5
	final = c.run_generator()
	print(final,c.run_discriminator(final),c.run_discriminator(np.array([innum,innum**2 + 10])))
	vals = [c.run_generator() for _ in data]
	plt.plot([val[0] for val in vals],[val[1] for val in vals],"bo")
	c.train(invals)
	vals = [c.run_generator() for _ in data]
	plt.plot([val[0] for val in vals],[val[1] for val in vals],"yo")
	plt.show()
	final = c.run_generator()
	print(final,c.run_discriminator(final),c.run_discriminator(np.array([innum,innum**2 + 10])))
