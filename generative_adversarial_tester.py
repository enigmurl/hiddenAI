if __name__ == "__main__":
	from hiddenAI.generative_adversarial import *
	from hiddenAI.layers.main_layers import *
	from hiddenAI.layers.activations import *
	import hiddenAI.optimizers as optimizers
	import hiddenAI.learning_rates as learning_rates
	lr = learning_rates.ConstantLearningRate(0.1)
	discriminator = Sequential([1],FullyConnected(2),Bias(),Softmax(),optimizer = optimizers.BatchGradientDescent(batch_size = 4,learning_rate = lr))
	generator     = Sequential([1],FullyConnected(1),Bias(),optimizer = optimizers.BatchGradientDescent(learning_rate = lr))
	c = GAN(discriminator,generator)
	numdata = 10000
	#data = target + np.random.randn(numdata,1)
	data = np.random.randint(-100,100,size = (numdata,1))
	output = data**1/2 + 10
	print("first:",c.generator.run(np.random.random(size = (1))))
	c.train(output,training_inputs = data)
	final = c.generator.run(np.random.random(size = (1)))
	innum = 10
	print("FINAL:",c.generator.run(np.array([innum])))	
	print("DISCRIMINATOR",c.run_discriminator(final),c.run_discriminator(np.array([innum**2 + 10])),"EXPECTED:",[1,0],[0,1])
