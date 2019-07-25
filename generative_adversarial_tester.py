import numpy as np
import math 
def noise(size = [1],a=0.1):
	return np.random.uniform(-a,a,size= size)
def getdata(vals,data,output):
	a = []
	b = []
	for ind,i in enumerate(data):
		l = np.random.choice([0,1])
		if l:
			a.append(np.concatenate((vals[ind],i)))
			b.append(np.array([1,0]))
		else:
			a.append(np.concatenate((output[ind],i)))
			b.append(np.array([0,1]))
	return a,b 
def getdloss(discriminator,data,outputs):
	num = len(data)
	loss = 0
	for i,dat in enumerate(data):
		runed = discriminator.run(dat)
		loss += discriminator.loss.apply_single_loss(runed,outputs[i],batch_size = num)
	try:
		return sum(loss)
	except:
		return loss
def getgloss(generator,discriminator,data):
	num = len(data)
	loss = 0
	for i,dat in enumerate(data):
		runed = generator.run(dat)
		druned = discriminator.run(np.concatenate((runed,dat[2:])))
		loss += discriminator.loss.apply_single_loss(druned,np.array([0,1]),batch_size = num)
	try:
		return sum(loss)
	except:
		return loss
if __name__ == "__main__":
	import matplotlib.pyplot as plt
	from hiddenAI.generative_adversarial import *
	from hiddenAI.layers.main_layers import *
	from hiddenAI.layers.activations import *
	import hiddenAI.optimizers as optimizers
	import hiddenAI.loss as loss
	import hiddenAI.learning_rates as learning_rates
	los = loss.MeanSquaredLoss()
	los =  loss.BinaryCrossEntropy()
	los2 = loss.MeanSquaredLoss()
	lr = learning_rates.ConstantLearningRate(0.001)
	lr2 = learning_rates.ConstantLearningRate(0.002)
	bsize = 32
	bsize2 = 32
	discriminator = Sequential([2],FullyConnected(4),Bias(),Tanh(),FullyConnected(4),Bias(),Tanh(),FullyConnected(2),Bias(),Softmax(),optimizer = optimizers.BatchGradientDescent(batch_size = bsize,learning_rate = lr,momentum = 0.9),loss=los)
	generator     = Sequential([3],FullyConnected(4),Bias(),Tanh(),FullyConnected(4),Bias(),Tanh(),FullyConnected(1),Bias(),Tanh(),optimizer = optimizers.BatchGradientDescent(learning_rate = lr2,batch_size = bsize2,momentum = 0.9),loss = los2)
	c = ConditionalGAN(discriminator,generator,[2],noise = noise)
	innum = np.array([1])
	innu2 = np.array([-0.5])
	numdata = 2560
	data = np.random.uniform(-1,1,size = (numdata,1))
	output = np.sin(np.pi*data)
	ntraindata = np.concatenate((noise((numdata,2)),data),axis = 1)
	final = c.run_generator(innum)
	fina2 = c.run_generator(innu2)
	print(final,c.run_discriminator(final,innum),c.run_discriminator(np.pi*np.sin(innum),innum))
	print(fina2,c.run_discriminator(fina2,innu2),c.run_discriminator(np.pi*np.sin(innu2),innu2))
	
	#generator.train(ntraindata,output)
	print(final,c.run_discriminator(final,innum),c.run_discriminator(np.pi*np.sin(innum),innum))
	
	#discriminator.train(dtraindata,answers)
	numloss = 100
	rvals = []
	lloss = 123
	for i in range(numloss):
		vals = np.array([c.run_generator(cond) for cond in data])
		dtraindata,answers = getdata(vals,data,output)
		c.train(output[:2560],data[:2560])
		a =getgloss(generator,discriminator,ntraindata[:2560])
		b= getdloss(discriminator,dtraindata[:2560],answers[:2560])	
		print("DISCRIMINATOR:",i,b)
		print("GENERATIORR  :",i,a)
		if a<lloss:
			lloss = a
			rvals = vals
	final = c.run_generator(innum)
	print(final,c.run_discriminator(final,innum),c.run_discriminator(np.pi*np.sin(innum),innum))
	vals = rvals 
	vals = np.array([c.run_generator(cond) for cond in data])
	dtraindata,answers = getdata(vals,data,output)
	plt.plot(data,vals,"go")
	
	vals = [[cond,c.run_generator(cond)] for cond in data]
	plt.plot([val[0] for val in vals],[val[1] for val in vals],"bo")
	#generator.train(ntraindata,output)
	
	numloss = 100
	for i in range(numloss):
		vals = np.array([c.run_generator(cond) for cond in data])
		dtraindata,answers = getdata(vals,data,output)
		c.train(output[:2560],data[:2560])	
		print("DISCRIMINATOR:",i,getdloss(discriminator,dtraindata[:2560],answers[:2560]))
		print("GENERATIORR  :",i,getgloss(generator,discriminator,ntraindata[:2560]))
	
	vals = [[cond,c.run_generator(cond)] for cond in data]
	plt.plot([val[0] for val in vals],[val[1] for val in vals],"yo")
	plt.plot(data,output,"ro")
	final = c.run_generator(innum)
	print(final,c.run_discriminator(final,innum),c.run_discriminator(np.pi*np.sin(innum),innum))
	plt.show()
