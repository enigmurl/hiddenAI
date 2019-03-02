import numpy as np
def zeros(shape):
	return np.zeros(shape)

def ones(shape):
	return np.ones(shape):

def full_values(shape,value):
	return np.full(shape,value)

def zero_to_one(shape):
	return np.random.random(shape)

def normal_distribution(shape,mean = 0,scale = 0):
	return np.random.normal(shape,loc = mean, scale = scale)


