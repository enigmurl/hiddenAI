from sequential import Sequential
from layers.main_layers import *
from layers.convolution import *
from layers.pooling import *
from layers.activations import *
import learning_rates
import optimizers
from mnist import MNIST
import random
import time

def asses_model(num_asses):
	score = 0
	for i in range(num_asses):
		choice = random.randint(0,num_data-1)
		real_image = formatted_inputs[choice]
		label_index = labels[choice]
		result = model.run(real_image)
		max_index = 0
		max_value = -1
		for ind,char in enumerate(result):
			if char > max_value:
				max_value = char
				max_index = ind
		for row in real_image[0]:
			new_line = ["#" if val > 0 else "." for val in row]
			print(" ".join(new_line))
		print("REAL",i, ":",character_map[label_index],"MACHINE:",character_map[max_index],"max-value",max_value)
		if max_index == label_index:
			print("correct")
			score += 1
	return score/num_asses

def rotate_image(image):
	
	flipped_image = np.fliplr(image)
	return np.rot90(flipped_image)

num_runs = 1
data_loader = MNIST("datasets/characterdata")
images,labels = data_loader.load_training()
print("IMAGES:" , len(images))
num_data =  697932 
images  = np.reshape(images[:num_data],(num_data,28,28))
images = [rotate_image(image) for image in images]
character_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
formatted_inputs = []
formatted_labels = []
start_time = time.time()
for ind,val in enumerate(images):
	output = np.zeros(62)
	#label_index  =  character_map.index(labels[ind])
	label_index = labels[ind]
	output[label_index] = 1
	new_image = np.array([[pxl/255 for pxl in images[ind]]])
	new_image = np.reshape(new_image,(1,28,28))
	formatted_labels.append(output)
	formatted_inputs.append(new_image)

#INITIALIZING THE MODEL + TRAINING
learning_rate = learning_rates.DecayLearningRate(0.1,1)
optimizer = optimizers.BatchGradientDescent(batch_size = 10,momentum = 0.9,learning_rate = learning_rate)
model = Sequential((1,28,28),
					Convolution2D(num_filters = 32,filter_size = (3,3),stride = (3,3)),
					MaxPooling2D(pooling_size = (2,2),stride = (2,2)),
					ReLU(), 
					Convolution2D(num_filters = 64,filter_size = (3,3),stride = (3,3)),
					MaxPooling2D(pooling_size = (2,2),stride = (2,2)),
					ReLU(), 
					#Convolution2D(num_filters = 128,filter_size = (2,2),stride = (2,2)),
					#MaxPooling2D(pooling_size = (2,2),stride = (2,2)),
					#ReLU(), 
					FullyConnected(128),
					Bias(),
					ReLU(),
					FullyConnected(62),
					Bias(),
					Softmax(),optimizer = optimizer)
#model.open_from_file("stored_weights/characterweights")
print("PRETRAIN:",asses_model(25))
model.train(formatted_inputs,formatted_labels, epoch = num_runs)
model.save_to_file("stored_weights/characterweights")

#ASSESING THE MODEL
print("POSTTRAIN:",asses_model(100))
print(time.time()-start_time)
