#TO BE PUT IN OPTIMIZERS OR IN SEQUENTIAL OBJECT: GRADIENT DESCENT AND STOCHASTIC GRADIENT DESCEND
def stochastic_gradient_descent(self,input_data,expected_outputs,epoch = 1,learning_rate = 1,print_epochs = True):
    loss_function_derivatives = self.loss.derivative_prev_layer

    training_data = list(zip(input_data,expected_outputs))

    for epoch_num in range(epoch):
	if print_epochs:
	    progress_bar = ProgressBar(start_message ="EPOCH NUM " + str(epoch_num) + " ",total = len(training_data), total_chars = 100)
	for data in training_data:
	    expected = data[1]
	    all_layers = self.training_run(data[0])
            current_derivative = learning_rate * self.compute_loss_derivative(all_layers,expected,loss_function_derivatives,batch_size = len(training_data))
	    self.derive_and_descend_one_data(current_derivative,all_layers)	#see if we need to make it self	
	    if print_epochs:
		progress_bar.update()
			


def gradient_descent(self,input_data,expected_outputs,epoch = 1, learning_rate = 0.001,print_epochs = True):
    loss_function_derivatives = self.loss.derivative_prev_layer
		
    training_data = list(zip(input_data,expected_outputs))

    for epoch_num in range(epoch):
	weight_gradients = self.blank_weights()
	progress_bar = ProgressBar(start_message ="EPOCH NUM " + str(epoch_num) + " ",total = len(training_data), total_chars = 100)
	for data in training_data:
	    expected = data[1]
	    all_layers = self.training_run(data[0])	
	    current_derivative = learning_rate * self.compute_loss_derivative(all_layers,expected,loss_function_derivatives,batch_size = len(training_data))
	    weight_gradients = self.derive_one_data(weight_gradients,current_derivative,all_layers)	#see if we need to make it self	
	    if print_epochs:
		progress_bar.update()
	self.descend(weight_gradients)


