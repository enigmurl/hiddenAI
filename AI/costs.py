
def derivative_mean_squared_cost(predicted_output,expected_output,batch_size = 1):
	return (2/batch_size) * (predicted_output - expected_output)
	
def mean_squared_cost(predicted_output,expected_output):
	partial_cost = predicted_output - expected_output
	cost =  partial_cost*partial_cost
	return cost

if __name__ == "__main__":
	pass
