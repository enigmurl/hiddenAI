import numpy as np

class MeanSquaredLoss:
	def __init__(self):
		pass	

	def derivative_prev_layer(self,predicted_output,expected_output,batch_size = 1):
		return (2/batch_size) * (predicted_output - expected_output)
	
	def apply_sing_loss(self,predicted_output,expected_output,batch_size = 1):
		partial_loss = predicted_output - expected_output
		loss =  partial_loss*partial_loss
		return loss/batch_size

class CrossEntropy:
	def __init__(self):
		pass
		
	def derivative_prev_layer(self,predicted_output,expected_output,batch_size = 1):
		predicted_clipped_output = np.clip(predicted_output,0.00000001,0.99999999)
		return -expected_output/(batch_size * predicted_clipped_output)
	
	def apply_single_loss(self,predicted_output,expected_output,batch_size = 1):
		predicted_clipped_output = np.clip(predicted_output,0.00000001,0.99999999)
		return -expected_output * np.log(predicted_clipped_output)/batch_size
		#return -(expected_output * np.log(predicted_clipped_output) + (1-expected_output)*np.log((1-predicted_clipped_output)))/batch_size	

if __name__ == "__main__":
	a = CrossEntropy()
	b = np.array([0.,0])
	c = np.array([1,0])
	print(a.apply_single_loss(b,c))
	for i in range(100):
		d =  - a.derivative_prev_layer(b,c,1)
		print(d)
		b = np.clip(b,0.0000001,0.99999999)
	print("B",b)		
	print(a.apply_single_loss(b,c))
