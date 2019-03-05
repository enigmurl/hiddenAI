class MeanSquaredLoss:
	def __init__(self):
		pass	

	def derivative_prev_layer(self,predicted_output,expected_output,batch_size = 1):
		return (2/batch_size) * (predicted_output - expected_output)
	
	def apply_loss(self,predicted_output,expected_output,batch_size = 1):
		partial_loss = predicted_output - expected_output
		loss =  partial_loss*partial_loss
		return loss/batch_size

if __name__ == "__main__":
	pass
