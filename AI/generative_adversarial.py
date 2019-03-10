from sequential import Sequential()

class GAN(Sequential):
	def __init__(self,discriminator, generator):
		self.discriminator = discriminator 
		self.generator = generator
		if self.generator.layers[-1].output_shape != self.discriminator[0].input_shape:
			raise ValueError("Generator output shape",self.generator.layers[-1].output_shape, "is not the same as the Discriminators input shape",self.discriminator[0].input_shape)

	def train(self,real_images):
		
	def run_generator(self,input_layer):
		return self.generator.run(

	def run_discriminator(self,input_layer):	
