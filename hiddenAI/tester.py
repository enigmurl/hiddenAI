import numpy as np
class First:
	def __init__(self):
		self.printMRO()
	def printMRO(self):
		print("first")
class Second(First):
	def __init__(self):
		super().__init__()
	def printMRO(self):
		print("second")
		
a = Second()
