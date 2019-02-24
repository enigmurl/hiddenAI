from info import Info
import numpy as np
import random
import sys

class Vector(Info):
	
	def __init__(self,data = 0,vecSize = 0,randomValues = False):
		if randomValues:
			self.data = [np.random.randn() for _ in range(vecSize)]
		elif type(data) == int:
			self.data = [data for _ in range(vecSize)]
		else:
			self.data = data
		self.infoType = True

	def totalSum(self):
		return sum(self.data)	

	def append(self,val):
		self.data.append(val)	

	def scale(self,val):
		return Vector([val*itm for itm in self])

	def __len__(self):
		return len(self.data)

	def __getitem__(self,ind):
		return self.data[ind]

	def __setitem__(self,ind,val):
		self.data[ind] = val

	def __mul__(self,vec):
		return Vector([val*vec[ind] for ind,val in enumerate(self)])

	def __add__(self,vec):
		return Vector([val+vec[ind] for ind,val in enumerate(self)])
	
	def __sub__(self,vec):
		return Vector([val-vec[ind] for ind,val in enumerate(self)])

	def __iter__(self):
		return self.data.__iter__()
	
	def __repr__(self):
		return self.data.__repr__()
