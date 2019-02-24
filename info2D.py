from info import Info
from info1D import Vector
import random 
import numpy as np

class Matrix(Info):
	def __init__(self, data = 0, rows = 2, cols = 2 , randomValues = False):
		if randomValues:
			self.data = [[np.random.randn() for _ in range(cols)] for _ in range(rows)]
		elif type(data) == int:
			self.data = [[data for _ in range(cols)] for _ in range(rows)]
		else:
			self.data =  data
		self.infoType = True	

	def totalSum(self):
		return sum([sum(row) for row in self])

	def scale(self,num):
		return Matrix([[val*num for val in row] for row in self])
	
	def __mul__(self,mtx2):
		'''
		structure = np.matmul(self.data,mtx2.data)
		if type(mtx2) == Vector:
			return Vector(structure)
		else:
			return Matrix(structure)
		'''
		output = []#may want to optimize by making this a numpy array
		if type(mtx2) == Vector:
			for row in self.data:
				runSum = 0
				for y,col in enumerate(row):
					runSum+=col*mtx2[y]
				output.append(runSum)	
			return Vector(output)
		elif type(mtx2) == Matrix:
			for row in self.data:
				output.append([])
				for x in range(self.__len__()):
					runSum = 0
					for y,col in enumerate(row):
						runSum+=col*mtx2[y][x]
						print(x,y,col,mtx2[y][x])
					output[-1].append(runSum)	
			return Matrix(output)
	def __add__(self,mtx2):
		return Matrix([[col+mtx2[x][y] for y,col in enumerate(row) ] for x,row in enumerate(self)])

	def __sub__(self,mtx2):
		return Matrix([[col-mtx2[x][y] for y,col in enumerate(row) ] for x,row in enumerate(self)])
	
	def __getitem__(self,ind):
		return self.data[ind]

	def __len__(self):
		return len(self.data)
	
	def __repr__(self):
		return str(self.data)
	
	def __iter__(self):
		return self.data.__iter__()
if __name__ == "__main__":
	a = Matrix(data = np.array([[2,0],[1,2]]))
	b = Matrix(data = np.array([[1,2],[3,4]]))
	print(b*a)
