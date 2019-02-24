from info1D import Vector
from info2D import Matrix

def derivativeCostMeanSquared(vect,expected):
	return (vect - expected).scale(2)
	
def costMeanSquared(vect,expected):
	partialCost = vect - expected
	cost =  partialCost*partialCost
	return cost

if __name__ == "__main__":
	pred = Vector([0,0])
	target = Vector([1,1])
	print("START COST",costMeanSquared(pred,target))
	for i in range(10):
		change = derivativeCostMeanSquared(pred,target)
		pred -= change 
		print(pred,change)
