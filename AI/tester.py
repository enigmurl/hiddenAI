import numpy as np
a = np.array([True,False])
b = np.array([2,1])
c = np.array([1,2])
b[a] = c[a]
print(b)
