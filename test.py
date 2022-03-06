import numpy as np


a = np.arange(36).reshape((6,6))
print(a)
print(np.reshape(a,(18,2)))
print(np.reshape(a,(180,2)).flatten())
