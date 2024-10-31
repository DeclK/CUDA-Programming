import matplotlib.pyplot as plt 
import numpy as np

# load input.txt
data = np.loadtxt('input.txt')

# data is a (128,) array
# plot data in scatter
length = len(data)
plt.scatter(np.arange(0, length), data, s=8)

plt.show()