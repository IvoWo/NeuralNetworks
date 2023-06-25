import numpy as np


sizes = [2, 3, 2]
biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

print(biases)
w = [(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
print(w)
print(weights)