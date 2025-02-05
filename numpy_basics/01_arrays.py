import numpy as np

# Creating arrays
print("Basic Array Creation:")
# From Python list
array1 = np.array([1, 2, 3, 4, 5])
print("From list:", array1)

# Using numpy functions
zeros = np.zeros(5)
ones = np.ones(3)
print("\nZeros array:", zeros)
print("Ones array:", ones)

# Range of numbers
range_array = np.arange(0, 10, 2)  # start, stop, step
print("\nRange array:", range_array)

# Reshape arrays
matrix = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
print("\nReshaped into 2x3 matrix:\n", matrix) 