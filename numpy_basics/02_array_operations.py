import numpy as np

# Basic array operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Array Operations:")
print("a =", a)
print("b =", b)
print("\nAddition:", a + b)
print("Multiplication:", a * b)
print("Square root:", np.sqrt(a))
print("Exponential:", np.exp(a))

# Broadcasting
matrix = np.array([[1, 2, 3],
                  [4, 5, 6]])
scalar = 2

print("\nBroadcasting:")
print("Original matrix:\n", matrix)
print("Matrix * scalar:\n", matrix * scalar) 