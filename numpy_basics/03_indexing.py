import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

print("Original array:\n", arr)

# Basic indexing
print("\nElement at position (1,2):", arr[1, 2])

# Slicing
print("\nFirst two rows:\n", arr[:2])
print("\nAll rows, columns 1 to 3:\n", arr[:, 1:3])

# Boolean indexing
mask = arr > 5
print("\nElements greater than 5:\n", arr[mask]) 