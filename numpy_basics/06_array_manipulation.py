import numpy as np

# Create a sample array
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("Original array:\n", arr)

# Concatenation
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
vertical = np.vstack((arr, arr2))
horizontal = np.hstack((arr, arr2))

print("\nVertical stack:\n", vertical)
print("\nHorizontal stack:\n", horizontal)

# Splitting arrays
print("\nSplitting arrays:")
arr3 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print("Split into 4:", np.split(arr3, 4))
print("Split at indices [3, 5]:", np.split(arr3, [3, 5]))

# Advanced reshaping
arr4 = np.arange(24)
reshaped = arr4.reshape(2, 3, 4)  # 3D array
print("\n3D reshape:\n", reshaped) 