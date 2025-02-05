import numpy as np

# Create a sample array
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])
print("Original array:\n", arr)

# Fancy indexing
indices = np.array([0, 2])
print("\nFancy indexing (rows 0 and 2):\n", arr[indices])

# Combined indexing
print("\nCombined indexing (rows 0,2 cols 1,3):\n", 
      arr[indices][:, [1, 3]])

# Where function
condition = arr > 5
print("\nWhere arr > 5:", np.where(condition))
print("Values where arr > 5:", arr[np.where(condition)])

# Masking with conditions
mask = (arr > 5) & (arr < 10)
print("\nValues between 5 and 10:\n", arr[mask]) 