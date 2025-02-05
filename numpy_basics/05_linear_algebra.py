import numpy as np

# Create matrices
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print("Matrix A:\n", A)
print("\nMatrix B:\n", B)

# Matrix operations
print("\nMatrix multiplication:\n", np.dot(A, B))
print("\nMatrix transpose:\n", A.T)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("\nEigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Solve linear equations: Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
print("\nSolution to linear equations:")
print("x =", x) 