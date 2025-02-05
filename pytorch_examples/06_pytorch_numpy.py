import torch
import numpy as np

# 1. Converting between NumPy arrays and PyTorch tensors
print("NumPy to PyTorch conversion:")
# NumPy array to Tensor
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
torch_tensor = torch.from_numpy(numpy_array)
print("NumPy array:\n", numpy_array)
print("PyTorch tensor:\n", torch_tensor)

# Tensor to NumPy array
back_to_numpy = torch_tensor.numpy()
print("\nBack to NumPy:\n", back_to_numpy)

# 2. Sharing memory
print("\nMemory sharing demonstration:")
numpy_array[0, 0] = 100
print("Modified NumPy array:\n", numpy_array)
print("PyTorch tensor (shared memory):\n", torch_tensor)

# 3. Working with different data types
print("\nData type handling:")
float_array = np.array([1.1, 2.2, 3.3], dtype=np.float32)
float_tensor = torch.from_numpy(float_array)
print("Float32 NumPy:", float_array.dtype)
print("Float32 PyTorch:", float_tensor.dtype)

# 4. Batch processing with both libraries
print("\nBatch processing example:")
# Create batch using NumPy
numpy_batch = np.random.randn(10, 3, 32, 32)  # Batch of 10 images
torch_batch = torch.from_numpy(numpy_batch)

# Process in PyTorch
torch_processed = torch_batch * 2 + 1
numpy_processed = torch_processed.numpy()

print("Batch shapes:")
print("NumPy:", numpy_batch.shape)
print("PyTorch:", torch_batch.shape) 