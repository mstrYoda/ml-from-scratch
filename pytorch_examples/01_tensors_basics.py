import torch

# Creating tensors
print("Basic Tensor Creation:")
x = torch.tensor([1, 2, 3, 4, 5])
print("From list:", x)

# Different tensor types
float_tensor = torch.FloatTensor([1, 2, 3])
long_tensor = torch.LongTensor([1, 2, 3])
print("\nDifferent types:")
print("Float:", float_tensor)
print("Long:", long_tensor)

# Tensor operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print("\nBasic Operations:")
print("Addition:", a + b)
print("Multiplication:", a * b)
print("Matrix Multiplication:", torch.matmul(a, b))

# Reshaping
c = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("\nReshaping:")
print("Original:", c)
print("Reshaped:", c.reshape(3, 2))
print("Transposed:", c.t())

# GPU Support (if available)
if torch.cuda.is_available():
    device = torch.device("cuda")
    x_gpu = x.to(device)
    print("\nGPU Tensor:", x_gpu) 