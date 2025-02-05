import torch
import numpy as np
import time

def compare_operations():
    # Create large arrays/tensors
    size = 1000000
    numpy_array = np.random.randn(size)
    torch_tensor = torch.from_numpy(numpy_array)
    
    # NumPy operations
    start_time = time.time()
    numpy_result = np.exp(numpy_array) + np.sin(numpy_array)
    numpy_time = time.time() - start_time
    
    # PyTorch CPU operations
    start_time = time.time()
    torch_result = torch.exp(torch_tensor) + torch.sin(torch_tensor)
    torch_time = time.time() - start_time
    
    print(f"NumPy time: {numpy_time:.4f} seconds")
    print(f"PyTorch CPU time: {torch_time:.4f} seconds")
    
    # PyTorch GPU operations (if available)
    if torch.cuda.is_available():
        torch_tensor_gpu = torch_tensor.cuda()
        start_time = time.time()
        torch_result_gpu = torch.exp(torch_tensor_gpu) + torch.sin(torch_tensor_gpu)
        torch_gpu_time = time.time() - start_time
        print(f"PyTorch GPU time: {torch_gpu_time:.4f} seconds")

# Run performance comparison
print("Performance Comparison:")
compare_operations()

# Memory usage comparison
def compare_memory():
    sizes = [1000, 10000, 50000]
    
    for size in sizes:
        print(f"\nSize: {size}")
        # NumPy array
        numpy_array = np.random.randn(size, size)
        numpy_memory = numpy_array.nbytes / 1024 / 1024  # MB
        
        # PyTorch tensor
        torch_tensor = torch.from_numpy(numpy_array)
        torch_memory = torch_tensor.element_size() * torch_tensor.nelement() / 1024 / 1024  # MB
        
        print(f"NumPy memory: {numpy_memory:.2f} MB")
        print(f"PyTorch memory: {torch_memory:.2f} MB")

# Run memory comparison
print("\nMemory Usage Comparison:")
compare_memory() 