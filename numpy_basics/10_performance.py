import numpy as np
import time

# Compare Python list vs NumPy array operations
size = 1000000

# Python list
python_list = list(range(size))
start_time = time.time()
python_result = [x ** 2 for x in python_list]
python_time = time.time() - start_time

# NumPy array
numpy_array = np.arange(size)
start_time = time.time()
numpy_result = numpy_array ** 2
numpy_time = time.time() - start_time

print(f"Python time: {python_time:.4f} seconds")
print(f"NumPy time: {numpy_time:.4f} seconds")
print(f"NumPy is {python_time/numpy_time:.1f}x faster")

# Memory views and copying
a = np.array([1, 2, 3])
b = a  # View
c = a.copy()  # Copy
a[0] = 5
print("\nAfter modifying 'a':")
print("a:", a)
print("b (view):", b)
print("c (copy):", c) 