import numpy as np

# Save and load arrays
arr = np.array([[1, 2, 3], [4, 5, 6]])
np.save('array.npy', arr)
loaded_arr = np.load('array.npy')
print("Loaded array:\n", loaded_arr)

# Save and load text files
np.savetxt('array.txt', arr)
loaded_txt = np.loadtxt('array.txt')
print("\nLoaded from text:\n", loaded_txt)

# Structured arrays
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]
values = [('Alice', 25, 1.75),
          ('Bob', 30, 1.85),
          ('Charlie', 35, 1.79)]
structured_arr = np.array(values, dtype=dtype)
print("\nStructured array:\n", structured_arr)
print("Names:", structured_arr['name']) 