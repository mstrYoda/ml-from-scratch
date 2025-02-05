import numpy as np

# Trigonometric functions
angles = np.array([0, 30, 45, 60, 90])
radians = np.deg2rad(angles)
print("Sine:", np.sin(radians))
print("Cosine:", np.cos(radians))

# Complex numbers
complex_arr = np.array([1+2j, 3+4j, 5+6j])
print("\nComplex array:", complex_arr)
print("Real part:", complex_arr.real)
print("Imaginary part:", complex_arr.imag)

# Universal functions (ufuncs)
arr = np.array([1, 2, 3, 4])
print("\nLog:", np.log(arr))
print("Cumulative sum:", np.cumsum(arr))
print("Gradient:", np.gradient(arr)) 