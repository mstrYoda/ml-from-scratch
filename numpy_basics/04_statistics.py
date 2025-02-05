import numpy as np

# Create sample data
data = np.array([14, 23, 15, 42, 35, 28, 20, 33])

print("Data:", data)
print("\nBasic Statistics:")
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Standard deviation:", np.std(data))
print("Variance:", np.var(data))

# Generate random data
random_data = np.random.normal(0, 1, 1000)  # mean=0, std=1, size=1000
print("\nRandom Normal Distribution:")
print("Mean:", np.mean(random_data))
print("Std:", np.std(random_data)) 