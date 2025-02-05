import matplotlib.pyplot as plt
import numpy as np

# Generate random data
np.random.seed(42)
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
sizes = 1000 * np.random.rand(50)

# Create figure with subplots
plt.figure(figsize=(12, 5))

# Basic scatter plot
plt.subplot(121)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='viridis')
plt.colorbar()
plt.title('Scatter with varying size and color')

# Different marker styles
plt.subplot(122)
markers = ['o', 's', '^', 'v', '<', '>', 'p', '*']
for idx, marker in enumerate(markers):
    plt.scatter(x[idx::8], y[idx::8], marker=marker, label=f'Marker {marker}')
plt.legend()
plt.title('Different Marker Styles')

plt.tight_layout()
plt.show() 