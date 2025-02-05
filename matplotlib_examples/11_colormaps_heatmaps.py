import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Create custom colormap
colors = ['darkred', 'red', 'orange', 'yellow', 'white']
n_bins = 100
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Create data for heatmap
data = np.random.normal(0, 1, (20, 20))
correlation_matrix = np.corrcoef(data)

plt.figure(figsize=(15, 5))

# Basic heatmap
plt.subplot(131)
plt.imshow(data, cmap=custom_cmap)
plt.colorbar()
plt.title('Custom Colormap Heatmap')

# Correlation matrix heatmap
plt.subplot(132)
im = plt.imshow(correlation_matrix, cmap='coolwarm')
plt.colorbar(im)
plt.title('Correlation Matrix')

# Annotated heatmap
plt.subplot(133)
small_data = np.random.randint(0, 100, size=(5, 5))
im = plt.imshow(small_data, cmap='YlOrRd')
plt.colorbar(im)

# Add text annotations
for i in range(5):
    for j in range(5):
        text = plt.text(j, i, small_data[i, j],
                       ha="center", va="center", color="black")

plt.title('Annotated Heatmap')
plt.tight_layout()
plt.show() 