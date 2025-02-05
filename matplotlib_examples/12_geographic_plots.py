import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

# Create a simple map-like visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Scatter plot with geographic points
np.random.seed(42)
lats = np.random.uniform(30, 50, 50)
lons = np.random.uniform(-120, -70, 50)
populations = np.random.uniform(100, 1000, 50)

ax1.scatter(lons, lats, s=populations/10, alpha=0.6, 
           c=populations, cmap='viridis')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_title('Geographic Point Data')
ax1.grid(True)

# Custom territory visualization
def draw_region(ax, center, size, color):
    circle = Circle(center, size, fill=False, color=color)
    ax.add_patch(circle)
    rect = Rectangle((center[0]-size/2, center[1]-size/2), 
                    size, size, fill=False, color=color)
    ax.add_patch(rect)

# Add some sample regions
centers = [(0, 0), (1, 1), (-1, -1)]
sizes = [0.5, 0.7, 0.3]
colors = ['red', 'blue', 'green']

for center, size, color in zip(centers, sizes, colors):
    draw_region(ax2, center, size, color)

ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax2.set_title('Custom Territory Visualization')
ax2.grid(True)

plt.tight_layout()
plt.show() 