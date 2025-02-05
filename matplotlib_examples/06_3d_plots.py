import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create data for 3D plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create figure
fig = plt.figure(figsize=(12, 5))

# Surface plot
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('Surface Plot')
fig.colorbar(surf)

# Wire frame plot
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
ax2.set_title('Wireframe Plot')

plt.tight_layout()
plt.show() 