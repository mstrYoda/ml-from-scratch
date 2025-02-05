import matplotlib.pyplot as plt
import numpy as np

# Create polar plot
plt.figure(figsize=(12, 5))

# Polar plot
plt.subplot(121, projection='polar')
theta = np.linspace(0, 2*np.pi, 100)
r = 2*np.cos(4*theta)
plt.plot(theta, r)
plt.title('Polar Plot')

# Custom visualization (e.g., radar chart)
plt.subplot(122)
categories = ['A', 'B', 'C', 'D', 'E']
values = [4, 3, 5, 2, 4]
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
values = np.concatenate((values, [values[0]]))  # complete the circle
angles = np.concatenate((angles, [angles[0]]))  # complete the circle

ax = plt.subplot(122, projection='polar')
ax.plot(angles, values)
ax.fill(angles, values, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.title('Radar Chart')

plt.tight_layout()
plt.show() 