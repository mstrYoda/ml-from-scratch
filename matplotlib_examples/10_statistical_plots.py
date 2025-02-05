import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# Create multiple datasets
np.random.seed(42)
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(2, 1.5, 1000)

plt.figure(figsize=(15, 5))

# Box Plot
plt.subplot(131)
plt.boxplot([data1, data2], labels=['Data 1', 'Data 2'])
plt.title('Box Plot')

# Violin Plot
plt.subplot(132)
plt.violinplot([data1, data2])
plt.title('Violin Plot')

# 2D Density Plot with confidence ellipses
ax = plt.subplot(133)
x = np.random.normal(0, 1, 1000)
y = x * 0.5 + np.random.normal(0, 0.5, 1000)

# Calculate the density
plt.scatter(x, y, alpha=0.5)

# Add confidence ellipses
cov = np.cov(x, y)
lambda_, v = np.linalg.eig(cov)
angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))

for n_std in [1, 2, 3]:
    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=lambda_[0]**0.5 * n_std * 2,
                      height=lambda_[1]**0.5 * n_std * 2,
                      angle=angle,
                      facecolor='none',
                      edgecolor='red',
                      alpha=0.3)
    ax.add_patch(ellipse)

plt.title('2D Density with\nConfidence Ellipses')
plt.tight_layout()
plt.show() 