import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('ggplot')

# Create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create figure with custom style
fig, ax = plt.subplots(figsize=(10, 6))

# Plot with custom styling
ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
ax.fill_between(x, y, alpha=0.3)

# Customize axes
ax.set_title('Styled Plot Example', fontsize=16, pad=20)
ax.set_xlabel('X-axis', fontsize=12)
ax.set_ylabel('Y-axis', fontsize=12)

# Add grid with custom style
ax.grid(True, linestyle='--', alpha=0.7)

# Customize spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add legend
ax.legend(frameon=True, facecolor='white', framealpha=1)

plt.show() 