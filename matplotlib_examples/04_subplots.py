import matplotlib.pyplot as plt
import numpy as np

# Create data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = x**2

# Create subplots using different methods
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1
ax1.plot(x, y1)
ax1.set_title('Sine')
ax1.set_xlabel('x')
ax1.set_ylabel('sin(x)')

# Plot 2
ax2.plot(x, y2, 'r-')
ax2.set_title('Cosine')
ax2.set_xlabel('x')
ax2.set_ylabel('cos(x)')

# Plot 3
ax3.plot(x, y3, 'g-')
ax3.set_title('Tangent')
ax3.set_xlabel('x')
ax3.set_ylabel('tan(x)')

# Plot 4
ax4.plot(x, y4, 'm-')
ax4.set_title('Quadratic')
ax4.set_xlabel('x')
ax4.set_ylabel('x²')

plt.tight_layout()
plt.show() 