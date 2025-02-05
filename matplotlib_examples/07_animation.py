import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Create figure and axis
fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))

# Animation update function
def animate(frame):
    line.set_ydata(np.sin(x + frame/10.0))
    return line,

# Create animation
ani = animation.FuncAnimation(
    fig, animate, frames=100, 
    interval=50, blit=True
)

ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1.1, 1.1)
plt.title('Animated Sine Wave')
plt.show() 