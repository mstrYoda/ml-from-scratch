import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons

# Create data
t = np.linspace(0, 10, 1000)
initial_freq = 3.0
initial_amp = 1.0

# Create figure and plot
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(t, initial_amp * np.sin(2 * np.pi * initial_freq * t))
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')

# Add sliders
ax_freq = plt.axes([0.2, 0.02, 0.6, 0.03])
ax_amp = plt.axes([0.2, 0.06, 0.6, 0.03])
freq_slider = Slider(ax_freq, 'Frequency', 0.1, 10.0, valinit=initial_freq)
amp_slider = Slider(ax_amp, 'Amplitude', 0.1, 2.0, valinit=initial_amp)

def update(val):
    freq = freq_slider.val
    amp = amp_slider.val
    line.set_ydata(amp * np.sin(2 * np.pi * freq * t))
    fig.canvas.draw_idle()

freq_slider.on_changed(update)
amp_slider.on_changed(update)

plt.show() 