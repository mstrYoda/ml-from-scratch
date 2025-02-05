import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Generate time series data
dates = [datetime.now() + timedelta(days=x) for x in range(100)]
values = np.cumsum(np.random.randn(100)) + 100

plt.figure(figsize=(15, 10))

# Basic time series
plt.subplot(211)
plt.plot(dates, values, 'b-', label='Value')
plt.fill_between(dates, values, min(values), alpha=0.1)
plt.title('Time Series with Trend')
plt.legend()
plt.grid(True)

# Candlestick-like plot
plt.subplot(212)
opens = values[:-1]
closes = values[1:]
highs = np.maximum(opens, closes) + np.random.rand(99)
lows = np.minimum(opens, closes) - np.random.rand(99)

colors = ['green' if close >= open else 'red' 
          for open, close in zip(opens, closes)]

# Plot candlesticks
for i in range(len(opens)):
    plt.vlines(dates[i], lows[i], highs[i], 
               color=colors[i], linewidth=1)
    plt.vlines(dates[i], opens[i], closes[i], 
               color=colors[i], linewidth=4)

plt.title('Candlestick-like Chart')
plt.grid(True)

plt.tight_layout()
plt.show() 