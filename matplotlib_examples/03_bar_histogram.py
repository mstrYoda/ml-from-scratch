import matplotlib.pyplot as plt
import numpy as np

# Bar chart
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
errors = [3, 5, 2, 4]

plt.figure(figsize=(12, 5))

# Simple bar chart
plt.subplot(121)
plt.bar(categories, values, yerr=errors, capsize=5)
plt.title('Bar Chart with Error Bars')

# Histogram
plt.subplot(122)
data = np.random.normal(100, 15, 1000)
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show() 