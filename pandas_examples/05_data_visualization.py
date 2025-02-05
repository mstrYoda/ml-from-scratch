import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create sample data
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=100),
    'sales': np.random.normal(100, 20, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'profit': np.random.normal(20, 5, 100)
})

# Basic line plot
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
df.plot(x='date', y='sales', ax=plt.gca(), title='Sales Over Time')

# Box plot
plt.subplot(2, 2, 2)
df.boxplot(column='sales', by='category', ax=plt.gca())
plt.title('Sales Distribution by Category')

# Scatter plot
plt.subplot(2, 2, 3)
df.plot.scatter(x='sales', y='profit', ax=plt.gca(), 
                title='Sales vs Profit')

# Histogram
plt.subplot(2, 2, 4)
df['sales'].hist(bins=20, ax=plt.gca())
plt.title('Sales Distribution')

plt.tight_layout()
plt.show()

# Seaborn visualizations
plt.figure(figsize=(15, 5))

# Kernel Density Plot
plt.subplot(1, 3, 1)
sns.kdeplot(data=df, x='sales', hue='category')
plt.title('Sales Density by Category')

# Violin Plot
plt.subplot(1, 3, 2)
sns.violinplot(data=df, x='category', y='sales')
plt.title('Sales Distribution by Category')

# Heat Map of Correlations
plt.subplot(1, 3, 3)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

plt.tight_layout()
plt.show() 