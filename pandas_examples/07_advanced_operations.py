import pandas as pd
import numpy as np

# Create sample data
df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'C'],
    'subcategory': ['X', 'Y', 'X', 'Y', 'X'],
    'value1': np.random.randn(5),
    'value2': np.random.randn(5)
})

print("Original DataFrame:")
print(df)

# Pivot tables
print("\nPivot Table:")
pivot = pd.pivot_table(df, 
                      values='value1',
                      index='category',
                      columns='subcategory',
                      aggfunc='mean')
print(pivot)

# Cross tabulation
print("\nCross Tabulation:")
print(pd.crosstab(df['category'], df['subcategory']))

# Multi-level indexing
df.set_index(['category', 'subcategory'], inplace=True)
print("\nMulti-level indexed DataFrame:")
print(df)

# Advanced selection
print("\nSelecting specific level:")
print(df.xs('X', level='subcategory'))

# Unstacking levels
print("\nUnstacked DataFrame:")
print(df.unstack(level='subcategory'))

# Complex transformations
def custom_transform(group):
    return (group - group.mean()) / group.std()

print("\nCustom group transformation:")
print(df.groupby(level='category').transform(custom_transform)) 