import pandas as pd
import numpy as np

# Create sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [10, 20, 30, np.nan, 50],
    'C': ['a', 'b', 'c', 'd', np.nan]
})

print("Original DataFrame:")
print(df)

# Handling missing values
print("\nHandling Missing Values:")
print("Drop NA rows:")
print(df.dropna())
print("\nFill NA with value:")
print(df.fillna(0))

# Adding/removing columns
df['D'] = df['A'] * 2
print("\nAdded column D:")
print(df)

# Remove column
df_dropped = df.drop('B', axis=1)
print("\nRemoved column B:")
print(df_dropped)

# Sorting
print("\nSorted by column A:")
print(df.sort_values('A'))

# Apply function
def double_value(x):
    return x * 2 if pd.notnull(x) else x

print("\nApply custom function to column A:")
print(df['A'].apply(double_value)) 