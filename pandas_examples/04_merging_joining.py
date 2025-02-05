import pandas as pd

# Create sample DataFrames
df1 = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'name': ['John', 'Anna', 'Peter', 'Linda'],
    'age': [28, 22, 35, 32]
})

df2 = pd.DataFrame({
    'id': [1, 2, 3, 5],
    'city': ['New York', 'Paris', 'London', 'Berlin'],
    'salary': [50000, 45000, 65000, 55000]
})

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# Different types of joins
print("\nInner Join:")
print(pd.merge(df1, df2, on='id', how='inner'))

print("\nLeft Join:")
print(pd.merge(df1, df2, on='id', how='left'))

print("\nRight Join:")
print(pd.merge(df1, df2, on='id', how='right'))

print("\nOuter Join:")
print(pd.merge(df1, df2, on='id', how='outer'))

# Concatenation
print("\nConcatenation (Vertical):")
print(pd.concat([df1, df1]))

print("\nConcatenation (Horizontal):")
print(pd.concat([df1, df2], axis=1)) 