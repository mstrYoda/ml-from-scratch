import pandas as pd
import numpy as np

# Creating DataFrames
print("Creating DataFrames:")
# From dictionary
data_dict = {
    'name': ['John', 'Anna', 'Peter', 'Linda'],
    'age': [28, 22, 35, 32],
    'city': ['New York', 'Paris', 'London', 'Tokyo']
}
df1 = pd.DataFrame(data_dict)
print("\nFrom dictionary:")
print(df1)

# From list of lists
data_list = [
    ['John', 28, 'New York'],
    ['Anna', 22, 'Paris'],
    ['Peter', 35, 'London']
]
df2 = pd.DataFrame(data_list, columns=['name', 'age', 'city'])
print("\nFrom list:")
print(df2)

# Basic operations
print("\nBasic Operations:")
print("Head of DataFrame:", df1.head(2))
print("\nDataFrame Info:")
print(df1.info())
print("\nDataFrame Description:")
print(df1.describe())

# Accessing data
print("\nAccessing Data:")
print("Single column:", df1['name'])
print("\nMultiple columns:")
print(df1[['name', 'age']])

# Basic filtering
print("\nFiltering:")
print("Age > 30:")
print(df1[df1['age'] > 30]) 