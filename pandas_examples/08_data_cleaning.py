import pandas as pd
import numpy as np

# Create messy data
df = pd.DataFrame({
    'name': ['John ', ' Jane', 'Bob', ' Sarah '],
    'age': ['25', '30', 'unknown', '35'],
    'salary': ['50,000', '60,000', '75000', 'NA'],
    'date': ['2023-01-01', '01/15/2023', '2023-02-01', '03-15-2023']
})

print("Original messy DataFrame:")
print(df)

# Clean string columns
df['name'] = df['name'].str.strip()
print("\nCleaned names:")
print(df['name'])

# Convert age to numeric, handling errors
df['age'] = pd.to_numeric(df['age'].replace('unknown', np.nan), errors='coerce')
print("\nCleaned ages:")
print(df['age'])

# Clean and standardize salary
df['salary'] = (df['salary'].replace('NA', np.nan)
                          .str.replace(',', '')
                          .astype(float))
print("\nCleaned salaries:")
print(df['salary'])

# Convert dates to datetime using a more flexible parsing
df['date'] = pd.to_datetime(df['date'], format='mixed')
print("\nStandardized dates:")
print(df['date'])

# Identify and handle duplicates
print("\nDuplicate check:")
print(df.duplicated().sum())

# Value counts and missing values
print("\nMissing values:")
print(df.isnull().sum())

# Data validation
def validate_age(age):
    return 0 <= age <= 120 if pd.notnull(age) else True

print("\nAge validation:")
print(df['age'].apply(validate_age)) 